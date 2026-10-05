"""Clustered localized MKL for mass, neutral-loss, and fragment-ion evidence.

Clustering is target-free.  Formula folds 0--1 learn cluster weights, fold 2
selects shrinkage toward the global kernel, fold 3 is an already-used inner
evaluation, and fold 4 remains sealed.  Every deployed weight is a function of
one clean spectrum and therefore admits an explicit shared feature map.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from audit_chemaware_counterfactual_rule_kernel import (
    bootstrap,
    build_feature_views,
    rank_queries,
    retrieval,
    sha256,
)
from audit_chemaware_counterfactual_rule_kernel_natural import (
    paired_rank_comparison,
    query_reference_dot,
    subset_graph,
)
from audit_chemaware_localized_typed_rule_kernel import (
    fixed_projection,
    spectrum_statistics,
    standardize_on_training_nodes,
)
from audit_chemaware_mass_kernel_embedding import KernelCache
from chemaware_iceberg_direct_core import stable_formula_folds
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_clustered_local_mkl_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--coordinate-sweeps", type=int, default=2)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument(
        "--shrinkage", type=float, nargs="+", default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-offset", type=float, default=0.137)
    parser.add_argument("--motif-dimension", type=int, default=64)
    parser.add_argument("--motif-top-rules", type=int, default=12)
    parser.add_argument("--official-projection-dim", type=int, default=16)
    parser.add_argument("--mass-projection-dim", type=int, default=8)
    parser.add_argument("--rule-projection-dim", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def projected(values: np.ndarray, output_dim: int, seed: int, batch: int = 4096) -> np.ndarray:
    matrix = fixed_projection(values.shape[1], output_dim, seed)
    output = np.empty((len(values), output_dim), dtype=np.float32)
    for left in range(0, len(values), batch):
        right = min(left + batch, len(values))
        output[left:right] = np.asarray(values[left:right], dtype=np.float32) @ matrix
    return output


def query_reference_dot_float32(
    feature: np.ndarray,
    query_node: np.ndarray,
    reference_node: np.ndarray,
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
) -> np.ndarray:
    """Accumulate float16 caches in float32 without materializing a float32 matrix."""

    output = np.empty(len(reference_node), dtype=np.float64)
    for query in range(len(query_node)):
        molecule_left, molecule_right = query_ptr[query : query + 2]
        pair_left = int(molecule_ptr[int(molecule_left)])
        pair_right = int(molecule_ptr[int(molecule_right)])
        output[pair_left:pair_right] = (
            np.asarray(feature[reference_node[pair_left:pair_right]], dtype=np.float32)
            @ np.asarray(feature[query_node[query]], dtype=np.float32)
        )
    return output


def cluster_features(
    official: np.ndarray,
    mass: np.ndarray,
    views: dict[str, np.ndarray],
    spectral: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    scalar = np.column_stack((
        views["gate_raw_neutral_loss_energy"][:, 0],
        views["gate_raw_fragment_ion_energy"][:, 0],
        views["gate_background_neutral_loss_energy"][:, 0],
        views["gate_background_fragment_ion_energy"][:, 0],
        views["gate_raw_neutral_loss_count"][:, 0],
        views["gate_raw_fragment_ion_count"][:, 0],
        np.sum(views["raw_neutral_loss"] * views["background_neutral_loss"], axis=1),
        np.sum(views["raw_fragment_ion"] * views["background_fragment_ion"], axis=1),
        spectral,
    )).astype(np.float32)
    return np.concatenate((
        projected(official, args.official_projection_dim, args.seed + 11),
        projected(mass, args.mass_projection_dim, args.seed + 19),
        projected(views["raw_neutral_loss"], args.rule_projection_dim, args.seed + 23),
        projected(views["raw_fragment_ion"], args.rule_projection_dim, args.seed + 37),
        scalar,
    ), axis=1)


def ranks_for_weights(
    official_pair: np.ndarray,
    kernel_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
    node_cluster: np.ndarray,
    pair_query_node: np.ndarray,
    reference_node: np.ndarray,
    compact: dict[str, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    query_cluster = node_cluster[pair_query_node]
    reference_cluster = node_cluster[reference_node]
    score = np.array(official_pair, copy=True)
    for channel, pair in enumerate(kernel_pair):
        coefficient = np.sqrt(
            weights[query_cluster, channel] * weights[reference_cluster, channel],
        )
        score += coefficient * pair
    molecule = np.maximum.reduceat(score, compact["molecule_ptr"][:-1])
    return rank_queries(molecule, compact["query_ptr"], compact["molecule_label"])


def objective(baseline: np.ndarray, rank: np.ndarray) -> tuple[int, int, float]:
    metric = retrieval(baseline, rank)
    return (
        int(metric["risk_utility_at_1"]), -int(metric["introduced_at_1"]),
        float(metric["delta_mrr"]),
    )


def optimize_global(
    official_pair: np.ndarray,
    kernel_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
    node_cluster: np.ndarray,
    pair_query_node: np.ndarray,
    reference_node: np.ndarray,
    compact: dict[str, np.ndarray],
    baseline: np.ndarray,
    train_index: np.ndarray,
    beta: tuple[float, ...] | list[float],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rows = []
    best = None
    best_key = None
    for mass_beta in beta:
        for nl_beta in beta:
            for cf_beta in beta:
                weights = np.tile((mass_beta, nl_beta, cf_beta), (node_cluster.max() + 1, 1)).astype(np.float64)
                rank = ranks_for_weights(
                    official_pair, kernel_pair, node_cluster, pair_query_node,
                    reference_node, compact, weights,
                )
                metric = retrieval(baseline[train_index], rank[train_index])
                row = {
                    "mass_beta": float(mass_beta), "neutral_loss_beta": float(nl_beta),
                    "fragment_ion_beta": float(cf_beta), **metric,
                }
                rows.append(row)
                key = (
                    int(metric["risk_utility_at_1"]), -int(metric["introduced_at_1"]),
                    float(metric["delta_mrr"]),
                    -(float(mass_beta) + float(nl_beta) + float(cf_beta)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = np.asarray((mass_beta, nl_beta, cf_beta), dtype=np.float64)
    if best is None:
        raise RuntimeError("global kernel grid is empty")
    return best, rows


def coordinate_optimize(
    initial: np.ndarray,
    official_pair: np.ndarray,
    kernel_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
    node_cluster: np.ndarray,
    pair_query_node: np.ndarray,
    reference_node: np.ndarray,
    compact: dict[str, np.ndarray],
    baseline: np.ndarray,
    train_index: np.ndarray,
    beta: tuple[float, ...] | list[float],
    sweeps: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    weights = np.tile(initial, (node_cluster.max() + 1, 1)).astype(np.float64)
    history = []
    for sweep in range(sweeps):
        changes = 0
        for cluster in range(len(weights)):
            for channel in range(weights.shape[1]):
                candidates = []
                for value in beta:
                    trial = weights.copy(); trial[cluster, channel] = float(value)
                    rank = ranks_for_weights(
                        official_pair, kernel_pair, node_cluster, pair_query_node,
                        reference_node, compact, trial,
                    )
                    metric = retrieval(baseline[train_index], rank[train_index])
                    candidates.append({
                        "value": float(value), **metric,
                        "key": [int(metric["risk_utility_at_1"]), -int(metric["introduced_at_1"]),
                                float(metric["delta_mrr"]), -float(value)],
                    })
                selected = max(candidates, key=lambda row: tuple(row["key"]))
                previous = float(weights[cluster, channel])
                weights[cluster, channel] = float(selected["value"])
                changes += int(previous != weights[cluster, channel])
                history.append({
                    "sweep": sweep, "cluster": cluster, "channel": channel,
                    "channel_name": ("mass", "neutral_loss", "fragment_ion")[channel],
                    "previous": previous, "selected": float(selected["value"]),
                    "selection_metric": {key: value for key, value in selected.items() if key != "key"},
                })
        if changes == 0:
            break
    return weights, history


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    train_pool = np.flatnonzero((fold == 0) | (fold == 1))
    validation_pool = np.flatnonzero(fold == 2)
    inner_pool = np.flatnonzero(fold == 3)
    outer_pool = np.flatnonzero(fold == 4)
    train = identity_balanced_queries(
        train_pool, body["query_ik14"], np.random.default_rng(args.sampling_seed + 1),
        args.train_identities,
    )
    validation = identity_balanced_queries(
        validation_pool, body["query_ik14"], np.random.default_rng(args.sampling_seed + 2),
        args.validation_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.sampling_seed + 19),
        args.max_inner_identities,
    )
    formula_sets = [set(body["query_formula"][x].astype(str)) for x in (train, validation, inner)]
    if any(formula_sets[i] & formula_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    selected_queries = np.concatenate((train, validation, inner))
    compact = subset_graph(body, selected_queries)
    node_rows = np.unique(np.concatenate((compact["query_row"], compact["reference_row"]))).astype(np.int64)
    node_position = {int(row): index for index, row in enumerate(node_rows)}
    query_node = np.asarray([node_position[int(row)] for row in compact["query_row"]], dtype=np.int64)
    reference_node = np.asarray([node_position[int(row)] for row in compact["reference_row"]], dtype=np.int64)
    molecule_query = np.repeat(np.arange(len(selected_queries)), np.diff(compact["query_ptr"]))
    pair_molecule = np.repeat(np.arange(len(compact["molecule_label"])), np.diff(compact["molecule_ptr"]))
    pair_query = molecule_query[pair_molecule]
    pair_query_node = query_node[pair_query]

    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    official_cache = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    positions = np.asarray([cache_position[int(row)] for row in node_rows], dtype=np.int64)
    official = np.asarray(official_cache[positions], dtype=np.float32)
    views, feature_report = build_feature_views(node_rows, cache_position, SimpleNamespace(**vars(args)))
    mass_cache = KernelCache(args, cache_position, variants=("mass",))
    mass = np.empty((len(node_rows), args.kernel_dim), dtype=np.float16)
    for index, row in enumerate(node_rows):
        mass[index] = mass_cache.get(int(row))["mass"]
        if (index + 1) % 4096 == 0:
            print(f"mass features {index + 1}/{len(node_rows)}", flush=True)
    mass_cache.cache.clear()
    spectral = spectrum_statistics(node_rows, cache_position, args.token_dir)

    train_molecule_end = int(compact["query_ptr"][len(train)])
    train_pair_end = int(compact["molecule_ptr"][train_molecule_end])
    training_nodes = np.unique(np.concatenate((query_node[:len(train)], reference_node[:train_pair_end])))
    clean_feature = cluster_features(official, mass, views, spectral, args)
    clean_feature, scaler = standardize_on_training_nodes(clean_feature, training_nodes)
    clustering = MiniBatchKMeans(
        n_clusters=args.clusters, random_state=args.seed, n_init=10,
        batch_size=4096, max_iter=200, reassignment_ratio=0.0,
    )
    clustering.fit(clean_feature[training_nodes])
    node_cluster = clustering.predict(clean_feature).astype(np.int64)
    cluster_counts = np.bincount(node_cluster, minlength=args.clusters)
    train_cluster_counts = np.bincount(node_cluster[training_nodes], minlength=args.clusters)
    if np.any(train_cluster_counts < 100):
        raise RuntimeError("target-free cluster contains too few training spectra")

    official_pair = query_reference_dot(
        official, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    mass_pair = query_reference_dot_float32(
        mass, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    nl_pair = query_reference_dot(
        views["raw_neutral_loss"], query_node, reference_node,
        compact["query_ptr"], compact["molecule_ptr"],
    )
    cf_pair = query_reference_dot(
        views["raw_fragment_ion"], query_node, reference_node,
        compact["query_ptr"], compact["molecule_ptr"],
    )
    kernels = (mass_pair, nl_pair, cf_pair)
    baseline_molecule = np.maximum.reduceat(official_pair, compact["molecule_ptr"][:-1])
    baseline = rank_queries(baseline_molecule, compact["query_ptr"], compact["molecule_label"])
    train_index = np.arange(len(train), dtype=np.int64)
    validation_index = np.arange(len(train), len(train) + len(validation), dtype=np.int64)
    inner_index = np.arange(len(train) + len(validation), len(selected_queries), dtype=np.int64)

    global_weight, global_grid = optimize_global(
        official_pair, kernels, node_cluster, pair_query_node, reference_node,
        compact, baseline, train_index, args.beta,
    )
    learned_weight, coordinate_history = coordinate_optimize(
        global_weight, official_pair, kernels, node_cluster, pair_query_node,
        reference_node, compact, baseline, train_index, args.beta, args.coordinate_sweeps,
    )
    shrinkage_rows = []
    shrinkage_ranks = {}
    for shrinkage in args.shrinkage:
        weight = global_weight[None, :] + float(shrinkage) * (learned_weight - global_weight[None, :])
        rank = ranks_for_weights(
            official_pair, kernels, node_cluster, pair_query_node, reference_node, compact, weight,
        )
        shrinkage_ranks[float(shrinkage)] = rank
        shrinkage_rows.append({
            "shrinkage": float(shrinkage),
            **retrieval(baseline[validation_index], rank[validation_index]),
        })
    selected_shrinkage = max(
        shrinkage_rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), -float(row["shrinkage"]),
        ),
    )
    shrinkage = float(selected_shrinkage["shrinkage"])
    selected_weight = global_weight[None, :] + shrinkage * (learned_weight - global_weight[None, :])
    selected_rank = shrinkage_ranks[shrinkage]
    global_rank = ranks_for_weights(
        official_pair, kernels, node_cluster, pair_query_node, reference_node,
        compact, np.tile(global_weight, (args.clusters, 1)),
    )
    inner_formula = body["query_formula"][inner].astype(str)
    selected_inner = retrieval(baseline[inner_index], selected_rank[inner_index])
    global_inner = retrieval(baseline[inner_index], global_rank[inner_index])
    absolute_ci = bootstrap(
        inner_formula, baseline[inner_index], selected_rank[inner_index],
        args.bootstrap_draws, args.seed + 500,
    )
    versus_global = paired_rank_comparison(
        selected_rank[inner_index], global_rank[inner_index], inner_formula,
        draws=args.bootstrap_draws, seed=args.seed + 600,
    )
    report = {
        "status": "CHEMAWARE_CLUSTERED_LOCAL_MKL_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 weight learning; fold 2 shrinkage selection; used inner fold 3 evaluation; fold 4 sealed",
        "claim_limit": "Frozen shared-feature development audit; inner fold is not external confirmation.",
        "data": {
            "train_queries": int(len(train)), "validation_queries": int(len(validation)),
            "inner_queries": int(len(inner)), "outer_queries_untouched": int(len(outer_pool)),
            "formula_overlap": 0, "unique_spectrum_rows": int(len(node_rows)),
            "reference_pairs": int(len(reference_node)),
        },
        "method": {
            "shared_map": "[official, sqrt(beta_mass(cluster(x)))*mass, sqrt(beta_nl(cluster(x)))*NL, sqrt(beta_cf(cluster(x)))*CF]",
            "cluster_input": "one unmodified spectrum only; target-free MiniBatchKMeans",
            "candidate_scores_ranks_identity_formula_used_by_cluster": False,
            "clusters": int(args.clusters), "channels": ["mass", "neutral_loss", "fragment_ion"],
            "cluster_counts": cluster_counts.tolist(),
            "training_cluster_counts": train_cluster_counts.tolist(),
            "global_weight": global_weight.tolist(),
            "learned_cluster_weight": learned_weight.tolist(),
            "selected_shrinkage": shrinkage,
            "selected_cluster_weight": selected_weight.tolist(),
            "embedding_dimension": int(official.shape[1] + mass.shape[1]
                                       + views["raw_neutral_loss"].shape[1]
                                       + views["raw_fragment_ion"].shape[1]),
        },
        "features": feature_report,
        "global_grid_on_training": global_grid,
        "coordinate_history_on_training": coordinate_history,
        "shrinkage_selection_on_validation": shrinkage_rows,
        "held_inner": {"clustered": selected_inner, "global": global_inner},
        "held_inner_absolute_formula_bootstrap_ci95": absolute_ci,
        "paired_inner": {"clustered_minus_global": versus_global},
        "gates": {
            "validation_selected_nonzero_localization": shrinkage > 0,
            "absolute_inner_ci_positive": absolute_ci[0] > 0,
            "increment_over_global_ci_positive": versus_global[
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "outer_fold_untouched": True,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
            "scaler": scaler,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_clustered_mkl_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=inner_formula,
            baseline_rank=baseline[inner_index], global_rank=global_rank[inner_index],
            clustered_rank=selected_rank[inner_index],
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "global_weight": global_weight.tolist(),
        "learned_cluster_weight": learned_weight.tolist(),
        "selected_shrinkage": shrinkage, "global": global_inner,
        "clustered": selected_inner, "paired_inner": report["paired_inner"],
        "gates": report["gates"], "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
