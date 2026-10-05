"""Natural-prevalence development audit for ChemAware rule co-occurrence kernels.

Fusion weights are selected on a 2,048-identity sample from formula folds 0--2
at their natural error rate, then evaluated once on every identity in the
already-used inner fold 3.  Fold 4 remains untouched.  This is a frozen shared
feature audit; no DreaMS parameters are updated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_counterfactual_rule_kernel import (
    bootstrap,
    build_feature_views,
    paired_dot,
    rank_queries,
    retrieval,
    sha256,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ("raw_rule_motif", "counterfactual_motif")
CONTROLS = (
    "raw_rule", "raw_motif_only", "local_background_motif_only",
    "row_permuted_rule_motif_only", "local_background", "row_permuted_rule",
    "raw_neutral_loss", "raw_fragment_ion",
    "background_neutral_loss", "background_fragment_ion",
    "row_permuted_neutral_loss", "row_permuted_fragment_ion",
)


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
        default=ROOT / "data/validation/chemaware_counterfactual_rule_kernel_natural_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--discovery-natural-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--counterfactual-offset", type=float, default=0.137)
    parser.add_argument("--motif-dimension", type=int, default=256)
    parser.add_argument("--motif-top-rules", type=int, default=12)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def subset_graph(body: dict[str, np.ndarray], queries: np.ndarray) -> dict[str, np.ndarray]:
    query_rows = []
    query_ptr = [0]
    molecule_ptr = [0]
    labels = []
    reference_rows = []
    for query in np.asarray(queries, dtype=np.int64):
        query_rows.append(int(body["query_row"][query]))
        left, right = map(int, body["query_ptr"][query : query + 2])
        for molecule in range(left, right):
            labels.append(int(body["molecule_label"][molecule]))
            pair_left, pair_right = map(int, body["molecule_ptr"][molecule : molecule + 2])
            reference_rows.extend(map(int, body["pair_candidate_row"][pair_left:pair_right]))
            molecule_ptr.append(len(reference_rows))
        query_ptr.append(len(labels))
    return {
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "molecule_label": np.asarray(labels, dtype=bool),
        "reference_row": np.asarray(reference_rows, dtype=np.int64),
    }


def paired_rank_comparison(
    left_rank: np.ndarray,
    right_rank: np.ndarray,
    formula: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    # Reuse bootstrap by treating the right method as the baseline.
    result = retrieval(right_rank, left_rank)
    result["formula_cluster_bootstrap_delta_recall1_ci95"] = bootstrap(
        formula, right_rank, left_rank, draws, seed,
    )
    return result


def query_reference_dot(
    feature: np.ndarray,
    query_node: np.ndarray,
    reference_node: np.ndarray,
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
) -> np.ndarray:
    """Match the frozen ledger's per-query matrix-vector accumulation order."""

    output = np.empty(len(reference_node), dtype=np.float64)
    for query in range(len(query_node)):
        molecule_left, molecule_right = query_ptr[query : query + 2]
        pair_left = int(molecule_ptr[int(molecule_left)])
        pair_right = int(molecule_ptr[int(molecule_right)])
        output[pair_left:pair_right] = (
            feature[reference_node[pair_left:pair_right]] @ feature[query_node[query]]
        )
    return output


def exact_pair_motif_kernel(
    base_feature: np.ndarray,
    query_node: np.ndarray,
    reference_node: np.ndarray,
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
) -> np.ndarray:
    """Normalized exact degree-two kernel over all distinct rule pairs.

    For explicit features ``p_ab(x)=r_a(x)r_b(x), a<b``, the dot product is
    ``0.5 * ((r.s)^2 - (r^2.s^2))``.  This avoids a 49,770-column materialized
    embedding and removes every feature-hash collision while remaining PSD.
    """

    r = np.asarray(base_feature, dtype=np.float32)
    dot = query_reference_dot(r, query_node, reference_node, query_ptr, molecule_ptr)
    squared_dot = query_reference_dot(
        r * r, query_node, reference_node, query_ptr, molecule_ptr,
    )
    sum_square = np.sum(r * r, axis=1, dtype=np.float64)
    sum_fourth = np.sum((r * r) ** 2, axis=1, dtype=np.float64)
    norm = np.sqrt(np.maximum(0.5 * (sum_square * sum_square - sum_fourth), 0.0))
    molecule_query = np.repeat(np.arange(len(query_node)), np.diff(query_ptr))
    pair_molecule = np.repeat(np.arange(len(molecule_ptr) - 1), np.diff(molecule_ptr))
    pair_query = molecule_query[pair_molecule]
    denominator = norm[query_node[pair_query]] * norm[reference_node]
    numerator = 0.5 * (dot * dot - squared_dot)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def main() -> None:
    args = arguments()
    if len({args.inner_fold, args.outer_fold}) != 2:
        raise ValueError("inner and outer folds overlap")
    if args.discovery_natural_identities <= 0 or args.max_inner_identities < 0:
        raise ValueError("invalid identity limits")
    required = [
        args.manifest, args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.rule_library,
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    train_pool = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer_pool = np.flatnonzero(fold == args.outer_fold)
    discovery = identity_balanced_queries(
        train_pool, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_inner_identities,
    )
    if set(body["query_formula"][discovery].astype(str)) & set(body["query_formula"][inner].astype(str)):
        raise RuntimeError("discovery and inner formulas overlap")
    selected_queries = np.concatenate((discovery, inner))
    compact = subset_graph(body, selected_queries)
    node_rows = np.unique(np.concatenate((compact["query_row"], compact["reference_row"]))).astype(np.int64)
    node_position = {int(row): index for index, row in enumerate(node_rows)}
    query_node = np.asarray([node_position[int(row)] for row in compact["query_row"]], dtype=np.int64)
    reference_node = np.asarray([node_position[int(row)] for row in compact["reference_row"]], dtype=np.int64)
    molecule_query = np.repeat(np.arange(len(selected_queries)), np.diff(compact["query_ptr"]))
    pair_molecule = np.repeat(np.arange(len(compact["molecule_label"])), np.diff(compact["molecule_ptr"]))
    pair_query = molecule_query[pair_molecule]

    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    official_cache = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    absent = [int(row) for row in node_rows if int(row) not in cache_position]
    if absent:
        raise RuntimeError(f"{len(absent)} required rows are absent from the cache")
    embedding = np.asarray(
        official_cache[[cache_position[int(row)] for row in node_rows]], dtype=np.float32,
    )
    feature_args = SimpleNamespace(**vars(args))
    views, feature_report = build_feature_views(node_rows, cache_position, feature_args)
    requested = PRIMARY + CONTROLS
    views = {name: views[name] for name in requested}

    official_pair = query_reference_dot(
        embedding, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    official_molecule = np.maximum.reduceat(official_pair, compact["molecule_ptr"][:-1])
    baseline_rank = rank_queries(
        official_molecule, compact["query_ptr"], compact["molecule_label"],
    )
    discovery_local = np.arange(len(discovery), dtype=np.int64)
    inner_local = np.arange(len(discovery), len(selected_queries), dtype=np.int64)
    inner_formula = body["query_formula"][inner].astype(str)

    reports = {}; inner_ranks = {}; kernel_pairs = {}

    def evaluate_kernel(name: str, dimension: int, kernel_pair: np.ndarray, seed_offset: int) -> None:
        ranks = {}
        selection = []
        for beta in args.beta:
            molecule_score = np.maximum.reduceat(
                official_pair + float(beta) * kernel_pair, compact["molecule_ptr"][:-1],
            )
            ranks[float(beta)] = rank_queries(
                molecule_score, compact["query_ptr"], compact["molecule_label"],
            )
            metric = retrieval(
                baseline_rank[discovery_local], ranks[float(beta)][discovery_local],
            )
            selection.append({"beta": float(beta), **metric})
        selected = max(
            selection,
            key=lambda item: (
                int(item["risk_utility_at_1"]), -int(item["introduced_at_1"]),
                float(item["delta_mrr"]), -float(item["beta"]),
            ),
        )
        beta = float(selected["beta"])
        held_rank = ranks[beta][inner_local]
        held_rank_by_beta = np.stack(
            [ranks[float(candidate)][inner_local] for candidate in args.beta], axis=1,
        )
        # This minimum observes the answer and is therefore only an action-space
        # ceiling.  Including beta=0 makes it a no-op-aware upper bound: it can
        # never turn an official top-1 success into a failure.
        held_oracle_rank = np.min(held_rank_by_beta, axis=1)
        inner_ranks[name] = held_rank
        reports[name] = {
            "dimension": int(dimension),
            "selection_on_natural_training_formulas": selection,
            "selected_beta": beta,
            "held_inner_all_beta": [
                {
                    "beta": float(candidate),
                    **retrieval(
                        baseline_rank[inner_local],
                        ranks[float(candidate)][inner_local],
                    ),
                }
                for candidate in args.beta
            ],
            "held_inner_no_op_aware_beta_oracle": {
                **retrieval(baseline_rank[inner_local], held_oracle_rank),
                "claim_limit": (
                    "Per-query beta is selected after observing the true rank; this is "
                    "headroom only and is not a deployable router or model result."
                ),
            },
            "held_inner": retrieval(baseline_rank[inner_local], held_rank),
            "held_inner_formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
                inner_formula, baseline_rank[inner_local], held_rank,
                args.bootstrap_draws, args.seed + seed_offset,
            ),
        }

    for variant_index, (name, feature) in enumerate(views.items()):
        kernel_pair = query_reference_dot(
            feature, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
        )
        kernel_pairs[name] = kernel_pair
        evaluate_kernel(name, feature.shape[1], kernel_pair, 100 + variant_index)
        print(f"completed natural rule-kernel variant {name}", flush=True)

    exact_sources = {
        "exact_rule_pair_motif": "raw_rule",
        "exact_background_pair_motif": "local_background",
        "exact_row_permuted_pair_motif": "row_permuted_rule",
    }
    exact_dimension = views["raw_rule"].shape[1] * (views["raw_rule"].shape[1] - 1) // 2
    for exact_index, (name, source) in enumerate(exact_sources.items()):
        kernel_pair = exact_pair_motif_kernel(
            views[source], query_node, reference_node,
            compact["query_ptr"], compact["molecule_ptr"],
        )
        kernel_pairs[name] = kernel_pair
        evaluate_kernel(name, exact_dimension, kernel_pair, 500 + exact_index)
        print(f"completed exact pair-rule variant {name}", flush=True)

    # A two-weight nonnegative multiple kernel keeps the proven raw-rule
    # channel intact and asks whether co-occurrence has any independent value.
    mkl_selection = []
    mkl_ranks = {}
    for raw_beta in args.beta:
        for motif_beta in args.beta:
            molecule_score = np.maximum.reduceat(
                official_pair
                + float(raw_beta) * kernel_pairs["raw_rule"]
                + float(motif_beta) * kernel_pairs["raw_motif_only"],
                compact["molecule_ptr"][:-1],
            )
            ranks = rank_queries(
                molecule_score, compact["query_ptr"], compact["molecule_label"],
            )
            mkl_ranks[(float(raw_beta), float(motif_beta))] = ranks
            mkl_selection.append({
                "raw_beta": float(raw_beta), "motif_beta": float(motif_beta),
                **retrieval(baseline_rank[discovery_local], ranks[discovery_local]),
            })
    mkl_selected = max(
        mkl_selection,
        key=lambda item: (
            int(item["risk_utility_at_1"]), -int(item["introduced_at_1"]),
            float(item["delta_mrr"]),
            -(float(item["raw_beta"]) + float(item["motif_beta"])),
        ),
    )
    mkl_key = (float(mkl_selected["raw_beta"]), float(mkl_selected["motif_beta"]))
    mkl_inner_rank = mkl_ranks[mkl_key][inner_local]
    inner_ranks["raw_plus_motif_mkl"] = mkl_inner_rank
    reports["raw_plus_motif_mkl"] = {
        "dimension": int(views["raw_rule"].shape[1] + views["raw_motif_only"].shape[1]),
        "selection_on_natural_training_formulas": mkl_selection,
        "selected_raw_beta": mkl_key[0],
        "selected_motif_beta": mkl_key[1],
        "held_inner": retrieval(baseline_rank[inner_local], mkl_inner_rank),
        "held_inner_formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
            inner_formula, baseline_rank[inner_local], mkl_inner_rank,
            args.bootstrap_draws, args.seed + 700,
        ),
    }

    exact_mkl_selection = []
    exact_mkl_ranks = {}
    for raw_beta in args.beta:
        for motif_beta in args.beta:
            molecule_score = np.maximum.reduceat(
                official_pair
                + float(raw_beta) * kernel_pairs["raw_rule"]
                + float(motif_beta) * kernel_pairs["exact_rule_pair_motif"],
                compact["molecule_ptr"][:-1],
            )
            ranks = rank_queries(
                molecule_score, compact["query_ptr"], compact["molecule_label"],
            )
            exact_mkl_ranks[(float(raw_beta), float(motif_beta))] = ranks
            exact_mkl_selection.append({
                "raw_beta": float(raw_beta), "motif_beta": float(motif_beta),
                **retrieval(baseline_rank[discovery_local], ranks[discovery_local]),
            })
    exact_mkl_selected = max(
        exact_mkl_selection,
        key=lambda item: (
            int(item["risk_utility_at_1"]), -int(item["introduced_at_1"]),
            float(item["delta_mrr"]),
            -(float(item["raw_beta"]) + float(item["motif_beta"])),
        ),
    )
    exact_mkl_key = (
        float(exact_mkl_selected["raw_beta"]), float(exact_mkl_selected["motif_beta"]),
    )
    exact_mkl_inner_rank = exact_mkl_ranks[exact_mkl_key][inner_local]
    inner_ranks["raw_plus_exact_motif_mkl"] = exact_mkl_inner_rank
    reports["raw_plus_exact_motif_mkl"] = {
        "dimension": int(views["raw_rule"].shape[1] + exact_dimension),
        "selection_on_natural_training_formulas": exact_mkl_selection,
        "selected_raw_beta": exact_mkl_key[0],
        "selected_motif_beta": exact_mkl_key[1],
        "held_inner": retrieval(baseline_rank[inner_local], exact_mkl_inner_rank),
        "held_inner_formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
            inner_formula, baseline_rank[inner_local], exact_mkl_inner_rank,
            args.bootstrap_draws, args.seed + 750,
        ),
    }

    def evaluate_typed_mkl(
        name: str,
        left_kernel: str,
        right_kernel: str,
        seed_offset: int,
    ) -> None:
        selection = []
        ranks_by_weight = {}
        for left_beta in args.beta:
            for right_beta in args.beta:
                molecule_score = np.maximum.reduceat(
                    official_pair
                    + float(left_beta) * kernel_pairs[left_kernel]
                    + float(right_beta) * kernel_pairs[right_kernel],
                    compact["molecule_ptr"][:-1],
                )
                ranks = rank_queries(
                    molecule_score, compact["query_ptr"], compact["molecule_label"],
                )
                key = (float(left_beta), float(right_beta))
                ranks_by_weight[key] = ranks
                selection.append({
                    "neutral_loss_beta": key[0],
                    "fragment_ion_beta": key[1],
                    **retrieval(baseline_rank[discovery_local], ranks[discovery_local]),
                })
        selected = max(
            selection,
            key=lambda item: (
                int(item["risk_utility_at_1"]), -int(item["introduced_at_1"]),
                float(item["delta_mrr"]),
                -(float(item["neutral_loss_beta"]) + float(item["fragment_ion_beta"])),
            ),
        )
        selected_key = (
            float(selected["neutral_loss_beta"]),
            float(selected["fragment_ion_beta"]),
        )
        held_rank = ranks_by_weight[selected_key][inner_local]
        held_all = np.stack(
            [rank[inner_local] for rank in ranks_by_weight.values()], axis=1,
        )
        oracle_rank = np.min(held_all, axis=1)
        inner_ranks[name] = held_rank
        reports[name] = {
            "dimension": int(
                views[left_kernel].shape[1] + views[right_kernel].shape[1]
            ),
            "selection_on_natural_training_formulas": selection,
            "selected_neutral_loss_beta": selected_key[0],
            "selected_fragment_ion_beta": selected_key[1],
            "held_inner": retrieval(baseline_rank[inner_local], held_rank),
            "held_inner_formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
                inner_formula, baseline_rank[inner_local], held_rank,
                args.bootstrap_draws, args.seed + seed_offset,
            ),
            "held_inner_no_op_aware_weight_oracle": {
                **retrieval(baseline_rank[inner_local], oracle_rank),
                "claim_limit": (
                    "The two weights are selected per query after observing the true rank; "
                    "this is headroom only, not a deployable method result."
                ),
            },
        }

    # NL and CF arise from precursor-minus-fragment and absolute fragment-mass
    # observations, respectively.  A block-separable conic kernel prevents one
    # evidence type from controlling the other's cosine normalization.
    evaluate_typed_mkl(
        "typed_rule_mkl", "raw_neutral_loss", "raw_fragment_ion", 1800,
    )
    evaluate_typed_mkl(
        "typed_background_control_mkl",
        "background_neutral_loss", "background_fragment_ion", 1850,
    )
    evaluate_typed_mkl(
        "typed_row_permuted_control_mkl",
        "row_permuted_neutral_loss", "row_permuted_fragment_ion", 1900,
    )

    comparisons = {}
    for primary_index, primary in enumerate(PRIMARY):
        comparisons[primary] = {}
        for control_index, control in enumerate(name for name in requested if name != primary):
            comparisons[primary][f"minus_{control}"] = paired_rank_comparison(
                inner_ranks[primary], inner_ranks[control], inner_formula,
                draws=args.bootstrap_draws,
                seed=args.seed + 1000 + primary_index * 100 + control_index,
            )
    comparisons["raw_plus_motif_mkl"] = {
        f"minus_{control}": paired_rank_comparison(
            inner_ranks["raw_plus_motif_mkl"], inner_ranks[control], inner_formula,
            draws=args.bootstrap_draws, seed=args.seed + 1500 + control_index,
        )
        for control_index, control in enumerate((
            "raw_rule", "raw_motif_only", "local_background_motif_only",
            "row_permuted_rule_motif_only",
        ))
    }
    comparisons["raw_plus_exact_motif_mkl"] = {
        f"minus_{control}": paired_rank_comparison(
            inner_ranks["raw_plus_exact_motif_mkl"], inner_ranks[control], inner_formula,
            draws=args.bootstrap_draws, seed=args.seed + 1700 + control_index,
        )
        for control_index, control in enumerate((
            "raw_rule", "exact_rule_pair_motif", "exact_background_pair_motif",
            "exact_row_permuted_pair_motif",
        ))
    }
    comparisons["typed_rule_mkl"] = {
        f"minus_{control}": paired_rank_comparison(
            inner_ranks["typed_rule_mkl"], inner_ranks[control], inner_formula,
            draws=args.bootstrap_draws, seed=args.seed + 2000 + control_index,
        )
        for control_index, control in enumerate((
            "raw_rule", "typed_background_control_mkl",
            "typed_row_permuted_control_mkl",
        ))
    }
    raw = reports["raw_rule_motif"]
    raw_pass = (
        raw["held_inner"]["corrected_at_1"] > raw["held_inner"]["introduced_at_1"]
        and raw["held_inner_formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0
        and comparisons["raw_rule_motif"]["minus_local_background_motif_only"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0
        and comparisons["raw_rule_motif"]["minus_row_permuted_rule_motif_only"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0
    )
    report = {
        "status": (
            "CHEMAWARE_RULE_COOCCURRENCE_NATURAL_DEVELOPMENT_PASS"
            if raw_pass else "CHEMAWARE_RULE_COOCCURRENCE_NATURAL_DEVELOPMENT_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "natural training-formula selection; already-used inner development; outer untouched",
        "claim_limit": (
            "A frozen shared-feature development result, not a DreaMS fine-tuning result.  Fold 3 has "
            "been used by prior development and fold 4 remains sealed."
        ),
        "data": {
            "discovery_queries": int(len(discovery)),
            "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(len(outer_pool)),
            "discovery_formulas": int(len(np.unique(body["query_formula"][discovery]))),
            "inner_formulas": int(len(np.unique(inner_formula))),
            "formula_overlap": 0,
            "unique_spectrum_rows": int(len(node_rows)),
            "reference_pairs_scored": int(len(reference_node)),
            "discovery_baseline": retrieval(
                baseline_rank[discovery_local], baseline_rank[discovery_local],
            ),
            "inner_baseline": retrieval(baseline_rank[inner_local], baseline_rank[inner_local]),
        },
        "features": feature_report,
        "variants": reports,
        "held_inner_paired_comparisons": comparisons,
        "gates": {
            "raw_rule_motif_absolute_ci_positive": raw[
                "held_inner_formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "raw_rule_motif_corrected_exceeds_introduced": (
                raw["held_inner"]["corrected_at_1"] > raw["held_inner"]["introduced_at_1"]
            ),
            "raw_rule_motif_beats_local_background_ci": comparisons[
                "raw_rule_motif"
            ]["minus_local_background_motif_only"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "raw_rule_motif_beats_row_permuted_ci": comparisons[
                "raw_rule_motif"
            ]["minus_row_permuted_rule_motif_only"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "counterfactual_increment_over_raw_ci": comparisons[
                "counterfactual_motif"
            ]["minus_raw_rule_motif"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "mkl_motif_weight_nonzero": reports["raw_plus_motif_mkl"]["selected_motif_beta"] > 0,
            "mkl_increment_over_raw_rule_ci": comparisons[
                "raw_plus_motif_mkl"
            ]["minus_raw_rule"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "exact_mkl_motif_weight_nonzero": reports[
                "raw_plus_exact_motif_mkl"
            ]["selected_motif_beta"] > 0,
            "exact_mkl_increment_over_raw_rule_ci": comparisons[
                "raw_plus_exact_motif_mkl"
            ]["minus_raw_rule"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "typed_mkl_increment_over_raw_rule_ci": comparisons[
                "typed_rule_mkl"
            ]["minus_raw_rule"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "typed_mkl_beats_background_control_ci": comparisons[
                "typed_rule_mkl"
            ]["minus_typed_background_control_mkl"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "typed_mkl_beats_row_permuted_control_ci": comparisons[
                "typed_rule_mkl"
            ]["minus_typed_row_permuted_control_mkl"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_rule_motif_natural_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=inner_formula,
            baseline_rank=baseline_rank[inner_local], **inner_ranks,
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "raw_rule_motif": reports["raw_rule_motif"]["held_inner"],
        "counterfactual_motif": reports["counterfactual_motif"]["held_inner"],
        "raw_plus_motif_mkl": reports["raw_plus_motif_mkl"]["held_inner"],
        "raw_plus_exact_motif_mkl": reports["raw_plus_exact_motif_mkl"]["held_inner"],
        "typed_rule_mkl": reports["typed_rule_mkl"]["held_inner"],
        "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
