"""Audit spectrum-localized NL/CF chemical kernels as one shared embedding.

The gate sees one clean spectrum only.  It is trained on formula folds 0--1
with a continuous hardest-candidate margin, checkpointed on fold 2, and
evaluated once on the already-used inner fold 3.  Fold 4 remains sealed.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

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
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_localized_rule_kernel_core import (
    LocalizedRuleGate,
    hardest_margin_loss,
    localized_pair_scores,
    localized_shared_embedding,
)
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
        default=ROOT / "data/validation/chemaware_localized_typed_rule_kernel_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--counterfactual-offset", type=float, default=0.137)
    parser.add_argument("--motif-dimension", type=int, default=64)
    parser.add_argument("--motif-top-rules", type=int, default=12)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--neutral-loss-cap", type=float, default=1.6)
    parser.add_argument("--fragment-ion-cap", type=float, default=1.6)
    parser.add_argument("--official-projection-dim", type=int, default=16)
    parser.add_argument("--rule-projection-dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gate-anchor", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--correct-query-weight", type=float, default=2.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def fixed_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(input_dim, output_dim)).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
    return matrix


def spectrum_statistics(
    node_rows: np.ndarray,
    cache_position: dict[int, int],
    token_dir: Path,
) -> np.ndarray:
    valid_all = np.load(token_dir / "valid.npy", mmap_mode="r")
    intensity_all = np.load(token_dir / "intensity_f32.npy", mmap_mode="r")
    mz_all = np.load(token_dir / "mz_f32.npy", mmap_mode="r")
    precursor_all = np.load(token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    position = np.asarray([cache_position[int(row)] for row in node_rows], dtype=np.int64)
    valid = np.asarray(valid_all[position], dtype=bool)
    intensity = np.maximum(np.asarray(intensity_all[position], dtype=np.float32), 0.0) * valid
    total = np.maximum(intensity.sum(axis=1, keepdims=True), 1e-12)
    probability = intensity / total
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1)
    sorted_intensity = np.sort(intensity, axis=1)[:, ::-1]
    mz = np.asarray(mz_all[position], dtype=np.float32)
    valid_mz = np.where(valid, mz, np.nan)
    mz_min = np.nanmin(valid_mz, axis=1)
    mz_max = np.nanmax(valid_mz, axis=1)
    return np.column_stack((
        np.log1p(np.asarray(precursor_all[position], dtype=np.float32)),
        np.sum(valid, axis=1), entropy,
        sorted_intensity[:, 0] / total[:, 0],
        np.sum(sorted_intensity[:, :5], axis=1) / total[:, 0],
        mz_min, mz_max, mz_max - mz_min,
    )).astype(np.float32)


def make_gate_features(
    official: np.ndarray,
    neutral_loss: np.ndarray,
    fragment_ion: np.ndarray,
    views: dict[str, np.ndarray],
    spectral: np.ndarray,
    *,
    seed: int,
    official_dim: int,
    rule_dim: int,
    control: bool,
) -> tuple[np.ndarray, list[str]]:
    nl = views["row_permuted_neutral_loss"] if control else neutral_loss
    cf = views["row_permuted_fragment_ion"] if control else fragment_ion
    official_projected = official @ fixed_projection(official.shape[1], official_dim, seed + 11)
    nl_projected = nl @ fixed_projection(nl.shape[1], rule_dim, seed + 23)
    cf_projected = cf @ fixed_projection(cf.shape[1], rule_dim, seed + 37)
    scalar_names = [
        "raw_nl_energy", "raw_cf_energy", "background_nl_energy", "background_cf_energy",
        "raw_nl_count", "raw_cf_count", "raw_background_nl_cosine",
        "raw_background_cf_cosine", "log_precursor", "peak_count", "peak_entropy",
        "top1_fraction", "top5_fraction", "mz_min", "mz_max", "mz_span",
    ]
    scalar = np.column_stack((
        views["gate_raw_neutral_loss_energy"][:, 0],
        views["gate_raw_fragment_ion_energy"][:, 0],
        views["gate_background_neutral_loss_energy"][:, 0],
        views["gate_background_fragment_ion_energy"][:, 0],
        views["gate_raw_neutral_loss_count"][:, 0],
        views["gate_raw_fragment_ion_count"][:, 0],
        np.sum(neutral_loss * views["background_neutral_loss"], axis=1),
        np.sum(fragment_ion * views["background_fragment_ion"], axis=1),
        spectral,
    )).astype(np.float32)
    feature = np.concatenate((official_projected, nl_projected, cf_projected, scalar), axis=1)
    names = (
        [f"official_fixed_projection_{index}" for index in range(official_dim)]
        + [f"{'permuted_' if control else ''}nl_fixed_projection_{index}" for index in range(rule_dim)]
        + [f"{'permuted_' if control else ''}cf_fixed_projection_{index}" for index in range(rule_dim)]
        + scalar_names
    )
    if not np.isfinite(feature).all():
        raise RuntimeError("non-finite clean spectrum gate feature")
    return feature, names


def standardize_on_training_nodes(
    feature: np.ndarray,
    training_nodes: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = feature[training_nodes].mean(axis=0, dtype=np.float64)
    scale = feature[training_nodes].std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    transformed = ((feature - mean) / scale).astype(np.float32)
    return transformed, {"mean": mean.tolist(), "scale": scale.tolist()}


def global_typed_grid(
    official_pair: np.ndarray,
    nl_pair: np.ndarray,
    cf_pair: np.ndarray,
    compact: dict[str, np.ndarray],
    baseline_rank: np.ndarray,
    query_index: np.ndarray,
    beta: list[float] | tuple[float, ...],
) -> tuple[tuple[float, float], list[dict[str, object]], dict[tuple[float, float], np.ndarray]]:
    rows = []
    ranks = {}
    for nl_beta in beta:
        for cf_beta in beta:
            molecule = np.maximum.reduceat(
                official_pair + float(nl_beta) * nl_pair + float(cf_beta) * cf_pair,
                compact["molecule_ptr"][:-1],
            )
            rank = rank_queries(molecule, compact["query_ptr"], compact["molecule_label"])
            key = (float(nl_beta), float(cf_beta))
            ranks[key] = rank
            rows.append({
                "neutral_loss_beta": key[0], "fragment_ion_beta": key[1],
                **retrieval(baseline_rank[query_index], rank[query_index]),
            })
    selected = max(
        rows,
        key=lambda item: (
            int(item["risk_utility_at_1"]), -int(item["introduced_at_1"]),
            float(item["delta_mrr"]),
            -(float(item["neutral_loss_beta"]) + float(item["fragment_ion_beta"])),
        ),
    )
    key = (float(selected["neutral_loss_beta"]), float(selected["fragment_ion_beta"]))
    return key, rows, ranks


def rank_from_gates(
    official_pair: np.ndarray,
    nl_pair: np.ndarray,
    cf_pair: np.ndarray,
    gates: np.ndarray,
    pair_query_node: np.ndarray,
    reference_node: np.ndarray,
    compact: dict[str, np.ndarray],
    *,
    nl_cap: float,
    cf_cap: float,
) -> np.ndarray:
    score = (
        official_pair
        + float(nl_cap) * gates[pair_query_node, 0] * gates[reference_node, 0] * nl_pair
        + float(cf_cap) * gates[pair_query_node, 1] * gates[reference_node, 1] * cf_pair
    )
    molecule = np.maximum.reduceat(score, compact["molecule_ptr"][:-1])
    return rank_queries(molecule, compact["query_ptr"], compact["molecule_label"])


def train_gate(
    feature: np.ndarray,
    official_pair: np.ndarray,
    nl_pair: np.ndarray,
    cf_pair: np.ndarray,
    query_node: np.ndarray,
    reference_node: np.ndarray,
    pair_query: np.ndarray,
    pair_molecule: np.ndarray,
    molecule_query: np.ndarray,
    molecule_label: np.ndarray,
    baseline_rank: np.ndarray,
    train_count: int,
    validation_index: np.ndarray,
    compact: dict[str, np.ndarray],
    initial_gate: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[LocalizedRuleGate, list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(args.seed)
    model = LocalizedRuleGate(feature.shape[1], initial_gate=0.5)
    with torch.no_grad():
        model.linear.bias[:] = torch.logit(torch.tensor(initial_gate, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    feature_t = torch.as_tensor(feature)
    train_molecule_end = int(compact["query_ptr"][train_count])
    train_pair_end = int(compact["molecule_ptr"][train_molecule_end])
    q_node_t = torch.as_tensor(query_node[pair_query[:train_pair_end]], dtype=torch.long)
    r_node_t = torch.as_tensor(reference_node[:train_pair_end], dtype=torch.long)
    pair_molecule_t = torch.as_tensor(pair_molecule[:train_pair_end], dtype=torch.long)
    molecule_query_t = torch.as_tensor(molecule_query[:train_molecule_end], dtype=torch.long)
    molecule_label_t = torch.as_tensor(molecule_label[:train_molecule_end], dtype=torch.bool)
    official_t = torch.as_tensor(official_pair[:train_pair_end], dtype=torch.float32)
    nl_t = torch.as_tensor(nl_pair[:train_pair_end], dtype=torch.float32)
    cf_t = torch.as_tensor(cf_pair[:train_pair_end], dtype=torch.float32)
    query_weight = np.where(
        baseline_rank[:train_count] == 1, args.correct_query_weight, 1.0,
    ).astype(np.float32)
    query_weight_t = torch.as_tensor(query_weight)
    target_gate = torch.as_tensor(initial_gate, dtype=torch.float32)
    history = []
    best = None
    best_key = None
    for epoch in range(args.epochs + 1):
        if epoch > 0:
            model.train(); optimizer.zero_grad(set_to_none=True)
            gates = model(feature_t)
            pair_score = localized_pair_scores(
                official_t, nl_t, cf_t, gates, q_node_t, r_node_t,
                neutral_loss_cap=args.neutral_loss_cap,
                fragment_ion_cap=args.fragment_ion_cap,
            )
            margin_loss, _ = hardest_margin_loss(
                pair_score, pair_molecule_t, molecule_query_t, molecule_label_t,
                query_weight_t, temperature=args.temperature, margin=args.margin,
            )
            anchor = torch.mean((gates - target_gate[None, :]) ** 2)
            loss = margin_loss + args.gate_anchor * anchor
            loss.backward(); optimizer.step()
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                gates_np = model(feature_t).numpy()
            rank = rank_from_gates(
                official_pair, nl_pair, cf_pair, gates_np,
                query_node[pair_query], reference_node, compact,
                nl_cap=args.neutral_loss_cap, cf_cap=args.fragment_ion_cap,
            )
            metric = retrieval(baseline_rank[validation_index], rank[validation_index])
            row = {
                "epoch": int(epoch), **metric,
                "neutral_loss_gate_mean": float(gates_np[:, 0].mean()),
                "fragment_ion_gate_mean": float(gates_np[:, 1].mean()),
            }
            history.append(row)
            key = (
                int(metric["risk_utility_at_1"]), -int(metric["introduced_at_1"]),
                float(metric["delta_mrr"]), -int(epoch),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = copy.deepcopy(model.state_dict())
    if best is None:
        raise RuntimeError("localized gate did not produce a checkpoint")
    model.load_state_dict(best)
    selected = max(
        history,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), -int(row["epoch"]),
        ),
    )
    return model, history, selected


def main() -> None:
    args = arguments()
    torch.set_num_threads(args.torch_threads)
    required = [
        args.manifest, args.rule_library, args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.token_dir / "valid.npy",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
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
        raise RuntimeError("train, validation, and inner formulas overlap")
    selected_queries = np.concatenate((train, validation, inner))
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
    if absent := [int(row) for row in node_rows if int(row) not in cache_position]:
        raise RuntimeError(f"{len(absent)} required spectrum rows absent from cache")
    positions = np.asarray([cache_position[int(row)] for row in node_rows], dtype=np.int64)
    official = np.asarray(official_cache[positions], dtype=np.float32)
    feature_args = SimpleNamespace(**vars(args))
    views, feature_report = build_feature_views(node_rows, cache_position, feature_args)
    neutral_loss = views["raw_neutral_loss"]
    fragment_ion = views["raw_fragment_ion"]
    spectral = spectrum_statistics(node_rows, cache_position, args.token_dir)

    official_pair = query_reference_dot(
        official, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    nl_pair = query_reference_dot(
        neutral_loss, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    cf_pair = query_reference_dot(
        fragment_ion, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    permuted_nl_pair = query_reference_dot(
        views["row_permuted_neutral_loss"], query_node, reference_node,
        compact["query_ptr"], compact["molecule_ptr"],
    )
    permuted_cf_pair = query_reference_dot(
        views["row_permuted_fragment_ion"], query_node, reference_node,
        compact["query_ptr"], compact["molecule_ptr"],
    )
    baseline_molecule = np.maximum.reduceat(official_pair, compact["molecule_ptr"][:-1])
    baseline_rank = rank_queries(baseline_molecule, compact["query_ptr"], compact["molecule_label"])
    train_index = np.arange(len(train), dtype=np.int64)
    validation_index = np.arange(len(train), len(train) + len(validation), dtype=np.int64)
    inner_index = np.arange(len(train) + len(validation), len(selected_queries), dtype=np.int64)

    global_key, global_grid, global_ranks = global_typed_grid(
        official_pair, nl_pair, cf_pair, compact, baseline_rank, train_index, args.beta,
    )
    initial_gate = (
        float(np.sqrt(global_key[0] / args.neutral_loss_cap)) if global_key[0] else 1e-3,
        float(np.sqrt(global_key[1] / args.fragment_ion_cap)) if global_key[1] else 1e-3,
    )
    if not all(0.0 < value < 1.0 for value in initial_gate):
        raise RuntimeError("kernel caps must exceed selected global weights")

    train_molecule_end = int(compact["query_ptr"][len(train)])
    train_pair_end = int(compact["molecule_ptr"][train_molecule_end])
    training_nodes = np.unique(np.concatenate((
        query_node[:len(train)], reference_node[:train_pair_end],
    )))
    correct_feature, feature_names = make_gate_features(
        official, neutral_loss, fragment_ion, views, spectral,
        seed=args.seed, official_dim=args.official_projection_dim,
        rule_dim=args.rule_projection_dim, control=False,
    )
    control_feature, control_feature_names = make_gate_features(
        official, neutral_loss, fragment_ion, views, spectral,
        seed=args.seed, official_dim=args.official_projection_dim,
        rule_dim=args.rule_projection_dim, control=True,
    )
    correct_feature, scaler = standardize_on_training_nodes(correct_feature, training_nodes)
    control_feature, control_scaler = standardize_on_training_nodes(control_feature, training_nodes)

    correct_model, correct_history, correct_selected = train_gate(
        correct_feature, official_pair, nl_pair, cf_pair, query_node, reference_node,
        pair_query, pair_molecule, molecule_query, compact["molecule_label"], baseline_rank,
        len(train), validation_index, compact, initial_gate, args,
    )
    control_model, control_history, control_selected = train_gate(
        control_feature, official_pair, permuted_nl_pair, permuted_cf_pair,
        query_node, reference_node, pair_query, pair_molecule, molecule_query,
        compact["molecule_label"], baseline_rank, len(train), validation_index,
        compact, initial_gate, args,
    )
    with torch.no_grad():
        correct_gates = correct_model(torch.as_tensor(correct_feature)).numpy()
        control_gates = control_model(torch.as_tensor(control_feature)).numpy()
    localized_rank = rank_from_gates(
        official_pair, nl_pair, cf_pair, correct_gates,
        query_node[pair_query], reference_node, compact,
        nl_cap=args.neutral_loss_cap, cf_cap=args.fragment_ion_cap,
    )
    control_rank = rank_from_gates(
        official_pair, permuted_nl_pair, permuted_cf_pair, control_gates,
        query_node[pair_query], reference_node, compact,
        nl_cap=args.neutral_loss_cap, cf_cap=args.fragment_ion_cap,
    )
    global_rank = global_ranks[global_key]
    inner_formula = body["query_formula"][inner].astype(str)
    localized_inner = retrieval(baseline_rank[inner_index], localized_rank[inner_index])
    control_inner = retrieval(baseline_rank[inner_index], control_rank[inner_index])
    global_inner = retrieval(baseline_rank[inner_index], global_rank[inner_index])
    localized_ci = bootstrap(
        inner_formula, baseline_rank[inner_index], localized_rank[inner_index],
        args.bootstrap_draws, args.seed + 500,
    )
    versus_global = paired_rank_comparison(
        localized_rank[inner_index], global_rank[inner_index], inner_formula,
        draws=args.bootstrap_draws, seed=args.seed + 600,
    )
    versus_control = paired_rank_comparison(
        localized_rank[inner_index], control_rank[inner_index], inner_formula,
        draws=args.bootstrap_draws, seed=args.seed + 700,
    )

    # Numerical proof that inference is an ordinary dot product between one
    # vector per clean spectrum, with no candidate list in the gate.
    phi = localized_shared_embedding(
        official, neutral_loss, fragment_ion, correct_gates,
        neutral_loss_cap=args.neutral_loss_cap, fragment_ion_cap=args.fragment_ion_cap,
    )
    audit_pair = np.linspace(0, len(reference_node) - 1, min(4096, len(reference_node)), dtype=np.int64)
    explicit = np.sum(
        phi[query_node[pair_query[audit_pair]]] * phi[reference_node[audit_pair]], axis=1,
    )
    direct = (
        official_pair[audit_pair]
        + args.neutral_loss_cap
        * correct_gates[query_node[pair_query[audit_pair]], 0]
        * correct_gates[reference_node[audit_pair], 0] * nl_pair[audit_pair]
        + args.fragment_ion_cap
        * correct_gates[query_node[pair_query[audit_pair]], 1]
        * correct_gates[reference_node[audit_pair], 1] * cf_pair[audit_pair]
    )
    primal_error = float(np.max(np.abs(explicit - direct)))

    report = {
        "status": "CHEMAWARE_LOCALIZED_TYPED_RULE_DEVELOPMENT_COMPLETE",
        "formal_training_authorized": False,
        "dreams_weights_updated": False,
        "localized_gate_trained": True,
        "scope": "formula folds 0-1 train; fold 2 checkpoint selection; used inner fold 3 evaluation; fold 4 sealed",
        "claim_limit": (
            "This is a frozen-feature shared-embedding development audit.  The inner fold has "
            "already been used by prior development and is not an external confirmation set."
        ),
        "data": {
            "train_queries": int(len(train)), "validation_queries": int(len(validation)),
            "inner_queries": int(len(inner)), "outer_queries_untouched": int(len(outer_pool)),
            "train_formulas": int(len(formula_sets[0])),
            "validation_formulas": int(len(formula_sets[1])),
            "inner_formulas": int(len(formula_sets[2])),
            "pairwise_formula_overlap": 0, "unique_spectrum_rows": int(len(node_rows)),
            "reference_pairs": int(len(reference_node)),
        },
        "method": {
            "shared_map": "[official, sqrt(nl_cap)*eta_nl(x)*r_nl(x), sqrt(cf_cap)*eta_cf(x)*r_cf(x)]",
            "gate_input": "one unmodified spectrum only",
            "candidate_scores_or_ranks_used_by_gate": False,
            "identity_or_formula_used_by_gate": False,
            "outcomes_used": "training loss and validation checkpoint selection only",
            "loss": "risk-weighted smooth hardest-negative margin",
            "correct_query_weight": float(args.correct_query_weight),
            "feature_dimension": int(correct_feature.shape[1]),
            "feature_names": feature_names,
            "control_feature_names": control_feature_names,
            "primal_dot_product_max_abs_error": primal_error,
            "global_weight_selected_on_train": {
                "neutral_loss": global_key[0], "fragment_ion": global_key[1],
            },
            "initial_gate": {"neutral_loss": initial_gate[0], "fragment_ion": initial_gate[1]},
            "caps": {"neutral_loss": args.neutral_loss_cap, "fragment_ion": args.fragment_ion_cap},
        },
        "global_grid_on_train": global_grid,
        "correct_localized_gate": {
            "validation_history": correct_history,
            "selected_validation_checkpoint": correct_selected,
            "inner": localized_inner,
            "inner_formula_cluster_bootstrap_delta_recall1_ci95": localized_ci,
            "gate_summary": {
                "neutral_loss_mean": float(correct_gates[:, 0].mean()),
                "neutral_loss_std": float(correct_gates[:, 0].std()),
                "fragment_ion_mean": float(correct_gates[:, 1].mean()),
                "fragment_ion_std": float(correct_gates[:, 1].std()),
            },
        },
        "constant_global_kernel": {"inner": global_inner},
        "row_permuted_control": {
            "validation_history": control_history,
            "selected_validation_checkpoint": control_selected,
            "inner": control_inner,
        },
        "paired_inner": {
            "localized_minus_constant_global": versus_global,
            "localized_minus_row_permuted_control": versus_control,
        },
        "gates": {
            "shared_primal_identity": primal_error < 2e-5,
            "absolute_inner_ci_positive": localized_ci[0] > 0,
            "localized_increment_over_global_ci_positive": versus_global[
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "localized_beats_row_permuted_control_ci": versus_control[
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0,
            "outer_fold_untouched": True,
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
            "scaler": scaler, "control_scaler": control_scaler,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_localized_typed_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=inner_formula,
            baseline_rank=baseline_rank[inner_index], global_rank=global_rank[inner_index],
            localized_rank=localized_rank[inner_index], control_rank=control_rank[inner_index],
        )
        torch.save({
            "state_dict": correct_model.state_dict(), "feature_names": feature_names,
            "mean": np.asarray(scaler["mean"], dtype=np.float32),
            "scale": np.asarray(scaler["scale"], dtype=np.float32),
            "initial_gate": initial_gate,
        }, temporary / "localized_gate.pt")
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "global": global_inner,
        "localized": localized_inner, "row_permuted_control": control_inner,
        "paired_inner": report["paired_inner"], "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
