"""Audit complementarity and deploy-visible routing of two frozen ChemAware results."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from audit_chemaware_mass_kernel_embedding import KernelCache
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from chemaware_spectrum_gate_core import SCALAR_NAMES, spectrum_scalar_vector
from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-ranks", type=Path, default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz")
    parser.add_argument("--candidate-policy", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_rule_residual_policy_v2/inner_policy.npz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_parent_candidate_complementarity_v1")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def rule_observables(cache: KernelCache, row: int) -> np.ndarray:
    raw = cache.get(int(row))
    true = np.asarray(raw["rule_response"], dtype=np.float64)
    local = np.stack([
        np.asarray(raw[name], dtype=np.float64) for name in (
            "rule_response_local_background_a", "rule_response_local_background_b",
            "rule_response_local_background_c",
        )
    ])
    support = true > 0
    probability = true[support]
    probability = probability / probability.sum() if probability.sum() else probability
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    centered = true - local.mean(axis=0)
    local_cosine = local @ true
    return np.asarray([
        np.sum(support), np.max(true, initial=0.0), entropy,
        np.linalg.norm(centered), np.max(local_cosine), np.mean(local_cosine),
    ], dtype=np.float64)


def standardized_logistic_prediction(
    feature: np.ndarray,
    target: np.ndarray,
    formula: np.ndarray,
) -> np.ndarray:
    unique_formula = np.unique(formula)
    splitter = GroupKFold(n_splits=min(5, len(unique_formula)))
    output = np.full(len(target), np.nan, dtype=np.float64)
    for train, test in splitter.split(feature, target, groups=formula):
        mean = feature[train].mean(axis=0)
        scale = feature[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        _, inverse, count = np.unique(formula[train], return_inverse=True, return_counts=True)
        weight = 1.0 / count[inverse]
        weight /= weight.mean()
        model = LogisticRegression(C=1.0, solver="liblinear", random_state=0)
        model.fit((feature[train] - mean) / scale, target[train], sample_weight=weight)
        output[test] = model.predict_proba((feature[test] - mean) / scale)[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("formula-grouped router did not cover every disagreement")
    return output


def route_rank(
    parent_rank: np.ndarray,
    candidate_rank: np.ndarray,
    disagreement: np.ndarray,
    choose_candidate: np.ndarray,
) -> np.ndarray:
    output = np.asarray(parent_rank).copy()
    index = np.flatnonzero(disagreement)
    output[index[choose_candidate]] = np.asarray(candidate_rank)[index[choose_candidate]]
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    parent = load_npz(args.parent_ranks)
    candidate = load_npz(args.candidate_policy)
    if not (
        np.array_equal(parent["query"], candidate["query"])
        and np.array_equal(parent["formula"].astype(str), candidate["formula"].astype(str))
        and np.array_equal(parent["baseline_rank"], candidate["baseline_rank"])
    ):
        raise RuntimeError("frozen parent and candidate policy are not query-aligned")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    row_position = {int(row): index for index, row in enumerate(rows)}
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library, top_peaks=32,
        kernel_dim=2048, bin_width=0.02, grid_offsets=4, intensity_power=0.5,
        mass_shift_da=0.137, pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=0.02, rule_channel_weight=1.0,
    )
    cache = KernelCache(kernel_args, row_position, variants=(
        "rule_response", "rule_response_local_background_a",
        "rule_response_local_background_b", "rule_response_local_background_c",
    ))
    query = parent["query"].astype(np.int64)
    query_rows = body["query_row"][query].astype(np.int64)
    baseline_rank = parent["baseline_rank"]
    parent_rank = parent["whitened_true_rank"]
    candidate_rank = candidate["correct_rank"]
    parent_hit = parent_rank == 1
    candidate_hit = candidate_rank == 1
    baseline_hit = baseline_rank == 1
    disagreement = parent_hit != candidate_hit
    disagreement_index = np.flatnonzero(disagreement)
    target = candidate_hit[disagreement].astype(np.int64)
    selected = candidate["selected_candidate_slot"] >= 0
    utility = np.asarray(candidate["best_predicted_utility"], dtype=np.float64)
    spectrum = np.stack([spectrum_scalar_vector(cache, int(row)) for row in query_rows])
    rule = np.stack([rule_observables(cache, int(row)) for row in query_rows])
    feature = np.column_stack((
        spectrum[disagreement], rule[disagreement], utility[disagreement],
        selected[disagreement].astype(np.float64),
    ))
    feature_names = [*SCALAR_NAMES,
        "active_rule_channels", "maximum_rule_response", "rule_response_entropy",
        "centered_rule_energy", "maximum_true_local_cosine", "mean_true_local_cosine",
        "candidate_policy_best_utility", "candidate_policy_selected",
    ]
    formula = parent["formula"].astype(str)
    disagreement_formula = formula[disagreement]
    probability = standardized_logistic_prediction(feature, target, disagreement_formula)
    routed_rank = route_rank(parent_rank, candidate_rank, disagreement, probability >= 0.5)
    selected_hybrid = np.asarray(parent_rank).copy()
    selected_hybrid[selected] = candidate_rank[selected]
    oracle_rank = np.where(parent_hit | candidate_hit, 1, np.minimum(parent_rank, candidate_rank))
    rng = np.random.default_rng(args.seed)
    null_delta = []
    selected_disagreement = selected[disagreement]
    for _ in range(args.permutations):
        permuted = target.copy()
        for stratum in (False, True):
            index = np.flatnonzero(selected_disagreement == stratum)
            permuted[index] = target[rng.permutation(index)]
        null_probability = standardized_logistic_prediction(feature, permuted, disagreement_formula)
        null_rank = route_rank(parent_rank, candidate_rank, disagreement, null_probability >= 0.5)
        null_delta.append(float(np.mean(null_rank == 1) - np.mean(parent_hit)))
    null_delta = np.asarray(null_delta)
    routed_increment = float(np.mean(routed_rank == 1) - np.mean(parent_hit))
    methods = {
        "shared_parent": retrieval(baseline_rank, parent_rank),
        "candidate_policy": retrieval(baseline_rank, candidate_rank),
        "candidate_if_policy_active_else_parent": retrieval(baseline_rank, selected_hybrid),
        "formula_oof_router": retrieval(baseline_rank, routed_rank),
        "oracle_union_upper_bound": retrieval(baseline_rank, oracle_rank),
    }
    overlap = {
        "parent_corrected": int(np.sum(~baseline_hit & parent_hit)),
        "candidate_corrected": int(np.sum(~baseline_hit & candidate_hit)),
        "corrected_overlap": int(np.sum(~baseline_hit & parent_hit & candidate_hit)),
        "corrected_union": int(np.sum(~baseline_hit & (parent_hit | candidate_hit))),
        "parent_unique_corrections": int(np.sum(~baseline_hit & parent_hit & ~candidate_hit)),
        "candidate_unique_corrections": int(np.sum(~baseline_hit & candidate_hit & ~parent_hit)),
        "parent_introduced": int(np.sum(baseline_hit & ~parent_hit)),
        "candidate_introduced": int(np.sum(baseline_hit & ~candidate_hit)),
        "introduction_overlap": int(np.sum(baseline_hit & ~parent_hit & ~candidate_hit)),
    }
    report = {
        "status": "CHEMAWARE_PARENT_CANDIDATE_COMPLEMENTARITY_COMPLETE",
        "formal_training_authorized": False, "weights_updated": False,
        "scope": "post-hoc formula-grouped cross-fit within already-used fold 3; fold 4 sealed",
        "claim_limit": "Oracle is only an upper bound. OOF router is feasibility evidence on a consumed development fold, not confirmation.",
        "methods": methods, "overlap": overlap,
        "disagreement": {
            "queries": int(len(target)), "formulas": int(len(np.unique(disagreement_formula))),
            "candidate_wins": int(target.sum()), "parent_wins": int(len(target) - target.sum()),
            "feature_names": feature_names,
            "oof_accuracy": float(np.mean((probability >= 0.5) == target)),
            "oof_candidate_precision": float(np.mean(target[probability >= 0.5])) if np.any(probability >= 0.5) else None,
            "oof_candidate_calls": int(np.sum(probability >= 0.5)),
        },
        "router_increment_over_parent": {
            "delta_recall1": routed_increment,
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, parent_rank, routed_rank, draws=args.bootstrap_draws, seed=args.seed + 1,
            ),
            "label_permutation_repeats": int(args.permutations),
            "label_permutation_null_quantiles": np.quantile(null_delta, (0, 0.5, 0.95, 1)).astype(float).tolist(),
            "empirical_p_ge_observed": float((1 + np.sum(null_delta >= routed_increment)) / (1 + len(null_delta))),
        },
        "outer_fold_untouched": True,
        "provenance": {
            "parent_ranks_sha256": sha256_file(args.parent_ranks),
            "candidate_policy_sha256": sha256_file(args.candidate_policy),
            "manifest_sha256": sha256_file(args.manifest),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_complementarity_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "router_oof.npz", query=query, formula=formula,
            baseline_rank=baseline_rank, parent_rank=parent_rank, candidate_rank=candidate_rank,
            routed_rank=routed_rank, oracle_rank=oracle_rank,
            disagreement_index=disagreement_index, candidate_win_target=target,
            candidate_probability=probability,
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "methods": methods, "overlap": overlap, "disagreement": report["disagreement"], "router_increment_over_parent": report["router_increment_over_parent"], "output": str((args.output / "report.json").resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
