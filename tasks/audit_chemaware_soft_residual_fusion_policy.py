"""Softly fuse frozen-parent candidate margins with orthogonal chemical utility."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audit_chemaware_candidate_evidence_policy as candidate_policy
import audit_chemaware_orthogonal_rule_residual_policy as orthogonal
from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from audit_chemaware_parent_conditioned_residual_policy import (
    ParentFeatureCache,
    condition_on_parent,
    fit_parent_transform,
)
from audit_chemaware_whitened_centered_rule_kernel import FourCenterCache
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_orthogonal_rule_policy_core import (
    base_and_chemical_feature_indices,
    signed_rule_contrast,
    validate_matched_tables,
)
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("zero_contrast", "reversed_contrast", "alignment_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-ranks", type=Path, default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_soft_residual_fusion_policy_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--parent-fit-identities", type=int, default=4096)
    parser.add_argument("--parent-final-fit-identities", type=int, default=8192)
    parser.add_argument("--base-shrinkage", type=float, default=0.50)
    parser.add_argument("--parent-mass-beta", type=float, default=0.40)
    parser.add_argument("--parent-rule-beta", type=float, default=0.80)
    parser.add_argument("--beta", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60))
    parser.add_argument("--global-mass-beta", type=float, default=0.10)
    parser.add_argument("--global-rule-beta", type=float, default=0.20)
    parser.add_argument("--residual-dose", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--promotion-scale", type=float, nargs="+", default=(0.001, 0.0025, 0.005, 0.010, 0.025, 0.050, 0.10))
    parser.add_argument("--threshold-quantiles", type=int, default=21)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--min-selected-formulas", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parent_molecule_scores(scored: dict[str, np.ndarray]) -> list[np.ndarray]:
    output = []
    for index, pair_score in enumerate(scored["global"]):
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        output.append(np.maximum.reduceat(np.asarray(pair_score, dtype=np.float32), pointer[:-1]))
    return output


def soft_fusion_rank(
    table: dict[str, np.ndarray],
    parent_score: list[np.ndarray],
    utility: np.ndarray,
    threshold: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rank = np.empty(len(parent_score), dtype=np.int16)
    active = np.zeros(len(parent_score), dtype=bool)
    for query, original in enumerate(parent_score):
        score = np.asarray(original, dtype=np.float64).copy()
        valid = np.flatnonzero(table["valid"][query])
        for slot in valid:
            excess = max(0.0, float(utility[query, slot]) - float(threshold))
            if excess <= 0:
                continue
            molecule = int(table["proposed_candidate"][query, slot])
            score[molecule] += float(scale) * excess
            active[query] = True
        labels = np.asarray(table["molecule_labels"][query], dtype=bool)
        if labels.shape != score.shape or np.sum(labels) != 1:
            raise RuntimeError("soft fusion molecule labels are invalid")
        rank[query] = strict_rank(score, labels)
    return rank, active


def utility_thresholds(utility: np.ndarray, quantiles: int) -> np.ndarray:
    best = np.max(utility, axis=1)
    finite = best[np.isfinite(best)]
    return np.unique(np.r_[
        np.quantile(finite, np.linspace(0.0, 1.0, int(quantiles))),
        np.nextafter(float(np.max(finite)), np.inf),
    ])


def select_soft_fusion(table, formula, parent_score, utilities, args):
    parent_rank, _ = soft_fusion_rank(
        table, parent_score, np.zeros_like(next(iter(next(iter(utilities.values())).values()))),
        1.0, 0.0,
    )
    rows = []
    for dose, arms in utilities.items():
        for threshold in utility_thresholds(arms["correct"], args.threshold_quantiles):
            for scale in args.promotion_scale:
                metrics = {}
                selected = 0
                selected_formulas = 0
                for name, utility in arms.items():
                    rank, active = soft_fusion_rank(
                        table, parent_score, utility, float(threshold), float(scale),
                    )
                    metrics[name] = retrieval(parent_rank, rank)
                    if name == "correct":
                        selected = int(active.sum())
                        selected_formulas = int(len(np.unique(formula[active])))
                if selected and selected_formulas < args.min_selected_formulas:
                    continue
                correct = metrics["correct"]
                if correct["risk_utility_at_1"] < 0 or correct["corrected_at_1"] < 2 * correct["introduced_at_1"]:
                    continue
                control_risk = {name: metrics[name]["risk_utility_at_1"] for name in CONTROLS}
                rows.append({
                    "dose": float(dose), "threshold": float(threshold), "scale": float(scale),
                    "selected": selected, "selected_formulas": selected_formulas,
                    "correct": correct, "controls": control_risk,
                    "minimum_specific_risk_advantage": int(min(
                        int(correct["risk_utility_at_1"]) - int(value) for value in control_risk.values()
                    )),
                })
    if not rows:
        raise RuntimeError("soft fusion selection produced no risk-admissible setting")
    selected = max(rows, key=lambda row: (
        int(row["minimum_specific_risk_advantage"]),
        int(row["correct"]["risk_utility_at_1"]),
        -int(row["correct"]["introduced_at_1"]), float(row["correct"]["delta_mrr"]),
        -float(row["dose"]), -float(row["scale"]), float(row["threshold"]),
    ))
    return selected, rows, parent_rank


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.train_identities = min(args.train_identities, 192)
        args.validation_identities = min(args.validation_identities, 128)
        args.max_inner_identities = 128
        args.parent_fit_identities = min(args.parent_fit_identities, 256)
        args.parent_final_fit_identities = min(args.parent_final_fit_identities, 512)
        args.max_iter = min(args.max_iter, 30)
        args.min_samples_leaf = min(args.min_samples_leaf, 20)
        args.min_selected_formulas = min(args.min_selected_formulas, 10)
        args.threshold_quantiles = min(args.threshold_quantiles, 11)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    actions = [(float(mass), float(rule)) for mass in args.beta for rule in args.beta if mass > 0 or rule > 0]
    global_action = actions.index((args.global_mass_beta, args.global_rule_beta))
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    pools = [np.flatnonzero(np.isin(fold, [0, 1])), np.flatnonzero(fold == 2), np.flatnonzero(fold == 3)]
    queries = [
        identity_balanced_queries(pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1), args.train_identities),
        identity_balanced_queries(pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2), args.validation_identities),
        identity_balanced_queries(pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19), args.max_inner_identities),
    ]
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    train_formula_fold = fold[queries[0]]
    parent_fit = identity_balanced_queries(pools[0], body["query_ik14"], np.random.default_rng(args.fold_seed + 11), args.parent_fit_identities)
    parent_final_fit = identity_balanced_queries(np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"], np.random.default_rng(args.fold_seed + 23), args.parent_final_fit_identities)
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library, top_peaks=32,
        kernel_dim=2048, bin_width=0.02, grid_offsets=4, intensity_power=0.5,
        mass_shift_da=0.137, pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=0.02, rule_channel_weight=1.0,
    )
    base = KernelCache(kernel_args, row_position, variants=(
        "mass", "rule_response", "rule_response_content_permuted",
        "rule_response_local_background_a", "rule_response_local_background_b",
        "rule_response_local_background_c",
    ))
    centers = FourCenterCache(base)
    initial_mean, initial_transform, initial_parent_report = fit_parent_transform(parent_fit, body, centers, args.base_shrinkage)
    final_mean, final_transform, final_parent_report = fit_parent_transform(parent_final_fit, body, centers, args.base_shrinkage)
    correct_tables = []
    control_tables = []
    parent_scores = []
    for index, query in enumerate(queries):
        scored = score_queries(query, body, official, row_position, base, (
            "mass", "rule_response", "rule_response_content_permuted",
        ))
        correct = candidate_policy.build_candidate_table(scored, actions, global_action, "rule_response")
        control = candidate_policy.build_candidate_table(scored, actions, global_action, "rule_response_content_permuted")
        correct["molecule_labels"] = np.asarray(scored["labels"], dtype=object)
        control["molecule_labels"] = np.asarray(scored["labels"], dtype=object)
        validate_matched_tables(correct, control)
        correct_tables.append(correct)
        control_tables.append(control)
        mean, transform = (initial_mean, initial_transform) if index < 2 else (final_mean, final_transform)
        parent_cache = ParentFeatureCache(centers, mean, transform)
        parent_scored = score_queries(query, body, official, row_position, parent_cache, ("mass", "parent_rule"))
        parent_scored = condition_on_parent(parent_scored, args.parent_mass_beta, args.parent_rule_beta)
        parent_scores.append(parent_molecule_scores(parent_scored))
        print(f"completed stacked inputs {index + 1}/3", flush=True)
    base_indices, chemical_indices = base_and_chemical_feature_indices(candidate_policy.FEATURE_NAMES)
    base_features = [table["feature"][..., base_indices] for table in correct_tables]
    contrasts = [signed_rule_contrast(correct, control, chemical_indices) for correct, control in zip(correct_tables, control_tables, strict=True)]
    channels = {target: orthogonal.fit_channel(
        correct_tables[0], formulas[0], train_formula_fold, base_features[0], contrasts[0],
        target, args, args.seed + (100 if target == "benefit" else 200),
    ) for target in ("benefit", "harmful")}
    contrast_arms = [orthogonal.make_contrast_arms(contrast, table, args.seed + 300 + index)[0] for index, (contrast, table) in enumerate(zip(contrasts, correct_tables, strict=True))]
    validation_utilities = {float(dose): {name: orthogonal.utility_for_contrast(
        channels, base_features[1], contrast, correct_tables[1]["valid"], float(dose), args.risk_penalty,
    )[0] for name, contrast in contrast_arms[1].items()} for dose in args.residual_dose}
    selected, selection_grid, validation_parent_rank = select_soft_fusion(
        correct_tables[1], formulas[1], parent_scores[1], validation_utilities, args,
    )
    inner_utilities = {name: orthogonal.utility_for_contrast(
        channels, base_features[2], contrast, correct_tables[2]["valid"], float(selected["dose"]), args.risk_penalty,
    )[0] for name, contrast in contrast_arms[2].items()}
    evaluated = {}
    for name, utility in inner_utilities.items():
        rank, active = soft_fusion_rank(
            correct_tables[2], parent_scores[2], utility,
            float(selected["threshold"]), float(selected["scale"]),
        )
        parent_rank, _ = soft_fusion_rank(correct_tables[2], parent_scores[2], utility, np.inf, 0.0)
        evaluated[name] = {"rank": rank, "active": active, "metric": retrieval(parent_rank, rank)}
    formal_protocol = args.parent_final_fit_identities == 8192 and args.max_inner_identities == 0
    frozen = np.load(args.parent_ranks)
    if formal_protocol and (not np.array_equal(frozen["query"], queries[2]) or not np.array_equal(frozen["whitened_true_rank"], parent_rank)):
        raise RuntimeError("soft fusion parent does not reproduce frozen parent")
    primary = evaluated["correct"]
    comparisons = {f"correct_minus_{name}": paired_rank_comparison(
        primary["rank"], evaluated[name]["rank"], formulas[2],
        draws=args.bootstrap_draws, seed=args.seed + 600 + index,
    ) for index, name in enumerate(CONTROLS)}
    increment_ci = bootstrap(formulas[2], parent_rank, primary["rank"], args.bootstrap_draws, args.seed + 500)
    absolute = retrieval(frozen["baseline_rank"], primary["rank"]) if formal_protocol else None
    gates = {
        "selected_nonzero_residual_dose": float(selected["dose"]) > 0,
        "selected_positive_promotion_scale": float(selected["scale"]) > 0,
        "validation_specific_advantage_positive": int(selected["minimum_specific_risk_advantage"]) > 0,
        "increment_over_parent_ci_positive": increment_ci[0] > 0,
        "increment_corrected_exceeds_twice_introduced": primary["metric"]["corrected_at_1"] > 2 * primary["metric"]["introduced_at_1"],
        **{f"beats_{name}_ci": comparisons[f"correct_minus_{name}"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0 for name in CONTROLS},
        "outer_fold_untouched": True,
    }
    report = {
        "status": "CHEMAWARE_SOFT_RESIDUAL_FUSION_PASS" if all(gates.values()) else "CHEMAWARE_SOFT_RESIDUAL_FUSION_FAIL",
        "formal_training_authorized": False, "weights_updated": False,
        "candidate_conditioned": True, "shared_embedding_result": False,
        "scope": "orthogonal utility fit folds 0-1; soft-fusion selection fold 2; inner fold 3; outer fold 4 sealed",
        "claim_limit": "Development soft-fusion evidence; formal absolute gain is emitted only when the frozen full parent is exactly reproduced.",
        "method": {"score": "parent_molecule_score + scale*relu(candidate_utility-threshold)", "candidate_proposals": "unchanged official-relative orthogonal residual table", "parent": "marginal-whitened centered-rule shared kernel", "same_dose_threshold_scale_for_controls": True},
        "data": {"train_queries": len(queries[0]), "validation_queries": len(queries[1]), "inner_queries": len(queries[2]), "outer_queries_untouched": int(np.sum(fold == 4))},
        "parent_transform": {"initial": initial_parent_report, "final": final_parent_report},
        "selection": selected, "selection_grid_size": len(selection_grid),
        "validation_parent": retrieval(correct_tables[1]["baseline_rank"], validation_parent_rank),
        "held_inner_increment_over_parent": {name: value["metric"] for name, value in evaluated.items()},
        "held_inner_increment_formula_ci95": increment_ci,
        "held_inner_absolute_from_official": absolute,
        "paired_inner": comparisons, "gates": gates,
        "provenance": {"parent_ranks_sha256": sha256_file(args.parent_ranks), "manifest_sha256": sha256_file(args.manifest), "rule_library_sha256": sha256_file(args.rule_library)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_soft_fusion_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(temporary / "inner_policy.npz", query=queries[2], formula=formulas[2], parent_rank=parent_rank, **{f"{name}_rank": value["rank"] for name, value in evaluated.items()}, correct_active=primary["active"])
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "selection": selected, "held_inner_increment_over_parent": report["held_inner_increment_over_parent"], "held_inner_increment_formula_ci95": increment_ci, "held_inner_absolute_from_official": absolute, "paired_inner": comparisons, "gates": gates, "output": str((args.output / "report.json").resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
