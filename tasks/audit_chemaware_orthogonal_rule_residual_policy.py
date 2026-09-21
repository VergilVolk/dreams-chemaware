"""Audit a paired orthogonal chemical increment over a non-rule policy.

The nuisance policy is trained only from official-embedding and mass-kernel
candidate geometry.  Correct-minus-content-permuted rule evidence is then fit
to cross-fitted label residuals.  Formula fold 2 selects one dose/threshold by
the worst paired advantage over the explicitly configured negative controls.
Fold 3 is the already-used inner audit; fold 4 is never used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
try:
    import sklearn
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ModuleNotFoundError:  # --triplet-evidence-only is NumPy-only.
    sklearn = None
    joblib = None
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None

import audit_chemaware_candidate_evidence_policy as candidate_policy
from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from chemaware_numpy_sampling import identity_balanced_queries, stable_formula_folds

action_policy = None
paired_rank_comparison = None
from chemaware_orthogonal_rule_policy_core import (
    base_and_chemical_feature_indices,
    combine_residual_score,
    permute_candidate_contrast_within_strata,
    rotate_candidate_contrast_truthblind,
    signed_rule_contrast,
    symmetric_center_contrast,
    validate_matched_tables,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTROL_NAMES = ("zero_contrast", "reversed_contrast", "alignment_permuted")
TRIPLET_EVIDENCE_METRICS = (
    "action_top_fraction",
    "action_largest_region_fraction",
    "action_same_neighbor_fraction",
    "action_margin_max",
    "action_margin_mean",
    "action_best_advantage_over_baseline",
    "global_action_advantage_over_baseline",
    "global_action_selects_candidate",
    "candidate_official_rank_fraction",
    "candidate_rule_rank_fraction",
    "candidate_rule_max",
    "candidate_rule_top2_mean",
    "delta_rule_max",
    "delta_rule_top2_mean",
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
        "--rule-control-variants", nargs="+",
        default=("rule_response_content_permuted",),
        choices=(
            "rule_response_content_permuted",
            "rule_response_content_permuted_b",
            "rule_response_content_permuted_c",
            "rule_response_local_background_a",
            "rule_response_local_background_b",
            "rule_response_local_background_c",
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_rule_residual_policy_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--global-mass-beta", type=float, default=0.1)
    parser.add_argument("--global-rule-beta", type=float, default=0.2)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--min-selected-formulas", type=int, default=50)
    parser.add_argument("--residual-dose", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument(
        "--contrast-representation", choices=("mean", "symmetric_summary"),
        default="mean",
    )
    parser.add_argument(
        "--selection-control-mode",
        choices=("legacy_audit", "deployment_safe"),
        default="legacy_audit",
        help=(
            "legacy_audit preserves the canonical truth-conditioned alignment "
            "control; deployment_safe replaces it with the label-free within-query "
            "candidate-rotation control"
        ),
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
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--triplet-evidence-only", action="store_true",
        help="write roles 0-3 multinull action evidence and stop before policy fitting",
    )
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def triplet_evidence_payload(
    query: np.ndarray,
    formula: np.ndarray,
    identity: np.ndarray,
    correct: dict[str, np.ndarray],
    controls: list[dict[str, np.ndarray]],
    control_names: tuple[str, ...],
    action_count: int,
) -> dict[str, np.ndarray]:
    """Serialize all directional events without applying a learned policy.

    This is a supervised training-only ledger.  It deliberately retains the
    outcome labels needed to construct native DreaMS positives/negatives, but
    it does not contain a deployment-time decision or an outer-fold row.
    """
    if len(controls) != len(control_names):
        raise ValueError("triplet evidence control names/tables differ")
    for control in controls:
        validate_matched_tables(correct, control)
        for key in ("benefit", "harmful", "rank"):
            if not np.array_equal(np.asarray(correct[key]), np.asarray(control[key])):
                raise ValueError(f"triplet outcome drifted across semantic arms: {key}")
    metric_index = np.asarray(
        [candidate_policy.FEATURE_NAMES.index(name) for name in TRIPLET_EVIDENCE_METRICS],
        dtype=np.int64,
    )
    tables = [correct, *controls]
    return {
        "schema": np.asarray("chemaware_multinull_triplet_evidence_v1"),
        "query": np.asarray(query, dtype=np.int64),
        "formula": np.asarray(formula).astype(str),
        "identity": np.asarray(identity).astype(str),
        "valid": np.asarray(correct["valid"], dtype=bool),
        "proposed_candidate": np.asarray(correct["proposed_candidate"], dtype=np.int16),
        "baseline_candidate": np.asarray(correct["baseline_candidate"], dtype=np.int16),
        "baseline_rank": np.asarray(correct["baseline_rank"], dtype=np.int16),
        "proposal_rank": np.asarray(correct["rank"], dtype=np.int16),
        "benefit": np.asarray(correct["benefit"], dtype=bool),
        "harmful": np.asarray(correct["harmful"], dtype=bool),
        "arm_names": np.asarray(("correct", *control_names)),
        "metric_names": np.asarray(TRIPLET_EVIDENCE_METRICS),
        "action_count": np.asarray(int(action_count), dtype=np.int16),
        "arm_metric": np.stack([
            np.asarray(table["feature"], dtype=np.float32)[..., metric_index]
            for table in tables
        ]),
    }


def event_arrays(
    table: dict[str, np.ndarray], formula: np.ndarray, feature: np.ndarray,
    target_name: str, query_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.arange(len(formula)) if query_mask is None else np.flatnonzero(query_mask)
    valid = np.asarray(table["valid"])[query]
    x = np.asarray(feature)[query][valid]
    y = np.asarray(table[target_name])[query][valid].astype(np.float64)
    query_weight = action_policy.formula_query_weight(np.asarray(formula)[query])
    count = valid.sum(axis=1)
    weight = np.concatenate([
        np.full(int(n), query_weight[index] / max(1, int(n)), dtype=np.float64)
        for index, n in enumerate(count)
    ])
    return x, y, weight


def new_classifier(args: argparse.Namespace, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=args.learning_rate, max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes, min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization, early_stopping=False,
        random_state=seed,
    )


def new_regressor(args: argparse.Namespace, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=args.learning_rate, max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes, min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization, early_stopping=False,
        random_state=seed,
    )


def fit_balanced_classifier(
    table: dict[str, np.ndarray], formula: np.ndarray, feature: np.ndarray,
    target_name: str, args: argparse.Namespace, seed: int,
    query_mask: np.ndarray | None = None,
) -> tuple[HistGradientBoostingClassifier, float]:
    x, y, event_weight = event_arrays(table, formula, feature, target_name, query_mask)
    sample_weight, prevalence = action_policy.balanced_event_weight(y.astype(bool), event_weight)
    model = new_classifier(args, seed)
    model.fit(x, y.astype(bool), sample_weight=sample_weight)
    return model, float(prevalence)


def predict_classifier(
    model: HistGradientBoostingClassifier, prevalence: float,
    feature: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    result = np.zeros(np.asarray(valid).shape, dtype=np.float64)
    result[valid] = action_policy.restore_prior(
        model.predict_proba(np.asarray(feature)[valid])[:, 1], prevalence,
    )
    return result


def crossfit_nuisance(
    table: dict[str, np.ndarray], formula: np.ndarray, formula_fold: np.ndarray,
    feature: np.ndarray, target_name: str, args: argparse.Namespace, seed: int,
) -> np.ndarray:
    valid = np.asarray(table["valid"], dtype=bool)
    prediction = np.full(valid.shape, np.nan, dtype=np.float64)
    roles = sorted(np.unique(formula_fold).tolist())
    if roles != [0, 1]:
        raise RuntimeError(f"nuisance cross-fit expected formula folds 0 and 1, got {roles}")
    for held in roles:
        train_mask = formula_fold != held
        held_mask = formula_fold == held
        model, prevalence = fit_balanced_classifier(
            table, formula, feature, target_name, args, seed + int(held), train_mask,
        )
        held_prediction = predict_classifier(model, prevalence, feature[held_mask], valid[held_mask])
        prediction[held_mask] = held_prediction
    if np.any(~np.isfinite(prediction[valid])):
        raise RuntimeError("nuisance cross-fit left non-finite candidate predictions")
    prediction[~valid] = 0.0
    return prediction


def fit_residual_model(
    table: dict[str, np.ndarray], formula: np.ndarray, contrast: np.ndarray,
    target_name: str, nuisance_oof: np.ndarray, args: argparse.Namespace, seed: int,
) -> tuple[HistGradientBoostingRegressor, np.ndarray, dict[str, float]]:
    valid = np.asarray(table["valid"], dtype=bool)
    raw_x, y, event_weight = event_arrays(table, formula, contrast, target_name)
    active = np.flatnonzero(np.std(raw_x, axis=0) > 1e-8).astype(np.int64)
    if not len(active):
        raise RuntimeError("all chemical contrast features are constant")
    target = y - np.asarray(nuisance_oof)[valid]
    model = new_regressor(args, seed)
    model.fit(raw_x[:, active], target, sample_weight=event_weight)
    fitted = model.predict(raw_x[:, active])
    diagnostics = {
        "active_features": int(len(active)),
        "residual_target_mean": float(np.average(target, weights=event_weight)),
        "residual_fit_weighted_mse": float(np.average((target - fitted) ** 2, weights=event_weight)),
        "residual_prediction_std": float(np.std(fitted)),
    }
    return model, active, diagnostics


def predict_residual(
    model: HistGradientBoostingRegressor, active: np.ndarray,
    contrast: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    output = np.zeros(np.asarray(valid).shape, dtype=np.float64)
    output[valid] = model.predict(np.asarray(contrast)[valid][:, active])
    return output


def fit_channel(
    train: dict[str, np.ndarray], train_formula: np.ndarray, train_fold: np.ndarray,
    train_base: np.ndarray, train_contrast: np.ndarray, target_name: str,
    args: argparse.Namespace, seed: int,
) -> dict[str, object]:
    nuisance_oof = crossfit_nuisance(
        train, train_formula, train_fold, train_base, target_name, args, seed,
    )
    nuisance_model, prevalence = fit_balanced_classifier(
        train, train_formula, train_base, target_name, args, seed + 20,
    )
    residual_model, active, diagnostics = fit_residual_model(
        train, train_formula, train_contrast, target_name, nuisance_oof,
        args, seed + 40,
    )
    return {
        "nuisance_model": nuisance_model,
        "prevalence": prevalence,
        "residual_model": residual_model,
        "active": active,
        "diagnostics": diagnostics,
    }


def predict_channel(
    channel: dict[str, object], base: np.ndarray, contrast: np.ndarray,
    valid: np.ndarray, dose: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nuisance = predict_classifier(
        channel["nuisance_model"], channel["prevalence"], base, valid,
    )
    residual = predict_residual(
        channel["residual_model"], channel["active"], contrast, valid,
    )
    return combine_residual_score(nuisance, residual, dose), nuisance, residual


def utility_for_contrast(
    channels: dict[str, dict[str, object]], base: np.ndarray, contrast: np.ndarray,
    valid: np.ndarray, dose: float, risk_penalty: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    benefit, benefit_base, benefit_residual = predict_channel(
        channels["benefit"], base, contrast, valid, dose,
    )
    harm, harm_base, harm_residual = predict_channel(
        channels["harmful"], base, contrast, valid, dose,
    )
    utility = benefit - float(risk_penalty) * harm
    utility[~valid] = -np.inf
    return utility, {
        "benefit": benefit, "harm": harm,
        "benefit_base": benefit_base, "harm_base": harm_base,
        "benefit_residual": benefit_residual, "harm_residual": harm_residual,
    }


def fit_direct_channels(
    table: dict[str, np.ndarray], formula: np.ndarray, feature: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> dict[str, dict[str, object]]:
    """Fit the same-feature ordinary learner used to test method necessity."""
    output: dict[str, dict[str, object]] = {}
    for offset, target in enumerate(("benefit", "harmful")):
        model, prevalence = fit_balanced_classifier(
            table, formula, feature, target, args, seed + offset,
        )
        output[target] = {"model": model, "prevalence": prevalence}
    return output


def direct_utility(
    channels: dict[str, dict[str, object]], feature: np.ndarray,
    valid: np.ndarray, risk_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = {}
    for target in ("benefit", "harmful"):
        predictions[target] = predict_classifier(
            channels[target]["model"], channels[target]["prevalence"], feature, valid,
        )
    utility = predictions["benefit"] - float(risk_penalty) * predictions["harmful"]
    utility[~valid] = -np.inf
    return utility, predictions["benefit"], predictions["harmful"]


def rank_at_threshold(
    table: dict[str, np.ndarray], utility: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return candidate_policy.ranks_at_threshold(table, utility, threshold)


def candidate_thresholds(utility: np.ndarray) -> np.ndarray:
    best = np.max(utility, axis=1)
    finite = best[np.isfinite(best)]
    return np.unique(np.r_[
        np.quantile(finite, np.linspace(0.0, 1.0, 101)),
        np.nextafter(np.max(finite), np.inf),
    ])


def select_specific_setting(
    table: dict[str, np.ndarray], formula: np.ndarray,
    utilities: dict[float, dict[str, np.ndarray]], min_selected_formulas: int,
    control_names: tuple[str, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    baseline_rank = np.asarray(table["baseline_rank"])
    for dose, arms in utilities.items():
        for threshold in candidate_thresholds(arms["correct"]):
            metrics: dict[str, dict[str, object]] = {}
            selected_count = 0
            selected_formulas = 0
            for name, utility in arms.items():
                rank, selected, _ = rank_at_threshold(table, utility, float(threshold))
                metrics[name] = retrieval(baseline_rank, rank)
                if name == "correct":
                    active = selected >= 0
                    selected_count = int(active.sum())
                    selected_formulas = int(len(np.unique(formula[active].astype(str))))
            if selected_count and selected_formulas < min_selected_formulas:
                continue
            correct_risk = int(metrics["correct"]["risk_utility_at_1"])
            control_risk = {name: int(metrics[name]["risk_utility_at_1"]) for name in control_names}
            if correct_risk < 0:
                continue
            if (
                int(metrics["correct"]["corrected_at_1"])
                < 2 * int(metrics["correct"]["introduced_at_1"])
            ):
                continue
            rows.append({
                "dose": float(dose), "threshold": float(threshold),
                "selected": selected_count, "selected_formulas": selected_formulas,
                "correct": metrics["correct"], "controls": control_risk,
                "minimum_specific_risk_advantage": int(
                    min(correct_risk - value for value in control_risk.values())
                ),
            })
    selected = max(
        rows,
        key=lambda row: (
            int(row["minimum_specific_risk_advantage"]),
            int(row["correct"]["risk_utility_at_1"]),
            -int(row["correct"]["introduced_at_1"]),
            float(row["correct"]["delta_mrr"]),
            -float(row["dose"]),
            float(row["threshold"]),
        ),
    )
    return selected, rows


def select_specific_setting_or_abstain(
    table: dict[str, np.ndarray], formula: np.ndarray,
    utilities: dict[float, dict[str, np.ndarray]], min_selected_formulas: int,
    control_names: tuple[str, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Use an explicit no-op when an ablation has no admissible policy."""
    try:
        return select_specific_setting(
            table, formula, utilities, min_selected_formulas, control_names,
        )
    except ValueError:
        finite = np.concatenate([
            values[np.isfinite(values)]
            for arms in utilities.values() for values in arms.values()
            if np.any(np.isfinite(values))
        ])
        abstain_threshold = np.nextafter(float(np.max(finite)), np.inf)
        return {
            "dose": float(next(iter(utilities))),
            "threshold": abstain_threshold,
            "selected": 0,
            "selected_formulas": 0,
            "correct": retrieval(table["baseline_rank"], table["baseline_rank"]),
            "controls": {name: 0 for name in control_names},
            "minimum_specific_risk_advantage": 0,
            "no_admissible_setting": True,
        }, []


def make_contrast_arms(
    contrast: np.ndarray, table: dict[str, np.ndarray], seed: int,
    null_contrasts: list[np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    permuted, source = permute_candidate_contrast_within_strata(
        contrast, table["valid"], table["baseline_rank"], seed,
    )
    rotated, _ = rotate_candidate_contrast_truthblind(
        contrast, table["valid"],
    )
    arms = {
        "correct": contrast,
        "zero_contrast": np.zeros_like(contrast),
        "reversed_contrast": -contrast,
        "alignment_permuted": permuted,
        "candidate_rotated_truthblind": rotated,
    }
    for index, null in enumerate(null_contrasts or []):
        arms[f"null_semantics_{index}"] = np.asarray(null, dtype=np.float32)
    return arms, source


def averaged_control_table(
    tables: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not tables:
        raise ValueError("at least one rule control table is required")
    first = tables[0]
    for table in tables[1:]:
        validate_matched_tables(first, table)
    output = dict(first)
    output["feature"] = np.mean(
        np.stack([np.asarray(table["feature"], dtype=np.float32) for table in tables]),
        axis=0,
    ).astype(np.float32)
    return output


def null_semantic_contrasts(
    tables: list[dict[str, np.ndarray]], chemical_indices: np.ndarray,
) -> list[np.ndarray]:
    if len(tables) < 2:
        return []
    output: list[np.ndarray] = []
    for index, table in enumerate(tables):
        others = averaged_control_table([x for j, x in enumerate(tables) if j != index])
        output.append(signed_rule_contrast(table, others, chemical_indices))
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.train_identities = min(args.train_identities, 192)
        args.validation_identities = min(args.validation_identities, 128)
        args.max_inner_identities = 128
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
        args.max_iter = min(args.max_iter, 30)
        args.min_samples_leaf = min(args.min_samples_leaf, 20)
        args.min_selected_formulas = min(args.min_selected_formulas, 10)
    actions = [
        (float(mass), float(rule))
        for mass in args.beta for rule in args.beta
        if float(mass) > 0.0 or float(rule) > 0.0
    ]
    global_action = actions.index((args.global_mass_beta, args.global_rule_beta))
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    pools = [
        np.flatnonzero((fold == 0) | (fold == 1)),
        np.flatnonzero(fold == 2), np.flatnonzero(fold == 3),
    ]
    queries = [
        identity_balanced_queries(pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1), args.train_identities),
        identity_balanced_queries(pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2), args.validation_identities),
        identity_balanced_queries(pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19), args.max_inner_identities),
    ]
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    train_formula_fold = fold[queries[0]]
    if sorted(np.unique(train_formula_fold).tolist()) != [0, 1]:
        raise RuntimeError("training pool does not contain exactly formula folds 0 and 1")

    kernel_args = SimpleNamespace(**vars(args))
    correct_tables: list[dict[str, np.ndarray]] = []
    control_table_sets: list[list[dict[str, np.ndarray]]] = []
    score_variants = ("mass", "rule_response", *tuple(args.rule_control_variants))
    if len(set(score_variants)) != len(score_variants):
        raise RuntimeError("rule control variants must be unique")
    if len(args.rule_control_variants) not in (1, 3):
        raise RuntimeError("use either one legacy rule null or all three multi-null controls")
    cache = KernelCache(kernel_args, row_position, variants=score_variants)
    for name, query in zip(("train", "validation", "inner"), queries, strict=True):
        scored = score_queries(
            query, body, official, row_position, cache,
            score_variants,
        )
        correct = candidate_policy.build_candidate_table(
            scored, actions, global_action, "rule_response",
        )
        controls = [
            candidate_policy.build_candidate_table(scored, actions, global_action, variant)
            for variant in args.rule_control_variants
        ]
        for control in controls:
            validate_matched_tables(correct, control)
        correct_tables.append(correct)
        control_table_sets.append(controls)
        print(
            f"completed {name}: queries={len(query)} candidate_rows={int(correct['valid'].sum())} cache={len(cache.cache)}",
            flush=True,
        )

    if args.triplet_evidence_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="chemaware_triplet_evidence_", dir=args.output.parent))
        try:
            role_names = ("train", "selection", "confirmation")
            file_names = (
                "train_triplet_evidence.npz",
                "selection_triplet_evidence.npz",
                "confirmation_triplet_evidence.npz",
            )
            for index, file_name in enumerate(file_names):
                np.savez_compressed(
                    temporary / file_name,
                    **triplet_evidence_payload(
                        queries[index], formulas[index],
                        body["query_ik14"][queries[index]],
                        correct_tables[index], control_table_sets[index],
                        tuple(args.rule_control_variants), len(actions),
                    ),
                )
            evidence_report = {
                "status": "CHEMAWARE_MULTINULL_TRIPLET_EVIDENCE_COMPLETE",
                "policy_fitted": False,
                "weights_updated": False,
                "scope": "roles 0-1 optimization evidence; role 2 recipe selection; role 3 confirmation; role 4 untouched",
                "roles": {
                    name: {
                        "queries": int(len(queries[index])),
                        "candidate_events": int(correct_tables[index]["valid"].sum()),
                        "unique_formulas": int(len(np.unique(formulas[index].astype(str)))),
                        "file": file_names[index],
                    }
                    for index, name in enumerate(role_names)
                },
                "action_grid": {
                    "actions": int(len(actions)),
                    "correct_arm": "rule_response",
                    "semantic_null_arms": list(args.rule_control_variants),
                },
                "contracts": {
                    "formula_roles_disjoint": True,
                    "correct_null_candidate_rows_matched": True,
                    "outer_role_4_not_scored": True,
                    "no_learned_policy_used_for_triplet_evidence": True,
                },
                "provenance": {
                    "manifest_sha256": sha256(args.manifest),
                    "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
                    "rule_library_sha256": sha256(args.rule_library),
                },
            }
            (temporary / "report.json").write_text(
                json.dumps(evidence_report, indent=2), encoding="utf-8",
            )
            temporary.replace(args.output)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(json.dumps(evidence_report, indent=2), flush=True)
        return

    global action_policy, paired_rank_comparison
    if sklearn is None or joblib is None:
        raise RuntimeError("scikit-learn and joblib are required for policy fitting")
    import audit_chemaware_conservative_action_policy as action_policy_module
    from audit_chemaware_counterfactual_rule_kernel_natural import (
        paired_rank_comparison as paired_rank_comparison_function,
    )
    action_policy = action_policy_module
    paired_rank_comparison = paired_rank_comparison_function
    candidate_policy.action_policy = action_policy_module
    candidate_policy.paired_rank_comparison = paired_rank_comparison_function

    base_indices, chemical_indices = base_and_chemical_feature_indices(candidate_policy.FEATURE_NAMES)
    base_features = [table["feature"][..., base_indices] for table in correct_tables]
    for correct_base, tables in zip(base_features, control_table_sets, strict=True):
        if any(
            not np.array_equal(correct_base, table["feature"][..., base_indices])
            for table in tables
        ):
            raise RuntimeError("nuisance feature block unexpectedly depends on rule content")
    if args.contrast_representation == "symmetric_summary":
        if len(args.rule_control_variants) != 3:
            raise RuntimeError("symmetric summary requires exactly three rule nulls")
        contrasts = []
        null_contrasts = []
        for correct, controls in zip(correct_tables, control_table_sets, strict=True):
            centers = [correct, *controls]
            contrasts.append(symmetric_center_contrast(centers, 0, chemical_indices))
            null_contrasts.append([
                symmetric_center_contrast(centers, index, chemical_indices)
                for index in range(1, len(centers))
            ])
    else:
        averaged_controls = [averaged_control_table(tables) for tables in control_table_sets]
        contrasts = [
            signed_rule_contrast(correct, control, chemical_indices)
            for correct, control in zip(correct_tables, averaged_controls, strict=True)
        ]
        null_contrasts = [
            null_semantic_contrasts(tables, chemical_indices)
            for tables in control_table_sets
        ]
    channels = {}
    for target in ("benefit", "harmful"):
        print(f"fitting orthogonal teacher channel: {target}", flush=True)
        channels[target] = fit_channel(
            correct_tables[0], formulas[0], train_formula_fold,
            base_features[0], contrasts[0], target, args,
            args.seed + (100 if target == "benefit" else 200),
        )
        print(f"completed orthogonal teacher channel: {target}", flush=True)

    contrast_arms: list[dict[str, np.ndarray]] = []
    permutation_sources: list[np.ndarray] = []
    for index in range(3):
        arms, source = make_contrast_arms(
            contrasts[index], correct_tables[index], args.seed + 300 + index,
            null_contrasts[index],
        )
        contrast_arms.append(arms)
        permutation_sources.append(source)

    # Scientific necessity control: a conventional learner receives exactly
    # the same nuisance and chemical features but no residualization.
    direct_channels = fit_direct_channels(
        correct_tables[0], formulas[0],
        np.concatenate((base_features[0], contrasts[0]), axis=-1),
        args, args.seed + 700,
    )

    validation_utilities: dict[float, dict[str, np.ndarray]] = {}
    for dose in args.residual_dose:
        validation_utilities[float(dose)] = {
            name: utility_for_contrast(
                channels, base_features[1], contrast, correct_tables[1]["valid"],
                float(dose), args.risk_penalty,
            )[0]
            for name, contrast in contrast_arms[1].items()
        }
    if args.selection_control_mode == "legacy_audit":
        control_names = tuple(
            name for name in contrast_arms[1]
            if name != "correct" and name != "candidate_rotated_truthblind"
        )
    else:
        control_names = tuple(
            name for name in contrast_arms[1]
            if name in {
                "zero_contrast", "reversed_contrast",
                "candidate_rotated_truthblind",
            } or name.startswith("null_semantics_")
        )
    if not control_names:
        raise RuntimeError("no selection controls were configured")
    selected, selection_grid = select_specific_setting(
        correct_tables[1], formulas[1], validation_utilities,
        args.min_selected_formulas, control_names,
    )
    selected_dose = float(selected["dose"])
    selected_threshold = float(selected["threshold"])

    direct_validation_utilities = {
        1.0: {
            name: direct_utility(
                direct_channels,
                np.concatenate((base_features[1], contrast), axis=-1),
                correct_tables[1]["valid"], args.risk_penalty,
            )[0]
            for name, contrast in contrast_arms[1].items()
        }
    }
    direct_selected, direct_selection_grid = select_specific_setting_or_abstain(
        correct_tables[1], formulas[1], direct_validation_utilities,
        args.min_selected_formulas, control_names,
    )

    nuisance_validation = {}
    for target in ("benefit", "harmful"):
        nuisance_validation[target] = predict_classifier(
            channels[target]["nuisance_model"], channels[target]["prevalence"],
            base_features[1], correct_tables[1]["valid"],
        )
    nuisance_validation_utility = (
        nuisance_validation["benefit"]
        - float(args.risk_penalty) * nuisance_validation["harmful"]
    )
    nuisance_validation_utility[~correct_tables[1]["valid"]] = -np.inf
    nuisance_selected, nuisance_selection_grid = candidate_policy.choose_threshold(
        correct_tables[1], formulas[1], nuisance_validation_utility,
        args.min_selected_formulas,
    )

    evaluated: dict[str, dict[str, object]] = {}
    prediction_detail: dict[str, dict[str, np.ndarray]] = {}
    for name, contrast in contrast_arms[2].items():
        utility, detail = utility_for_contrast(
            channels, base_features[2], contrast, correct_tables[2]["valid"],
            selected_dose, args.risk_penalty,
        )
        rank, selected_candidate, best = rank_at_threshold(
            correct_tables[2], utility, selected_threshold,
        )
        evaluated[name] = {
            "rank": rank, "selected": selected_candidate, "best": best,
            "utility": utility,
            "metric": retrieval(correct_tables[2]["baseline_rank"], rank),
            "ci": bootstrap(
                formulas[2], correct_tables[2]["baseline_rank"], rank,
                args.bootstrap_draws, args.seed + 500,
            ),
        }
        prediction_detail[name] = detail

    primary = evaluated["correct"]
    comparisons = {
        f"correct_minus_{name}": paired_rank_comparison(
            primary["rank"], evaluated[name]["rank"], formulas[2],
            draws=args.bootstrap_draws, seed=args.seed + 600 + index,
        )
        for index, name in enumerate(control_names)
    }

    direct_held_utility, direct_held_benefit, direct_held_harm = direct_utility(
        direct_channels,
        np.concatenate((base_features[2], contrasts[2]), axis=-1),
        correct_tables[2]["valid"], args.risk_penalty,
    )
    direct_rank, direct_slot, direct_best = rank_at_threshold(
        correct_tables[2], direct_held_utility, float(direct_selected["threshold"]),
    )
    nuisance_held = {}
    for target in ("benefit", "harmful"):
        nuisance_held[target] = predict_classifier(
            channels[target]["nuisance_model"], channels[target]["prevalence"],
            base_features[2], correct_tables[2]["valid"],
        )
    nuisance_held_utility = (
        nuisance_held["benefit"] - float(args.risk_penalty) * nuisance_held["harmful"]
    )
    nuisance_held_utility[~correct_tables[2]["valid"]] = -np.inf
    nuisance_rank, nuisance_slot, nuisance_best = rank_at_threshold(
        correct_tables[2], nuisance_held_utility, float(nuisance_selected["threshold"]),
    )
    always_rank, always_slot, always_best = rank_at_threshold(
        correct_tables[2], primary["utility"], -np.inf,
    )
    mechanism_ablation = {
        "same_feature_direct": {
            "selection": direct_selected,
            "selection_grid_size": int(len(direct_selection_grid)),
            "held": retrieval(correct_tables[2]["baseline_rank"], direct_rank),
            "orthogonal_minus_direct": paired_rank_comparison(
                primary["rank"], direct_rank, formulas[2],
                draws=args.bootstrap_draws, seed=args.seed + 720,
            ),
        },
        "nuisance_only": {
            "selection": nuisance_selected,
            "selection_grid_size": int(len(nuisance_selection_grid)),
            "held": retrieval(correct_tables[2]["baseline_rank"], nuisance_rank),
            "orthogonal_minus_nuisance": paired_rank_comparison(
                primary["rank"], nuisance_rank, formulas[2],
                draws=args.bootstrap_draws, seed=args.seed + 721,
            ),
        },
        "no_abstention": {
            "held": retrieval(correct_tables[2]["baseline_rank"], always_rank),
            "orthogonal_abstention_minus_always_act": paired_rank_comparison(
                primary["rank"], always_rank, formulas[2],
                draws=args.bootstrap_draws, seed=args.seed + 722,
            ),
        },
    }
    report = {
        "status": "CHEMAWARE_ORTHOGONAL_RULE_RESIDUAL_POLICY_COMPLETE",
        "selection_control_mode": args.selection_control_mode,
        "selection_control_names": list(control_names),
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "folds 0-1 nuisance cross-fit and residual fit; fold 2 specificity selection; used inner fold 3; fold 4 sealed",
        "claim_limit": (
            "Development candidate policy. Absolute gain includes the non-rule nuisance policy; "
            "only paired correct-minus-control differences can support a chemical claim."
        ),
        "method": {
            "objective": "cross-fitted label residual from official+mass nuisance, predicted by correct-minus-content-permuted rule evidence",
            "residual_weighting": "natural formula-balanced candidate weight; no positive-class rebalance",
            "residual_link": "unclipped linear score; no probability-boundary saturation",
            "selection": (
                "maximize minimum validation risk-utility advantage over: "
                + ", ".join(control_names)
            ),
            "nuisance_feature_names": [candidate_policy.FEATURE_NAMES[i] for i in base_indices],
            "chemical_feature_names": [candidate_policy.FEATURE_NAMES[i] for i in chemical_indices],
            "rule_control_variants": list(args.rule_control_variants),
            "contrast_representation": args.contrast_representation,
            "controls": list(control_names),
            "same_model_threshold_and_dose_for_controls": True,
            "explicit_no_op": True,
        },
        "data": {
            "train_queries": int(len(queries[0])),
            "validation_queries": int(len(queries[1])),
            "inner_queries": int(len(queries[2])),
            "outer_queries_untouched": int(np.sum(fold == 4)),
            "outer_queries_not_scored_by_this_run": int(np.sum(fold == 4)),
            "formula_overlap": 0,
            "candidate_rows": [int(table["valid"].sum()) for table in correct_tables],
        },
        "crossfit_diagnostics": {
            name: channel["diagnostics"] for name, channel in channels.items()
        },
        "selection": selected,
        "selection_grid_size": int(len(selection_grid)),
        "held_inner": {name: result["metric"] for name, result in evaluated.items()},
        "held_inner_absolute_formula_bootstrap_ci95": {
            name: result["ci"] for name, result in evaluated.items()
        },
        "paired_inner": comparisons,
        "mechanism_ablation": mechanism_ablation,
        "gates": {
            "selected_nonzero_chemical_dose": selected_dose > 0,
            "validation_minimum_specific_risk_advantage_positive": selected["minimum_specific_risk_advantage"] > 0,
            "absolute_inner_ci_positive": primary["ci"][0] > 0,
            "corrected_exceeds_twice_introduced": (
                primary["metric"]["corrected_at_1"] > 2 * primary["metric"]["introduced_at_1"]
            ),
            **{
                f"beats_{name}_ci": comparisons[f"correct_minus_{name}"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0
                for name in control_names
            },
            "outer_fold_untouched": True,
            "outer_role_4_not_scored_by_this_run": True,
        },
        "contracts": {
            "nuisance_features_rule_free": all(
                "rule" not in candidate_policy.FEATURE_NAMES[i].lower()
                and "action" not in candidate_policy.FEATURE_NAMES[i].lower()
                for i in base_indices
            ),
            "correct_control_candidate_rows_matched": True,
            "nuisance_features_bit_identical_across_arms": True,
            "formula_crossfit_roles": [0, 1],
            "permutation_preserves_candidate_count_and_baseline_correctness": True,
        },
        "replay_contract": {
            "arguments": {
                key: (str(value.resolve()) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "environment": {
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            },
            "query_sha256": {
                name: array_sha256(query)
                for name, query in zip(("train", "validation", "inner"), queries, strict=True)
            },
            "contrast_sha256": {
                name: array_sha256(contrast)
                for name, contrast in zip(("train", "validation", "inner"), contrasts, strict=True)
            },
            "alignment_permutation_source_sha256": {
                name: array_sha256(source)
                for name, source in zip(("train", "validation", "inner"), permutation_sources, strict=True)
            },
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_orthogonal_rule_", dir=args.output.parent))
    try:
        frozen_bundle = {
            "schema": "chemaware_truthblind_candidate_policy_v1",
            "feature_names": list(candidate_policy.FEATURE_NAMES),
            "actions": [list(action) for action in actions],
            "global_action": int(global_action),
            "channels": channels,
            "dose": float(selected_dose),
            "threshold": float(selected_threshold),
            "risk_penalty": float(args.risk_penalty),
            "rule_key": "rule_response",
            "control_rule_keys": list(args.rule_control_variants),
            "contrast_representation": str(args.contrast_representation),
            "training_formula_roles": [0, 1],
            "selection_formula_role": 2,
            "development_formula_role": 3,
            "outer_formula_role": 4,
            "baselines": {
                "same_feature_direct": {
                    "channels": direct_channels,
                    "threshold": float(direct_selected["threshold"]),
                    "feature_layout": "nuisance_then_correct_minus_control_chemical",
                },
                "nuisance_only": {
                    "threshold": float(nuisance_selected["threshold"]),
                    "model_source": "primary_channel_nuisance_models",
                    "feature_layout": "nuisance_only",
                },
            },
        }
        policy_path = temporary / "truthblind_policy.joblib"
        joblib.dump(frozen_bundle, policy_path, compress=3)
        policy_metadata = {
            key: value for key, value in frozen_bundle.items()
            if key not in {"channels", "baselines"}
        }
        policy_metadata["channel_contract"] = {
            name: {
                "active_features": np.asarray(channel["active"], dtype=np.int64).tolist(),
                "prevalence": float(channel["prevalence"]),
                "nuisance_model_type": type(channel["nuisance_model"]).__name__,
                "residual_model_type": type(channel["residual_model"]).__name__,
            }
            for name, channel in channels.items()
        }
        policy_metadata["baseline_contract"] = {
            "same_feature_direct": {
                "threshold": float(direct_selected["threshold"]),
                "feature_layout": "nuisance_then_correct_minus_control_chemical",
                "benefit_model_type": type(direct_channels["benefit"]["model"]).__name__,
                "harmful_model_type": type(direct_channels["harmful"]["model"]).__name__,
            },
            "nuisance_only": {
                "threshold": float(nuisance_selected["threshold"]),
                "feature_layout": "nuisance_only",
                "model_source": "primary_channel_nuisance_models",
            },
            "selection_data": "formula role 2 only",
            "outer_tuning": False,
        }
        (temporary / "truthblind_policy.json").write_text(
            json.dumps(policy_metadata, indent=2), encoding="utf-8",
        )
        report["frozen_truthblind_policy"] = {
            "path": "truthblind_policy.joblib",
            "sha256": sha256(policy_path),
            "metadata_path": "truthblind_policy.json",
            "truth_fields_accepted_by_inference": [],
            "deployment_inputs": [
                "query spectrum", "candidate reference spectra",
                "official DreaMS scores", "mass scores", "rule scores",
            ],
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz",
            query=queries[2], formula=formulas[2],
            identity=body["query_ik14"][queries[2]].astype(str),
            baseline_rank=correct_tables[2]["baseline_rank"],
            **{f"{name}_rank": result["rank"] for name, result in evaluated.items()},
            **{f"{name}_selected_candidate_slot": result["selected"] for name, result in evaluated.items()},
            **{f"{name}_candidate_utility": result["utility"] for name, result in evaluated.items()},
            valid_candidate=correct_tables[2]["valid"],
            proposed_candidate=correct_tables[2]["proposed_candidate"],
            proposal_rank=correct_tables[2]["rank"],
            baseline_candidate=correct_tables[2]["baseline_candidate"],
            selected_candidate_slot=primary["selected"],
            best_predicted_utility=primary["best"],
            same_feature_direct_rank=direct_rank,
            same_feature_direct_selected_candidate_slot=direct_slot,
            same_feature_direct_best_utility=direct_best,
            same_feature_direct_candidate_utility=direct_held_utility,
            nuisance_only_rank=nuisance_rank,
            nuisance_only_selected_candidate_slot=nuisance_slot,
            nuisance_only_best_utility=nuisance_best,
            nuisance_only_candidate_utility=nuisance_held_utility,
            no_abstention_rank=always_rank,
            no_abstention_selected_candidate_slot=always_slot,
            no_abstention_best_utility=always_best,
            selected_dose=np.asarray(selected_dose),
            selected_threshold=np.asarray(selected_threshold),
        )
        validation_predictions = {}
        for name, utility in validation_utilities[selected_dose].items():
            rank, selected_candidate, best = rank_at_threshold(
                correct_tables[1], utility, selected_threshold,
            )
            validation_predictions[name] = {
                "rank": rank, "selected": selected_candidate,
                "best": best, "utility": utility,
            }
        direct_validation_rank, direct_validation_slot, direct_validation_best = rank_at_threshold(
            correct_tables[1], direct_validation_utilities[1.0]["correct"],
            float(direct_selected["threshold"]),
        )
        nuisance_validation_rank, nuisance_validation_slot, nuisance_validation_best = rank_at_threshold(
            correct_tables[1], nuisance_validation_utility,
            float(nuisance_selected["threshold"]),
        )
        np.savez_compressed(
            temporary / "validation_policy.npz",
            query=queries[1], formula=formulas[1],
            identity=body["query_ik14"][queries[1]].astype(str),
            baseline_rank=correct_tables[1]["baseline_rank"],
            **{
                f"{name}_rank": result["rank"]
                for name, result in validation_predictions.items()
            },
            **{
                f"{name}_selected_candidate_slot": result["selected"]
                for name, result in validation_predictions.items()
            },
            **{
                f"{name}_candidate_utility": result["utility"]
                for name, result in validation_predictions.items()
            },
            valid_candidate=correct_tables[1]["valid"],
            proposed_candidate=correct_tables[1]["proposed_candidate"],
            proposal_rank=correct_tables[1]["rank"],
            baseline_candidate=correct_tables[1]["baseline_candidate"],
            same_feature_direct_rank=direct_validation_rank,
            same_feature_direct_selected_candidate_slot=direct_validation_slot,
            same_feature_direct_best_utility=direct_validation_best,
            same_feature_direct_candidate_utility=direct_validation_utilities[1.0]["correct"],
            nuisance_only_rank=nuisance_validation_rank,
            nuisance_only_selected_candidate_slot=nuisance_validation_slot,
            nuisance_only_best_utility=nuisance_validation_best,
            nuisance_only_candidate_utility=nuisance_validation_utility,
            selected_dose=np.asarray(selected_dose),
            selected_threshold=np.asarray(selected_threshold),
        )
        np.savez_compressed(
            temporary / "train_triplet_evidence.npz",
            **triplet_evidence_payload(
                queries[0], formulas[0], body["query_ik14"][queries[0]],
                correct_tables[0], control_table_sets[0],
                tuple(args.rule_control_variants),
                len(actions),
            ),
        )
        np.savez_compressed(
            temporary / "selection_triplet_evidence.npz",
            **triplet_evidence_payload(
                queries[1], formulas[1], body["query_ik14"][queries[1]],
                correct_tables[1], control_table_sets[1],
                tuple(args.rule_control_variants),
                len(actions),
            ),
        )
        np.savez_compressed(
            temporary / "confirmation_triplet_evidence.npz",
            **triplet_evidence_payload(
                queries[2], formulas[2], body["query_ik14"][queries[2]],
                correct_tables[2], control_table_sets[2],
                tuple(args.rule_control_variants),
                len(actions),
            ),
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selection": report["selection"],
        "held_inner": report["held_inner"], "paired_inner": report["paired_inner"],
        "gates": report["gates"], "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
