"""Leakage-safe scalar edge calibration and paired target/decoy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class PlattModel:
    coefficient: float
    intercept: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        logits = self.coefficient * np.asarray(values, dtype=float) + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_platt(values: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> PlattModel:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if not (len(values) == len(labels) == len(weights)) or len(np.unique(labels)) != 2:
        raise ValueError("Platt calibration requires aligned examples from both classes")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("calibration values and weights must be finite and weights positive")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=20260831)
    model.fit(values[:, None], labels, sample_weight=weights)
    coefficient = float(model.coef_[0, 0])
    if coefficient <= 0:
        raise RuntimeError(f"edge score is not positively oriented after fitting: coefficient={coefficient}")
    return PlattModel(coefficient=coefficient, intercept=float(model.intercept_[0]))


def _expanded(
    positive: np.ndarray, negative: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate((positive[mask], negative[mask])),
        np.concatenate((np.ones(mask.sum(), dtype=int), np.zeros(mask.sum(), dtype=int))),
        np.concatenate((weights[mask], weights[mask])),
    )


def _derived_probabilities(base: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = dict(base)
    if {"author_dp", "official_dreams"}.issubset(base):
        result["author_official_intersection"] = (
            np.minimum(base["author_dp"][0], base["official_dreams"][0]),
            np.minimum(base["author_dp"][1], base["official_dreams"][1]),
        )
    if {"author_dp", "noise_tuned_dreams"}.issubset(base):
        result["author_noise_intersection"] = (
            np.minimum(base["author_dp"][0], base["noise_tuned_dreams"][0]),
            np.minimum(base["author_dp"][1], base["noise_tuned_dreams"][1]),
        )
    return result


def choose_target_decoy_threshold(
    positive: np.ndarray, negative: np.ndarray, weights: np.ndarray, alpha: float
) -> float:
    """Choose the most permissive training-only threshold with decoy/target <= alpha."""
    thresholds = np.unique(np.concatenate((positive, negative)))[::-1]
    best = np.inf
    best_target = -1.0
    for threshold in thresholds:
        target = float(weights[positive >= threshold].sum())
        decoy = float(weights[negative >= threshold].sum())
        fdr = decoy / max(target, 1e-12)
        if fdr <= alpha and target > best_target:
            best_target = target
            best = float(threshold)
    return best


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def cluster_bootstrap_delta(
    values: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    *,
    resamples: int = 5000,
    seed: int = 20260831,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    weights = np.asarray(weights, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    observed = _weighted_mean(values - reference, weights)
    draws = np.empty(resamples, dtype=float)
    indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    for draw in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        draws[draw] = _weighted_mean(values[selected] - reference[selected], weights[selected])
    return {
        "mean_delta": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(unique)),
        "resamples": int(resamples),
    }


def crossfit_edge_scores(
    raw_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    folds: np.ndarray,
    weights: np.ndarray,
    components: np.ndarray,
    formulas: np.ndarray,
    *,
    fdr_levels: tuple[float, ...] = (0.01, 0.05, 0.10),
    bootstrap_resamples: int = 5000,
) -> tuple[dict[str, object], dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Nested component-fold calibration; thresholds never inspect their outer fold."""
    folds = np.asarray(folds, dtype=int)
    weights = np.asarray(weights, dtype=float)
    components = np.asarray(components, dtype=object)
    formulas = np.asarray(formulas, dtype=object)
    n = len(folds)
    if n == 0 or len(np.unique(folds)) < 4:
        raise ValueError("at least four non-empty component folds are required")
    if not (len(weights) == len(components) == len(formulas) == n):
        raise ValueError("fold, weight, component and formula arrays must align")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("edge weights must be finite and positive")
    component_fold_counts = {
        component: len(np.unique(folds[components == component]))
        for component in np.unique(components)
    }
    leaking_components = [component for component, count in component_fold_counts.items() if count != 1]
    if leaking_components:
        raise RuntimeError(
            "component leakage across outer folds: "
            f"{len(leaking_components)} components; examples={leaking_components[:5]}"
        )
    for name, (positive, negative) in raw_scores.items():
        if len(positive) != n or len(negative) != n:
            raise ValueError(f"unaligned score arm: {name}")
        if np.any(~np.isfinite(positive)) or np.any(~np.isfinite(negative)):
            raise ValueError(f"non-finite score arm: {name}")

    arm_names = list(raw_scores)
    derived_names = ["author_official_intersection"]
    if "noise_tuned_dreams" in raw_scores:
        derived_names.append("author_noise_intersection")
    all_names = arm_names + derived_names
    oof = {name: (np.full(n, np.nan), np.full(n, np.nan)) for name in all_names}
    accepted = {
        name: {alpha: (np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)) for alpha in fdr_levels}
        for name in all_names
    }
    threshold_log: dict[str, dict[str, float | None]] = {name: {} for name in all_names}

    unique_folds = np.unique(folds)
    for outer in unique_folds:
        train = folds != outer
        test = folds == outer
        base_test: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (positive, negative) in raw_scores.items():
            values, labels, sample_weights = _expanded(positive, negative, weights, train)
            calibration = fit_platt(values, labels, sample_weights)
            base_test[name] = (calibration.predict(positive[test]), calibration.predict(negative[test]))
        test_probabilities = _derived_probabilities(base_test)
        for name, (positive_probability, negative_probability) in test_probabilities.items():
            oof[name][0][test] = positive_probability
            oof[name][1][test] = negative_probability

        # Inner cross-fitting produces threshold-selection probabilities without
        # scoring an example using a calibrator fitted on that same component.
        inner_probability = {
            name: (np.full(n, np.nan), np.full(n, np.nan)) for name in all_names
        }
        for inner in unique_folds[unique_folds != outer]:
            inner_train = (folds != outer) & (folds != inner)
            inner_test = folds == inner
            base_inner: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name, (positive, negative) in raw_scores.items():
                values, labels, sample_weights = _expanded(positive, negative, weights, inner_train)
                calibration = fit_platt(values, labels, sample_weights)
                base_inner[name] = (
                    calibration.predict(positive[inner_test]),
                    calibration.predict(negative[inner_test]),
                )
            for name, (positive_probability, negative_probability) in _derived_probabilities(base_inner).items():
                inner_probability[name][0][inner_test] = positive_probability
                inner_probability[name][1][inner_test] = negative_probability

        for name in all_names:
            positive_inner, negative_inner = inner_probability[name]
            if np.any(~np.isfinite(positive_inner[train])) or np.any(~np.isfinite(negative_inner[train])):
                raise RuntimeError(f"incomplete nested probabilities for {name}, outer fold {outer}")
            for alpha in fdr_levels:
                threshold = choose_target_decoy_threshold(
                    positive_inner[train], negative_inner[train], weights[train], alpha
                )
                threshold_log[name][f"fold_{outer}|fdr_{alpha:.2f}"] = (
                    float(threshold) if np.isfinite(threshold) else None
                )
                accepted[name][alpha][0][test] = test_probabilities[name][0] >= threshold
                accepted[name][alpha][1][test] = test_probabilities[name][1] >= threshold

    for name, (positive, negative) in oof.items():
        if np.any(~np.isfinite(positive)) or np.any(~np.isfinite(negative)):
            raise RuntimeError(f"OOF calibration is incomplete for {name}")

    report: dict[str, object] = {}
    if "author_dp" not in oof:
        raise ValueError("raw_scores must contain the exact author_dp arm")
    author_accuracy = (oof["author_dp"][0] > oof["author_dp"][1]).astype(float)
    for name, (positive, negative) in oof.items():
        accuracy = (positive > negative).astype(float)
        edge_values = np.concatenate((positive, negative))
        edge_labels = np.concatenate((np.ones(n, dtype=int), np.zeros(n, dtype=int)))
        edge_weights = np.concatenate((weights, weights))
        corrected = int(np.sum((author_accuracy == 0) & (accuracy == 1)))
        introduced = int(np.sum((author_accuracy == 1) & (accuracy == 0)))
        corrected_weight = float(weights[(author_accuracy == 0) & (accuracy == 1)].sum())
        introduced_weight = float(weights[(author_accuracy == 1) & (accuracy == 0)].sum())
        fdr_report: dict[str, object] = {}
        for alpha in fdr_levels:
            target_selected, decoy_selected = accepted[name][alpha]
            target_weight = float(weights[target_selected].sum())
            decoy_weight = float(weights[decoy_selected].sum())
            fdr_report[f"{alpha:.2f}"] = {
                "target_recall": target_weight / float(weights.sum()),
                "decoy_recall": decoy_weight / float(weights.sum()),
                "empirical_fdr": decoy_weight / max(target_weight, 1e-12),
            }
        report[name] = {
            "paired_accuracy": _weighted_mean(accuracy, weights),
            "paired_margin": _weighted_mean(positive - negative, weights),
            "average_precision": float(average_precision_score(edge_labels, edge_values, sample_weight=edge_weights)),
            "weighted_brier": _weighted_mean((edge_values - edge_labels) ** 2, edge_weights),
            "corrected_vs_author": corrected,
            "introduced_vs_author": introduced,
            "edge_weighted_corrected_vs_author": corrected_weight,
            "edge_weighted_introduced_vs_author": introduced_weight,
            "component_bootstrap_accuracy_delta_vs_author": cluster_bootstrap_delta(
                accuracy, author_accuracy, weights, components, resamples=bootstrap_resamples
            ),
            "formula_bootstrap_accuracy_delta_vs_author": cluster_bootstrap_delta(
                accuracy, author_accuracy, weights, formulas, resamples=bootstrap_resamples, seed=20260832
            ),
            "fixed_fdr": fdr_report,
        }

    full_artifacts: dict[str, object] = {}
    full_base_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mask = np.ones(n, dtype=bool)
    for name, (positive, negative) in raw_scores.items():
        values, labels, sample_weights = _expanded(positive, negative, weights, mask)
        model = fit_platt(values, labels, sample_weights)
        full_artifacts[name] = {
            "kind": "scalar_platt_logistic",
            "coefficient": model.coefficient,
            "intercept": model.intercept,
        }
        full_base_probabilities[name] = (model.predict(positive), model.predict(negative))
    full_artifacts["derived_arms"] = {
        "author_official_intersection": "min(P_author, P_official_dreams)",
        **(
            {"author_noise_intersection": "min(P_author, P_noise_tuned_dreams)"}
            if "noise_tuned_dreams" in raw_scores
            else {}
        ),
    }
    full_artifacts["outer_thresholds"] = threshold_log
    # OOF thresholds belong only to cross-fitted evaluation.  The external
    # deployment artifact uses a full-data calibration refit and must select
    # its threshold on probabilities from that same refit, otherwise the
    # probability scale and the threshold can silently differ.
    oof_thresholds: dict[str, dict[str, float | None]] = {}
    for name, (positive, negative) in oof.items():
        oof_thresholds[name] = {}
        for alpha in fdr_levels:
            threshold = choose_target_decoy_threshold(positive, negative, weights, alpha)
            oof_thresholds[name][f"fdr_{alpha:.2f}"] = (
                float(threshold) if np.isfinite(threshold) else None
            )
    full_artifacts["evaluation_thresholds_from_nested_oof"] = oof_thresholds

    deployment_thresholds: dict[str, dict[str, float | None]] = {}
    for name, (positive, negative) in _derived_probabilities(full_base_probabilities).items():
        deployment_thresholds[name] = {}
        for alpha in fdr_levels:
            threshold = choose_target_decoy_threshold(positive, negative, weights, alpha)
            deployment_thresholds[name][f"fdr_{alpha:.2f}"] = (
                float(threshold) if np.isfinite(threshold) else None
            )
    full_artifacts["deployment_thresholds_full_refit"] = deployment_thresholds
    return report, oof, full_artifacts
