"""Leakage-resistant query-level expert routing for the grand-fusion study.

The router never learns a new candidate score.  It selects, per query, one
already-frozen expert score vector.  Features are scale-free summaries of
agreement, confidence gaps and cross-expert ranks; candidate count and
reference-spectrum multiplicity are deliberately excluded.

Labels are used only by the trainer.  Application uses :func:`query_features`
and frozen per-expert correctness models.  Tied top scores are always treated
as abstentions/failures, matching the project's strict retrieval contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


MODEL_SETTINGS = dict(
    max_iter=160,
    learning_rate=0.06,
    max_leaf_nodes=15,
    min_samples_leaf=80,
    l2_regularization=2.0,
    early_stopping=False,
    random_state=20261004,
)


def _strict_winner(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        return -1
    winners = np.flatnonzero(values == np.max(values))
    return int(winners[0]) if len(winners) == 1 else -1


def _descending_percentile(values: np.ndarray) -> np.ndarray:
    """Return 1 for best and 0 for worst, with exact ties averaged."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = np.mean(ranks[order[start:stop]])
        start = stop
    return ranks / max(len(values) - 1, 1)


def _normalized_gap(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    spread = float(ordered[0] - ordered[-1])
    return float((ordered[0] - ordered[1]) / spread) if spread > 0 else 0.0


def feature_names(methods: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for method in methods:
        names.extend((
            f"gap::{method}",
            f"winner_support::{method}",
            f"winner_mean_percentile::{method}",
            f"winner_min_percentile::{method}",
        ))
    for first_index, first in enumerate(methods):
        for second in methods[first_index + 1:]:
            names.append(f"same_winner::{first}::{second}")
    return tuple(names)


def query_features(
    molecule_scores: dict[str, np.ndarray],
    query_ptr: np.ndarray,
    methods: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build label-free query features and strict local winners."""
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    if query_ptr.ndim != 1 or len(query_ptr) < 2 or query_ptr[0] != 0:
        raise ValueError("invalid query_ptr")
    molecules = int(query_ptr[-1])
    for method in methods:
        value = np.asarray(molecule_scores[method], dtype=np.float64)
        if value.shape != (molecules,) or not np.all(np.isfinite(value)):
            raise ValueError(f"malformed molecule scores for {method}")

    output: list[list[float]] = []
    winners = np.full((len(query_ptr) - 1, len(methods)), -1, dtype=np.int64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        blocks = [np.asarray(molecule_scores[name][left:right], dtype=np.float64) for name in methods]
        local_winners = [_strict_winner(block) for block in blocks]
        winners[query] = local_winners
        percentiles = [_descending_percentile(block) for block in blocks]
        row: list[float] = []
        for method_index, block in enumerate(blocks):
            winner = local_winners[method_index]
            if winner < 0:
                support = mean_percentile = min_percentile = 0.0
            else:
                support = sum(other == winner for other in local_winners) / len(methods)
                at_winner = np.asarray([rank[winner] for rank in percentiles], dtype=np.float64)
                mean_percentile = float(at_winner.mean())
                min_percentile = float(at_winner.min())
            row.extend((_normalized_gap(block), support, mean_percentile, min_percentile))
        for first in range(len(methods)):
            for second in range(first + 1, len(methods)):
                row.append(float(local_winners[first] >= 0 and local_winners[first] == local_winners[second]))
        output.append(row)
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (len(query_ptr) - 1, len(feature_names(methods))):
        raise RuntimeError("query feature contract drifted")
    return features, winners


class ConstantProbability:
    """Small sklearn-compatible fallback for a constant correctness label."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        p = np.full(len(features), self.probability, dtype=np.float64)
        return np.column_stack((1.0 - p, p))


def fit_correctness_models(
    features: np.ndarray,
    correctness: np.ndarray,
    methods: tuple[str, ...],
) -> dict[str, object]:
    models: dict[str, object] = {}
    for column, method in enumerate(methods):
        target = np.asarray(correctness[:, column], dtype=np.int8)
        if len(np.unique(target)) == 1:
            models[method] = ConstantProbability(float(target[0]))
        else:
            model = HistGradientBoostingClassifier(**MODEL_SETTINGS)
            model.fit(features, target)
            models[method] = model
    return models


def predict_correctness(
    models: dict[str, object], features: np.ndarray, methods: tuple[str, ...],
) -> np.ndarray:
    return np.column_stack([
        np.asarray(models[method].predict_proba(features), dtype=np.float64)[:, 1]
        for method in methods
    ])


def correctness_from_winners(winners: np.ndarray, query_ptr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    result = np.zeros_like(winners, dtype=np.int8)
    for query, left in enumerate(np.asarray(query_ptr[:-1], dtype=np.int64)):
        for method in range(winners.shape[1]):
            winner = int(winners[query, method])
            result[query, method] = int(winner >= 0 and labels[int(left) + winner] == 1)
    return result


def route(
    probabilities: np.ndarray,
    winners: np.ndarray,
    methods: tuple[str, ...],
    default_method: str,
    threshold: float,
) -> np.ndarray:
    """Select an expert only when its predicted advantage clears threshold."""
    default = methods.index(default_method)
    chosen = np.full(len(probabilities), default, dtype=np.int64)
    for query in range(len(probabilities)):
        eligible = np.flatnonzero(winners[query] >= 0)
        if len(eligible) == 0:
            continue
        best = int(eligible[np.argmax(probabilities[query, eligible])])
        if best != default and probabilities[query, best] - probabilities[query, default] >= threshold:
            chosen[query] = best
    return chosen


def risk_ledger(
    correctness: np.ndarray, chosen: np.ndarray, methods: tuple[str, ...],
    default_method: str, risk_lambda: float = 2.0,
) -> dict[str, float | int]:
    default = methods.index(default_method)
    rows = np.arange(len(chosen))
    base = correctness[:, default].astype(bool)
    selected = correctness[rows, chosen].astype(bool)
    corrected = int((~base & selected).sum())
    introduced = int((base & ~selected).sum())
    return {
        "queries": int(len(chosen)),
        "switches": int((chosen != default).sum()),
        "default_correct": int(base.sum()),
        "selected_correct": int(selected.sum()),
        "corrected": corrected,
        "introduced": introduced,
        "net": corrected - introduced,
        "risk_net": corrected - risk_lambda * introduced,
        "accuracy": float(selected.mean()),
        "default_accuracy": float(base.mean()),
    }


def select_threshold(
    probabilities: np.ndarray,
    winners: np.ndarray,
    correctness: np.ndarray,
    methods: tuple[str, ...],
    default_method: str,
    thresholds: tuple[float, ...],
    risk_lambda: float,
) -> tuple[float, list[dict[str, float | int]]]:
    rows = []
    for threshold in thresholds:
        chosen = route(probabilities, winners, methods, default_method, threshold)
        ledger = risk_ledger(correctness, chosen, methods, default_method, risk_lambda)
        ledger["threshold"] = float(threshold)
        rows.append(ledger)
    # First protect risk-net, then ordinary net, then prefer fewer switches and
    # the more conservative (larger) threshold.
    best = max(rows, key=lambda row: (
        row["risk_net"], row["net"], -row["switches"], row["threshold"],
    ))
    return float(best["threshold"]), rows


@dataclass(frozen=True)
class RouterContract:
    methods: tuple[str, ...]
    default_method: str
    threshold: float
    risk_lambda: float

