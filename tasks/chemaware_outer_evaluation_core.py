"""Pure evaluation helpers for the sealed ChemAware outer fold."""
from __future__ import annotations

import numpy as np


RECALL_K = (1, 3, 5, 10, 20, 50)


def strict_rank(score: np.ndarray, label: np.ndarray) -> int:
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    if score.ndim != 1 or label.shape != score.shape or not np.any(label):
        raise ValueError("candidate scores and labels are invalid")
    positive = float(score[np.flatnonzero(label)[0]])
    return 1 + int(np.sum(score[~label] >= positive))


def promoted_scores(score: np.ndarray, selected_candidate: int) -> np.ndarray:
    output = np.asarray(score, dtype=np.float64).copy()
    selected = int(selected_candidate)
    if selected >= 0:
        if selected >= len(output):
            raise IndexError("selected candidate is outside the query")
        output[selected] = np.nextafter(float(np.max(output)), np.inf)
    return output


def evaluate_selections(
    candidate_ptr: np.ndarray, official_score: np.ndarray, label: np.ndarray,
    selections: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    ptr = np.asarray(candidate_ptr, dtype=np.int64)
    score = np.asarray(official_score, dtype=np.float64)
    truth = np.asarray(label, dtype=bool)
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or ptr[-1] != len(score):
        raise ValueError("candidate pointer is invalid")
    if truth.shape != score.shape or np.any(np.diff(ptr) < 1):
        raise ValueError("candidate labels are not aligned")
    query_count = len(ptr) - 1
    if any(np.asarray(value).shape != (query_count,) for value in selections.values()):
        raise ValueError("selection arrays are not aligned")
    baseline = np.empty(query_count, dtype=np.int16)
    ranks = {name: np.empty(query_count, dtype=np.int16) for name in selections}
    proposed_score = {name: score.copy() for name in selections}
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:], strict=True)):
        local_score = score[int(left):int(right)]
        local_truth = truth[int(left):int(right)]
        baseline[query] = strict_rank(local_score, local_truth)
        for name, selected in selections.items():
            promoted = promoted_scores(local_score, int(selected[query]))
            proposed_score[name][int(left):int(right)] = promoted
            ranks[name][query] = strict_rank(promoted, local_truth)
    return baseline, ranks, proposed_score


def retrieval_metrics(baseline_rank: np.ndarray, rank: np.ndarray) -> dict[str, object]:
    baseline = np.asarray(baseline_rank, dtype=np.int64)
    proposed = np.asarray(rank, dtype=np.int64)
    if baseline.shape != proposed.shape or baseline.ndim != 1:
        raise ValueError("retrieval ranks are not aligned")
    output: dict[str, object] = {
        "queries": int(len(baseline)),
        "baseline_mrr": float(np.mean(1.0 / baseline)),
        "mrr": float(np.mean(1.0 / proposed)),
        "delta_mrr": float(np.mean(1.0 / proposed - 1.0 / baseline)),
        "corrected_at_1": int(np.sum((baseline > 1) & (proposed == 1))),
        "introduced_at_1": int(np.sum((baseline == 1) & (proposed > 1))),
        "rank_improved": int(np.sum(proposed < baseline)),
        "rank_worsened": int(np.sum(proposed > baseline)),
    }
    for k in RECALL_K:
        output[f"baseline_recall{k}"] = float(np.mean(baseline <= k))
        output[f"recall{k}"] = float(np.mean(proposed <= k))
        output[f"delta_recall{k}"] = float(np.mean(proposed <= k) - np.mean(baseline <= k))
    output["risk_utility_at_1"] = (
        int(output["corrected_at_1"]) - 2 * int(output["introduced_at_1"])
    )
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * ((left + 1) + right)
        left = right
    return ranks


def binary_auc(score: np.ndarray, label: np.ndarray) -> float:
    values = np.asarray(score, dtype=np.float64)
    truth = np.asarray(label, dtype=bool)
    positive = int(np.sum(truth)); negative = int(np.sum(~truth))
    if values.ndim != 1 or truth.shape != values.shape or not positive or not negative:
        return float("nan")
    rank = _average_ranks(values)
    return float(
        (np.sum(rank[truth]) - positive * (positive + 1) / 2.0) / (positive * negative)
    )


def auc_metrics(candidate_ptr: np.ndarray, score: np.ndarray, label: np.ndarray) -> dict[str, object]:
    ptr = np.asarray(candidate_ptr, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    truth = np.asarray(label, dtype=bool)
    per_query = []
    for left, right in zip(ptr[:-1], ptr[1:], strict=True):
        value = binary_auc(values[int(left):int(right)], truth[int(left):int(right)])
        if np.isfinite(value):
            per_query.append(value)
    return {
        "micro_auc": binary_auc(values, truth),
        "macro_auc": float(np.mean(per_query)) if per_query else None,
        "macro_auc_queries": int(len(per_query)),
    }


def formula_cluster_ci(
    formula: np.ndarray, left_rank: np.ndarray, right_rank: np.ndarray,
    *, draws: int, seed: int, k: int = 1,
) -> list[float]:
    formula = np.asarray(formula, dtype=str)
    left = np.asarray(left_rank, dtype=np.int64)
    right = np.asarray(right_rank, dtype=np.int64)
    if not (formula.shape == left.shape == right.shape) or draws <= 0:
        raise ValueError("formula bootstrap inputs are invalid")
    unique, inverse = np.unique(formula, return_inverse=True)
    delta = (left <= int(k)).astype(float) - (right <= int(k)).astype(float)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        selected = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        samples[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def paired_comparison(
    left_rank: np.ndarray, right_rank: np.ndarray, formula: np.ndarray,
    *, draws: int, seed: int,
) -> dict[str, object]:
    left = np.asarray(left_rank, dtype=np.int64)
    right = np.asarray(right_rank, dtype=np.int64)
    return {
        "delta_recall1": float(np.mean(left == 1) - np.mean(right == 1)),
        "delta_mrr": float(np.mean(1.0 / left - 1.0 / right)),
        "left_correct_relative_to_right": int(np.sum((left == 1) & (right > 1))),
        "left_errors_relative_to_right": int(np.sum((left > 1) & (right == 1))),
        "formula_cluster_bootstrap_delta_recall1_ci95": formula_cluster_ci(
            formula, left, right, draws=draws, seed=seed,
        ),
    }
