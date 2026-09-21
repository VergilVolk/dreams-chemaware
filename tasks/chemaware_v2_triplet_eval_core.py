"""Pure NumPy metrics for ChemAware V2 direct-triplet evaluation."""
from __future__ import annotations

import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact ROC AUC via average ranks, including score ties."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive_n = int(labels.sum())
    negative_n = int((~labels).sum())
    if not positive_n or not negative_n:
        raise ValueError("binary AUC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    left = 0
    while left < len(scores):
        right = left + 1
        while right < len(scores) and sorted_scores[right] == sorted_scores[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * ((left + 1) + right)
        left = right
    statistic = ranks[labels].sum() - positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def evaluate_graph(
    encoded: np.ndarray, rows: np.ndarray, manifest: dict[str, np.ndarray],
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = {int(row): index for index, row in enumerate(rows)}
    ranks = np.empty(len(queries), dtype=np.int32)
    positive_scores = np.empty(len(queries), dtype=np.float32)
    negative_scores: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for output_row, query in enumerate(queries):
        query = int(query)
        molecule_left, molecule_right = map(
            int, manifest["query_ptr"][query:query + 2],
        )
        pair_left = int(manifest["molecule_ptr"][molecule_left])
        pair_right = int(manifest["molecule_ptr"][molecule_right])
        candidate_rows = manifest["pair_candidate_row"][pair_left:pair_right]
        candidate_position = np.asarray([position[int(row)] for row in candidate_rows])
        query_position = position[int(manifest["query_row"][query])]
        pair_scores = encoded[candidate_position] @ encoded[query_position]
        local_ptr = manifest["molecule_ptr"][molecule_left:molecule_right + 1] - pair_left
        molecule_scores = np.maximum.reduceat(pair_scores, local_ptr[:-1])
        ranks[output_row] = 1 + int(np.sum(molecule_scores[1:] >= molecule_scores[0]))
        positive_scores[output_row] = molecule_scores[0]
        negative_scores.append(np.asarray(molecule_scores[1:], dtype=np.float32))
        all_scores.append(np.asarray(molecule_scores, dtype=np.float32))
        all_labels.append(np.r_[1, np.zeros(len(molecule_scores) - 1, dtype=np.int8)])
    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(all_labels)
    macro_auc = np.mean([
        (np.sum(pos > neg) + 0.5 * np.sum(pos == neg)) / len(neg)
        for pos, neg in zip(positive_scores, negative_scores, strict=True)
    ])
    return ranks, positive_scores, np.asarray([
        np.max(negative) for negative in negative_scores
    ]), np.asarray([binary_auc(flat_labels, flat_scores), macro_auc])


def summarize(ranks: np.ndarray, positive: np.ndarray, negative: np.ndarray, auc: np.ndarray) -> dict:
    return {
        "queries": int(len(ranks)),
        "recall1": float(np.mean(ranks <= 1)),
        "recall3": float(np.mean(ranks <= 3)),
        "recall5": float(np.mean(ranks <= 5)),
        "recall10": float(np.mean(ranks <= 10)),
        "recall20": float(np.mean(ranks <= 20)),
        "recall50": float(np.mean(ranks <= 50)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_positive_margin": float(np.mean(positive - negative)),
        "micro_auc": float(auc[0]),
        "macro_auc": float(auc[1]),
    }


def paired_summary(
    baseline: np.ndarray, current: np.ndarray, formula: np.ndarray,
    draws: int, seed: int,
) -> dict:
    delta = (current <= 1).astype(np.float64) - (baseline <= 1).astype(np.float64)
    formulas, inverse = np.unique(formula.astype(str), return_inverse=True)
    cluster_sum = np.bincount(inverse, weights=delta)
    cluster_count = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(formulas), size=len(formulas))
        samples[draw] = cluster_sum[selected].sum() / cluster_count[selected].sum()
    return {
        "delta_recall1": float(np.mean(delta)),
        "delta_mrr": float(np.mean(1.0 / current) - np.mean(1.0 / baseline)),
        "corrected_at_1": int(np.sum((baseline > 1) & (current == 1))),
        "introduced_at_1": int(np.sum((baseline == 1) & (current > 1))),
        "formula_cluster_bootstrap_delta_recall1_ci95": [
            float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)),
        ],
    }
