"""Pure NumPy metrics for ChemAware V2 direct-triplet evaluation."""
from __future__ import annotations

import numpy as np


def numerical_rank_replay_audit(
    encoded: np.ndarray,
    rows: np.ndarray,
    manifest: dict[str, np.ndarray],
    queries: np.ndarray,
    expected_rank: np.ndarray,
    observed_rank: np.ndarray,
    *,
    tie_tolerance: float = 5e-7,
    maximum_fraction: float = 0.005,
    maximum_count: int = 3,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Identify only rank changes explainable by float32 boundary ties.

    A frozen rank is replay-compatible when it falls inside the rank interval
    obtained by moving every negative within ``tie_tolerance`` of the positive
    to either side of the boundary.  This is stricter than merely allowing a
    small mismatch count: any mismatch away from a numerical tie fails closed.
    """
    queries = np.asarray(queries, dtype=np.int64)
    expected_rank = np.asarray(expected_rank, dtype=np.int64)
    observed_rank = np.asarray(observed_rank, dtype=np.int64)
    if len(queries) != len(expected_rank) or len(queries) != len(observed_rank):
        raise ValueError("rank replay arrays do not align")
    if tie_tolerance < 0 or not 0 < maximum_fraction <= 0.01 or maximum_count < 0:
        raise ValueError("invalid numerical replay audit tolerance")

    position = {int(row): index for index, row in enumerate(rows)}
    mismatch = np.flatnonzero(observed_rank != expected_rank)
    stable = np.ones(len(queries), dtype=bool)
    stable[mismatch] = False
    audit: list[dict[str, object]] = []
    unexplained = []
    for output_row in mismatch:
        query = int(queries[output_row])
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
        positive = float(molecule_scores[0])
        negative = np.asarray(molecule_scores[1:], dtype=np.float64)
        gaps = negative - positive
        minimum_rank = 1 + int(np.sum(gaps > tie_tolerance))
        maximum_rank = 1 + int(np.sum(gaps >= -tie_tolerance))
        expected = int(expected_rank[output_row])
        explained = minimum_rank <= expected <= maximum_rank
        nearest_gap = float(np.min(np.abs(gaps)))
        row = {
            "policy_position": int(output_row),
            "manifest_query": query,
            "expected_rank": expected,
            "observed_rank": int(observed_rank[output_row]),
            "tie_compatible_rank_interval": [minimum_rank, maximum_rank],
            "nearest_negative_positive_score_gap": nearest_gap,
            "tie_tolerance": float(tie_tolerance),
            "tie_explained": bool(explained),
        }
        audit.append(row)
        if not explained:
            unexplained.append(row)

    maximum = min(maximum_count, max(1, int(np.ceil(maximum_fraction * len(queries)))))
    if unexplained or len(mismatch) > maximum:
        raise RuntimeError(
            "official retrieval replay differs beyond the audited numerical-boundary "
            f"contract: mismatches={len(mismatch)} maximum={maximum} "
            f"unexplained={len(unexplained)} first={audit[:10]}"
        )
    return stable, audit


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
    cluster_mean = cluster_sum / cluster_count
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    equal_formula_samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(formulas), size=len(formulas))
        samples[draw] = cluster_sum[selected].sum() / cluster_count[selected].sum()
        equal_formula_samples[draw] = np.mean(cluster_mean[selected])
    return {
        "delta_recall1": float(np.mean(delta)),
        "delta_recall1_estimand": "query_weighted",
        "delta_mrr": float(np.mean(1.0 / current) - np.mean(1.0 / baseline)),
        "corrected_at_1": int(np.sum((baseline > 1) & (current == 1))),
        "introduced_at_1": int(np.sum((baseline == 1) & (current > 1))),
        "formula_cluster_bootstrap_delta_recall1_ci95": [
            float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)),
        ],
        "equal_formula_delta_recall1_sensitivity": float(np.mean(cluster_mean)),
        "equal_formula_cluster_bootstrap_delta_recall1_ci95": [
            float(np.quantile(equal_formula_samples, 0.025)),
            float(np.quantile(equal_formula_samples, 0.975)),
        ],
        "formula_cluster_size_range": [
            int(np.min(cluster_count)), int(np.max(cluster_count)),
        ],
    }
