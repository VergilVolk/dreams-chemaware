#!/usr/bin/env python
"""Shared, task-faithful metrics for the BioAware unified benchmark.

The benchmark has one relevant candidate per query.  Candidate ranking uses
the project's conservative tie rule: every negative tied with the truth ranks
ahead of the truth.  Query AUROC remains the standard Mann-Whitney statistic
and therefore gives score ties half credit.  These two quantities answer
different questions and are deliberately kept separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


TOP_K = (1, 2, 5, 10, 20)
REQUIRED_CANDIDATE_COLUMNS = {
    "query_id",
    "candidate_id",
    "is_truth",
    "formula_cluster",
    "source",
    "polarity",
}
REQUIRED_PREDICTION_COLUMNS = {"query_id", "candidate_id", "score"}


@dataclass(frozen=True)
class ClusterInterval:
    mean: float
    ci_low: float
    ci_high: float
    clusters: int
    resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "clusters": self.clusters,
            "resamples": self.resamples,
        }


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        if not values.isin([0, 1]).all():
            raise RuntimeError(f"{name} must contain only 0/1")
        return values.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise RuntimeError(f"{name} contains non-boolean values")
    return mapped.astype(bool)


def validate_candidate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize a query/candidate denominator."""
    missing = REQUIRED_CANDIDATE_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"candidate manifest missing columns: {sorted(missing)}")
    output = frame.copy()
    for column in ("query_id", "candidate_id", "formula_cluster", "source", "polarity"):
        output[column] = output[column].astype(str)
        if output[column].str.len().eq(0).any():
            raise RuntimeError(f"candidate manifest has empty {column}")
    output["is_truth"] = _as_bool(output["is_truth"], "is_truth")
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("candidate manifest has duplicate query/candidate rows")
    grouped = output.groupby("query_id", sort=False)
    truth_counts = grouped["is_truth"].sum()
    if not truth_counts.eq(1).all():
        bad = truth_counts.loc[~truth_counts.eq(1)].head().to_dict()
        raise RuntimeError(f"every query must have exactly one truth: {bad}")
    sizes = grouped.size()
    if not sizes.ge(2).all():
        bad = sizes.loc[~sizes.ge(2)].head().to_dict()
        raise RuntimeError(f"every query must have at least two candidates: {bad}")
    for column in ("formula_cluster", "source", "polarity"):
        counts = grouped[column].nunique(dropna=False)
        if not counts.eq(1).all():
            raise RuntimeError(f"{column} must be query-constant")
    if "near_query" in output.columns:
        output["near_query"] = _as_bool(output["near_query"], "near_query")
        if not grouped["near_query"].nunique(dropna=False).eq(1).all():
            raise RuntimeError("near_query must be query-constant")
    return output.sort_values(["query_id", "candidate_id"], kind="stable").reset_index(drop=True)


def align_predictions(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Require an exact candidate-key match and return scores in manifest order."""
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise RuntimeError(f"prediction file missing columns: {sorted(missing)}")
    scores = predictions.loc[:, ["query_id", "candidate_id", "score"]].copy()
    scores["query_id"] = scores["query_id"].astype(str)
    scores["candidate_id"] = scores["candidate_id"].astype(str)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise").astype(float)
    if not np.isfinite(scores["score"].to_numpy()).all():
        raise RuntimeError("prediction file contains non-finite scores")
    if scores.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("prediction file has duplicate query/candidate rows")
    expected = candidates.loc[:, ["query_id", "candidate_id"]]
    merged = expected.merge(
        scores,
        on=["query_id", "candidate_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    counts = merged["_merge"].value_counts().to_dict()
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise RuntimeError(f"prediction candidate keys differ from manifest: {counts}")
    merged = merged.drop(columns="_merge")
    if len(merged) != len(candidates):
        raise RuntimeError("prediction alignment changed candidate row count")
    return merged


def _query_auroc(truth_score: float, negative_scores: np.ndarray) -> float:
    credit = (truth_score > negative_scores).astype(float)
    credit += 0.5 * (truth_score == negative_scores)
    return float(credit.mean())


def evaluate_method(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    method_id: str,
) -> pd.DataFrame:
    """Produce one strict, auditable result row per query."""
    aligned = align_predictions(candidates, predictions)
    work = candidates.copy()
    work["score"] = aligned["score"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for query_id, group in work.groupby("query_id", sort=False):
        truth = group.loc[group["is_truth"]]
        negatives = group.loc[~group["is_truth"]]
        truth_score = float(truth["score"].iloc[0])
        negative_scores = negatives["score"].to_numpy(float)
        rank = 1 + int(np.sum(negative_scores >= truth_score))
        row: dict[str, object] = {
            "method_id": str(method_id),
            "query_id": str(query_id),
            "formula_cluster": str(group["formula_cluster"].iloc[0]),
            "source": str(group["source"].iloc[0]),
            "polarity": str(group["polarity"].iloc[0]),
            "candidate_count": int(len(group)),
            "truth_score": truth_score,
            "strict_rank": rank,
            "top1": rank == 1,
            "reciprocal_rank": 1.0 / rank,
            "query_auroc": _query_auroc(truth_score, negative_scores),
            "truth_margin_vs_best_negative": truth_score - float(negative_scores.max()),
        }
        if "near_query" in group.columns:
            row["near_query"] = bool(group["near_query"].iloc[0])
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != candidates["query_id"].nunique() or output["query_id"].duplicated().any():
        raise RuntimeError("per-query evaluation coverage changed")
    return output


def summarize_method(per_query: pd.DataFrame) -> dict[str, float | int]:
    ranks = per_query["strict_rank"].to_numpy(int)
    output: dict[str, float | int] = {
        "n_queries": int(len(per_query)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_2": float(np.mean(ranks <= 2)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "recall_at_20": float(np.mean(ranks <= 20)),
        "mrr": float(per_query["reciprocal_rank"].mean()),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "macro_query_auroc": float(per_query["query_auroc"].mean()),
        "mean_truth_margin_vs_best_negative": float(
            per_query["truth_margin_vs_best_negative"].mean()
        ),
    }
    for k in (5, 10, 20):
        discount = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
        output[f"ndcg_at_{k}"] = float(discount.mean())
    return output


def exact_mcnemar_p(corrected: int, introduced: int) -> float:
    """Two-sided exact McNemar p-value using the discordant binomial.

    Computed in log space so that large discordant totals cannot overflow;
    for small totals the value is identical to the direct binomial sum.
    """
    corrected = int(corrected)
    introduced = int(introduced)
    if corrected < 0 or introduced < 0:
        raise ValueError("McNemar counts must be nonnegative")
    total = corrected + introduced
    if total == 0:
        return 1.0
    smaller = min(corrected, introduced)
    log_half_total = total * math.log(2.0)
    log_masses = [
        math.lgamma(total + 1)
        - math.lgamma(k + 1)
        - math.lgamma(total - k + 1)
        - log_half_total
        for k in range(smaller + 1)
    ]
    peak = max(log_masses)
    tail = math.exp(peak) * sum(math.exp(value - peak) for value in log_masses)
    # tail may underflow to 0.0 for astronomically small p; that is the
    # correct double representation and never raises.
    return float(min(1.0, 2.0 * tail))


def cluster_bootstrap_mean(
    values: Iterable[float],
    clusters: Iterable[str],
    *,
    resamples: int,
    seed: int,
) -> ClusterInterval:
    """Paired cluster bootstrap of a query-level mean.

    Clusters are sampled with replacement and all query rows belonging to a
    sampled cluster are retained.  This keeps unequal cluster sizes in the
    estimand instead of first macro-averaging clusters.
    """
    values_array = np.asarray(list(values), dtype=float)
    cluster_array = np.asarray(list(clusters)).astype(str)
    if len(values_array) != len(cluster_array) or len(values_array) == 0:
        raise ValueError("bootstrap inputs must be non-empty and aligned")
    if not np.isfinite(values_array).all():
        raise ValueError("bootstrap values must be finite")
    labels, inverse = np.unique(cluster_array, return_inverse=True)
    sums = np.bincount(inverse, weights=values_array).astype(float)
    counts = np.bincount(inverse).astype(float)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=float)
    n_clusters = len(labels)
    for index in range(int(resamples)):
        sample = rng.integers(0, n_clusters, size=n_clusters)
        draws[index] = float(sums[sample].sum() / counts[sample].sum())
    return ClusterInterval(
        mean=float(values_array.mean()),
        ci_low=float(np.quantile(draws, 0.025)),
        ci_high=float(np.quantile(draws, 0.975)),
        clusters=int(n_clusters),
        resamples=int(resamples),
    )


def compare_methods(
    baseline: pd.DataFrame,
    contender: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Paired comparison on the exact same query denominator."""
    base = baseline.copy()
    test = contender.copy()
    if base["query_id"].duplicated().any() or test["query_id"].duplicated().any():
        raise RuntimeError("method comparison requires one row per query")
    joined = base.merge(
        test,
        on=["query_id", "formula_cluster", "source", "polarity"],
        how="outer",
        validate="one_to_one",
        suffixes=("_baseline", "_contender"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("methods do not share the exact query denominator")
    joined = joined.drop(columns="_merge")
    before = joined["top1_baseline"].astype(bool)
    after = joined["top1_contender"].astype(bool)
    corrected = int((~before & after).sum())
    introduced = int((before & ~after).sum())
    joined["top1_delta"] = after.astype(float) - before.astype(float)
    joined["mrr_delta"] = (
        joined["reciprocal_rank_contender"] - joined["reciprocal_rank_baseline"]
    )
    joined["query_auroc_delta"] = (
        joined["query_auroc_contender"] - joined["query_auroc_baseline"]
    )
    comparison: dict[str, object] = {
        "n_queries": int(len(joined)),
        "corrected": corrected,
        "introduced": introduced,
        "net_corrections": corrected - introduced,
        "risk_net_lambda_2": corrected - 2 * introduced,
        "mcnemar_exact_p": exact_mcnemar_p(corrected, introduced),
        "delta_recall_at_1": float(joined["top1_delta"].mean()),
        "delta_mrr": float(joined["mrr_delta"].mean()),
        "delta_macro_query_auroc": float(joined["query_auroc_delta"].mean()),
        "formula_cluster_top1_ci": cluster_bootstrap_mean(
            joined["top1_delta"],
            joined["formula_cluster"],
            resamples=resamples,
            seed=seed,
        ).as_dict(),
        "source_cluster_top1_ci": cluster_bootstrap_mean(
            joined["top1_delta"],
            joined["source"],
            resamples=resamples,
            seed=seed + 1,
        ).as_dict(),
    }
    return comparison, joined

