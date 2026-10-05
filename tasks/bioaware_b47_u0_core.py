#!/usr/bin/env python
"""Pure aggregation primitives for the B47 truth-blind unary audit."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import pandas as pd


ALLOWED_ADDUCTS = ("[M+H]+", "[M+Na]+")


@lru_cache(maxsize=None)
def order_statistic_weights(population_size: int, sample_size: int) -> np.ndarray:
    """Weights for E[max] of a uniform subset sampled without replacement.

    If the sorted population is x[0] <= ... <= x[n-1], x[j] is the subset
    maximum exactly when it is selected together with k-1 values among the j
    lower values.  Therefore P(max=x[j]) = C(j,k-1)/C(n,k).
    """
    n, k = int(population_size), int(sample_size)
    if n <= 0 or k <= 0 or k > n:
        raise ValueError(f"invalid order statistic sizes n={n} k={k}")
    denominator = math.comb(n, k)
    weights = np.zeros(n, dtype=np.float64)
    for j in range(k - 1, n):
        weights[j] = math.comb(j, k - 1) / denominator
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("order-statistic weights do not sum to one")
    weights.setflags(write=False)
    return weights


def expected_max_without_replacement(values: np.ndarray, sample_size: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("expected-max values must be a finite non-empty vector")
    if sample_size > len(values):
        return float("nan")
    ordered = np.sort(values, kind="stable")
    return float(ordered @ order_statistic_weights(len(ordered), int(sample_size)))


def aggregate_candidate_scores(
    references: pd.DataFrame,
    subset_sizes: tuple[int, ...] = (2, 4, 8),
) -> pd.DataFrame:
    """Collapse reference-spectrum scores without reading identity truth.

    ``mean_score`` is exactly the expected score after exposing one uniformly
    sampled reference spectrum.  ``expected_max_k`` is the exact expectation
    after exposing k references, conditional on a candidate having at least k.
    These are counterfactual *score* summaries, not performance estimates.
    """
    required = {
        "query_id", "candidate_id", "candidate_formula", "reference_row",
        "reference_adduct", "mass_error_ppm", "spectral_score",
    }
    missing = required - set(references.columns)
    if missing:
        raise ValueError(f"reference scores miss {sorted(missing)}")
    if references.empty:
        raise ValueError("reference score table is empty")
    if not np.isfinite(references["spectral_score"].to_numpy(float)).all():
        raise ValueError("reference scores contain non-finite values")
    if not references["reference_adduct"].isin(ALLOWED_ADDUCTS).all():
        raise ValueError("unexpected reference adduct")

    keys = ["query_id", "candidate_id"]
    ordered = references.sort_values(keys + ["reference_row"], kind="stable").reset_index(drop=True)
    key_change = np.ones(len(ordered), dtype=bool)
    key_change[1:] = (
        ordered["query_id"].to_numpy(object)[1:] != ordered["query_id"].to_numpy(object)[:-1]
    ) | (
        ordered["candidate_id"].to_numpy(object)[1:] != ordered["candidate_id"].to_numpy(object)[:-1]
    )
    starts = np.flatnonzero(key_change)
    ends = np.r_[starts[1:], len(ordered)]
    scores = ordered["spectral_score"].to_numpy(np.float64)
    rows: list[dict[str, object]] = []
    for start, end in zip(starts, ends, strict=True):
        group = ordered.iloc[start:end]
        group_scores = scores[start:end]
        formulas = group["candidate_formula"].astype(str).unique()
        adducts = group["reference_adduct"].astype(str).unique()
        if len(formulas) != 1:
            raise ValueError("candidate has inconsistent molecular formulas")
        if len(adducts) != 1:
            # With a 10-ppm observed-ion window, one neutral candidate cannot
            # simultaneously instantiate [M+H]+ and [M+Na]+.  A violation is
            # therefore an upstream namespace or mass-semantics error.
            raise ValueError("candidate spans multiple adduct branches")
        best_local = int(np.argmax(group_scores))
        row: dict[str, object] = {
            "query_id": str(group.iloc[0]["query_id"]),
            "candidate_id": str(group.iloc[0]["candidate_id"]),
            "candidate_formula": str(formulas[0]),
            "reference_adduct": str(adducts[0]),
            "reference_count": int(end - start),
            "max_score": float(group_scores[best_local]),
            "mean_score": float(group_scores.mean()),
            "score_std": float(group_scores.std(ddof=0)),
            "max_minus_mean": float(group_scores.max() - group_scores.mean()),
            "best_reference_row": int(group.iloc[best_local]["reference_row"]),
            "best_reference_mass_error_ppm": float(group.iloc[best_local]["mass_error_ppm"]),
        }
        for k in subset_sizes:
            row[f"expected_max_{int(k)}"] = expected_max_without_replacement(group_scores, int(k))
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.duplicated(keys).any():
        raise RuntimeError("candidate aggregation emitted duplicate keys")
    if (output["max_minus_mean"] < -1e-12).any():
        raise RuntimeError("maximum score is below mean score")
    return output


def top_summary(
    candidates: pd.DataFrame,
    score_column: str,
    prefix: str,
) -> pd.DataFrame:
    required = {
        "query_id", "candidate_id", "reference_adduct", "reference_count", score_column,
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table misses {sorted(missing)}")
    rows: list[dict[str, object]] = []
    finite = candidates[np.isfinite(candidates[score_column].to_numpy(float))]
    for query_id, group in finite.groupby("query_id", sort=False):
        ordered = group.sort_values(
            [score_column, "candidate_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        if len(ordered) < 2:
            continue
        top_score = float(ordered.loc[0, score_column])
        tied = np.isclose(ordered[score_column].to_numpy(float), top_score, atol=1e-12, rtol=0)
        unique = int(tied.sum()) == 1
        rows.append({
            "query_id": str(query_id),
            f"{prefix}_top_candidate_id": str(ordered.loc[0, "candidate_id"]) if unique else "",
            f"{prefix}_top_adduct": str(ordered.loc[0, "reference_adduct"]) if unique else "",
            f"{prefix}_top_score": top_score,
            f"{prefix}_second_score": float(ordered.loc[1, score_column]),
            f"{prefix}_margin": top_score - float(ordered.loc[1, score_column]),
            f"{prefix}_top_tie_count": int(tied.sum()),
            f"{prefix}_unique_top1": bool(unique),
            f"{prefix}_top_reference_count": int(ordered.loc[0, "reference_count"]),
            f"{prefix}_candidate_count": int(len(ordered)),
        })
    return pd.DataFrame(rows)


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "minimum": None, "median": None, "p90": None, "maximum": None}
    return {
        "n": int(len(values)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(np.max(values)),
    }


def candidate_exposure_summary(candidates: pd.DataFrame) -> dict[str, object]:
    """Summarise catalogue exposure without outcome or network information.

    The same candidate identity can be offered to many queries and candidates
    can have very different numbers of reference spectra.  Both are potential
    catalogue shortcuts, so U0 records them before any labelled comparison.
    """
    required = {
        "query_id", "candidate_id", "candidate_formula", "reference_adduct",
        "reference_count",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate exposure table misses {sorted(missing)}")
    if candidates.empty or candidates.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("candidate exposure requires unique non-empty query/candidate rows")
    if (candidates["reference_count"].to_numpy(np.int64) < 1).any():
        raise ValueError("candidate exposure contains zero reference spectra")

    by_query = candidates.groupby("query_id", sort=False)
    query_candidates = by_query["candidate_id"].nunique().astype(np.int64)
    query_formulas = by_query["candidate_formula"].nunique().astype(np.int64)
    query_references = by_query["reference_count"].sum().astype(np.int64)
    query_max_count = by_query["reference_count"].max().astype(np.float64)
    query_median_count = by_query["reference_count"].median().astype(np.float64)
    imbalance = query_max_count / query_median_count

    by_identity = candidates.groupby("candidate_id", sort=False)
    identity_queries = by_identity["query_id"].nunique().astype(np.int64)
    identity_studies = (
        by_identity["study"].nunique().astype(np.int64)
        if "study" in candidates.columns else pd.Series(dtype=np.int64)
    )
    descending = np.sort(identity_queries.to_numpy(np.int64))[::-1]
    top_n = max(1, int(math.ceil(len(descending) * 0.01)))
    top_share = float(descending[:top_n].sum() / descending.sum())

    adduct = {
        str(name): {
            "candidate_query_pairs": int(len(group)),
            "candidate_identities": int(group["candidate_id"].nunique()),
            "reference_count_per_candidate": quantiles(
                group["reference_count"].to_numpy(np.float64)
            ),
        }
        for name, group in candidates.groupby("reference_adduct", sort=True)
    }
    return {
        "candidate_identities_per_query": quantiles(query_candidates.to_numpy(float)),
        "candidate_formulas_per_query": quantiles(query_formulas.to_numpy(float)),
        "candidate_reference_spectra_per_query": quantiles(query_references.to_numpy(float)),
        "maximum_to_median_reference_count_within_query": quantiles(
            imbalance.to_numpy(float)
        ),
        "fraction_queries_with_reference_count_ratio_ge_2": float((imbalance >= 2).mean()),
        "fraction_queries_with_reference_count_ratio_ge_4": float((imbalance >= 4).mean()),
        "queries_per_candidate_identity": quantiles(identity_queries.to_numpy(float)),
        "studies_per_candidate_identity": (
            quantiles(identity_studies.to_numpy(float)) if len(identity_studies) else None
        ),
        "top_1pct_candidate_identities_share_of_candidate_query_pairs": top_share,
        "by_adduct": adduct,
    }


def spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    """Tie-aware Spearman correlation without an optional SciPy dependency."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >=2")
    left_rank = left.rank(method="average").to_numpy(np.float64)
    right_rank = right.rank(method="average").to_numpy(np.float64)
    if not np.isfinite(left_rank).all() or not np.isfinite(right_rank).all():
        raise ValueError("Spearman inputs contain non-finite values")
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])
