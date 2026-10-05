#!/usr/bin/env python
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from bioaware_b47_u0_core import (
    aggregate_candidate_scores,
    candidate_exposure_summary,
    expected_max_without_replacement,
    spearman_without_scipy,
    top_summary,
)


def test_expected_max_matches_brute_force() -> None:
    values = np.asarray([-0.2, 0.1, 0.4, 0.9], dtype=float)
    for k in (1, 2, 3, 4):
        brute = np.mean([max(part) for part in combinations(values.tolist(), k)])
        observed = expected_max_without_replacement(values, k)
        assert np.isclose(observed, brute, atol=1e-12, rtol=0)
    curve = [expected_max_without_replacement(values, k) for k in (1, 2, 3, 4)]
    assert np.all(np.diff(curve) >= 0)
    assert np.isnan(expected_max_without_replacement(values, 5))


def synthetic_references() -> pd.DataFrame:
    rows = []
    for query, candidate, formula, adduct, scores in (
        ("q1", "A", "C1", "[M+H]+", [0.5, 0.9]),
        ("q1", "B", "C2", "[M+Na]+", [0.75]),
        ("q2", "C", "C3", "[M+H]+", [0.8]),
        ("q2", "D", "C4", "[M+H]+", [0.8]),
    ):
        for index, score in enumerate(scores):
            rows.append({
                "query_id": query, "candidate_id": candidate,
                "candidate_formula": formula, "reference_row": len(rows),
                "reference_adduct": adduct, "mass_error_ppm": float(index),
                "spectral_score": score,
            })
    return pd.DataFrame(rows)


def test_aggregation_exposes_extreme_value_lift() -> None:
    result = aggregate_candidate_scores(synthetic_references(), subset_sizes=(2,))
    a = result[result["candidate_id"].eq("A")].iloc[0]
    b = result[result["candidate_id"].eq("B")].iloc[0]
    assert a.reference_count == 2
    assert np.isclose(a.max_score, 0.9)
    assert np.isclose(a.mean_score, 0.7)
    assert np.isclose(a.max_minus_mean, 0.2)
    assert np.isclose(a.expected_max_2, 0.9)
    assert np.isnan(b.expected_max_2)


def test_top_summary_counts_ties_adversely() -> None:
    candidates = aggregate_candidate_scores(synthetic_references(), subset_sizes=(2,))
    maximum = top_summary(candidates, "max_score", "max")
    q1 = maximum[maximum.query_id.eq("q1")].iloc[0]
    q2 = maximum[maximum.query_id.eq("q2")].iloc[0]
    assert q1.max_top_candidate_id == "A"
    assert bool(q1.max_unique_top1)
    assert q2.max_top_candidate_id == ""
    assert not bool(q2.max_unique_top1)
    assert q2.max_top_tie_count == 2


def test_candidate_cannot_cross_adduct_branches() -> None:
    frame = synthetic_references()
    frame.loc[1, "reference_adduct"] = "[M+Na]+"
    try:
        aggregate_candidate_scores(frame)
    except ValueError as error:
        assert "multiple adduct" in str(error)
    else:
        raise AssertionError("mixed-adduct candidate was accepted")


def test_spearman_has_no_optional_scipy_dependency() -> None:
    assert np.isclose(
        spearman_without_scipy(pd.Series([3.0, 1.0, 2.0]), pd.Series([30.0, 10.0, 20.0])),
        1.0,
    )


def test_candidate_exposure_counts_catalogue_shortcuts() -> None:
    candidates = aggregate_candidate_scores(synthetic_references(), subset_sizes=(2,))
    candidates["study"] = ["s1", "s1", "s2", "s2"]
    summary = candidate_exposure_summary(candidates)
    assert summary["candidate_identities_per_query"]["minimum"] == 2.0
    assert summary["candidate_reference_spectra_per_query"]["maximum"] == 3.0
    assert np.isclose(
        summary["fraction_queries_with_reference_count_ratio_ge_2"], 0.0
    )
    assert set(summary["by_adduct"]) == {"[M+H]+", "[M+Na]+"}
    assert np.isclose(
        spearman_without_scipy(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0])),
        -1.0,
    )


if __name__ == "__main__":
    test_expected_max_matches_brute_force()
    test_aggregation_exposes_extreme_value_lift()
    test_top_summary_counts_ties_adversely()
    test_candidate_cannot_cross_adduct_branches()
    test_spearman_has_no_optional_scipy_dependency()
    test_candidate_exposure_counts_catalogue_shortcuts()
    print("[test_bioaware_b47_u0_reference_bias] PASS")
