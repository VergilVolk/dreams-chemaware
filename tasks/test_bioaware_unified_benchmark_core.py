#!/usr/bin/env python
"""Dependency-light unit checks for the BioAware unified benchmark core."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from bioaware_unified_benchmark_core import (  # noqa: E402
    align_predictions,
    compare_methods,
    evaluate_method,
    exact_mcnemar_p,
    summarize_method,
    validate_candidate_manifest,
)


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.DataFrame(
        [
            ("q1", "a", True, "f1", "s1", "negative", True),
            ("q1", "b", False, "f1", "s1", "negative", True),
            ("q1", "c", False, "f1", "s1", "negative", True),
            ("q2", "d", True, "f2", "s1", "positive", False),
            ("q2", "e", False, "f2", "s1", "positive", False),
            ("q3", "f", True, "f3", "s2", "negative", True),
            ("q3", "g", False, "f3", "s2", "negative", True),
        ],
        columns=[
            "query_id", "candidate_id", "is_truth", "formula_cluster",
            "source", "polarity", "near_query",
        ],
    )
    baseline = pd.DataFrame(
        [
            ("q1", "a", 0.8), ("q1", "b", 0.8), ("q1", "c", 0.2),
            ("q2", "d", 0.9), ("q2", "e", 0.1),
            ("q3", "f", 0.4), ("q3", "g", 0.5),
        ],
        columns=["query_id", "candidate_id", "score"],
    )
    contender = pd.DataFrame(
        [
            ("q1", "a", 0.81), ("q1", "b", 0.8), ("q1", "c", 0.2),
            ("q2", "d", 0.8), ("q2", "e", 0.9),
            ("q3", "f", 0.6), ("q3", "g", 0.5),
        ],
        columns=["query_id", "candidate_id", "score"],
    )
    return candidates, baseline, contender


def main() -> None:
    candidates, baseline_scores, contender_scores = fixture()
    candidates = validate_candidate_manifest(candidates)
    baseline = evaluate_method(candidates, baseline_scores, "official_dreams")
    contender = evaluate_method(candidates, contender_scores, "bioaware")

    # q1 proves that ties count against the truth for strict ranks, while its
    # standard query AUROC gives the tied pair half credit.
    q1 = baseline.set_index("query_id").loc["q1"]
    assert int(q1["strict_rank"]) == 2
    assert np.isclose(float(q1["query_auroc"]), 0.75)

    base_summary = summarize_method(baseline)
    test_summary = summarize_method(contender)
    assert np.isclose(base_summary["recall_at_1"], 1 / 3)
    assert np.isclose(test_summary["recall_at_1"], 2 / 3)
    assert test_summary["ndcg_at_5"] > base_summary["ndcg_at_5"]

    comparison, joined = compare_methods(
        baseline, contender, resamples=200, seed=7
    )
    assert len(joined) == 3
    assert comparison["corrected"] == 2
    assert comparison["introduced"] == 1
    assert comparison["net_corrections"] == 1
    assert comparison["risk_net_lambda_2"] == 0
    assert np.isclose(comparison["delta_recall_at_1"], 1 / 3)
    assert 0 <= float(comparison["mcnemar_exact_p"]) <= 1
    assert exact_mcnemar_p(0, 0) == 1.0

    # Regression: large discordant totals must not overflow and must stay
    # consistent with the direct binomial sum on small totals.
    assert np.isclose(exact_mcnemar_p(1, 1), 1.0)
    assert np.isclose(exact_mcnemar_p(2, 0), 0.5)
    assert np.isclose(exact_mcnemar_p(3, 0), 0.25)
    assert np.isclose(exact_mcnemar_p(4, 0), 0.125)
    moderate = exact_mcnemar_p(50, 80)
    assert 1e-6 < moderate < 0.05
    # 3000 vs 10000 is below the smallest positive double; it must report
    # 0.0 instead of raising OverflowError as the direct sum did.
    assert exact_mcnemar_p(3000, 10000) == 0.0
    balanced_large = exact_mcnemar_p(6000, 6000)
    assert 0.9 < balanced_large <= 1.0

    missing = contender_scores.iloc[:-1].copy()
    try:
        align_predictions(candidates, missing)
    except RuntimeError as error:
        assert "candidate keys differ" in str(error)
    else:
        raise AssertionError("missing prediction row did not fail closed")

    duplicated = candidates.copy()
    duplicated = pd.concat([duplicated, duplicated.iloc[[0]]], ignore_index=True)
    try:
        validate_candidate_manifest(duplicated)
    except RuntimeError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate candidate row did not fail closed")

    print("[test_bioaware_unified_benchmark_core] PASS", flush=True)


if __name__ == "__main__":
    main()

