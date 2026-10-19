from __future__ import annotations

import pandas as pd

from evaluate_contrast_atlas_selective_curve import selective_curve
from test_build_contrast_atlas_query_outcomes import fixtures


def test_selective_curve_reports_monotone_coverage_and_risk() -> None:
    candidates, registry, truth = fixtures()
    extra = candidates.copy()
    # Make q1/T2 a wrong but lower-confidence answer that enters only at 0.2.
    mask = (
        (extra["query_study_id"] == "Q1")
        & (extra["target_study_id"] == "T2")
        & (extra["world"] == "all")
    )
    extra.loc[mask, "calibrated_association"] = "similar_response"
    extra.loc[mask, "q_similar"] = 0.15
    curve = selective_curve(extra, registry, truth, thresholds=(0.01, 0.10, 0.20))
    all_world = curve.loc[curve["world"] == "all"].sort_values("fdr_threshold")
    assert all_world["coverage"].is_monotonic_increasing
    assert set(all_world["queries"]) == {2}
    assert all_world.iloc[-1]["answered_queries"] >= all_world.iloc[0]["answered_queries"]
