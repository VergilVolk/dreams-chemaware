from __future__ import annotations

import numpy as np
import pandas as pd

from summarize_contrast_atlas_wse_edge_opportunity_controls import (
    IDENTITY_TRACK,
    distribution,
    subset_summary,
)


def test_subset_summary_keeps_directional_and_canonical_units_distinct() -> None:
    rows = []
    for direction, rank, pvalue in (("a_to_b", 1.0, 0.01), ("b_to_a", 7.0, 0.20)):
        rows.append({
            "relation_direction": direction,
            "left_study_id": "A",
            "left_contrast_id": "a",
            "right_study_id": "B",
            "right_contrast_id": "b",
            "relation_class": "family",
            "evidence_world": "unresolved",
            "evidence_track": IDENTITY_TRACK,
            "best_face_rank": rank,
            "best_face_query_maxT_p": pvalue,
            "similar_rank": rank,
            "opposite_rank": rank + 5,
            "similar_query_maxT_p": pvalue,
            "opposite_query_maxT_p": 1.0,
        })
    summary = subset_summary(pd.DataFrame(rows), {"family"}, "unresolved")
    assert summary["directional_relations"] == 2
    assert summary["directional_top1"] == 1
    assert summary["directional_top5"] == 1
    assert summary["canonical_relations"] == 1
    assert summary["canonical_any_top1"] == 1
    assert summary["canonical_both_top1"] == 0
    assert summary["directional_best_face_bonferroni_005"] == 1
    assert summary["canonical_any_best_face_bonferroni_005"] == 1
    assert summary["canonical_both_best_face_bonferroni_005"] == 0


def test_distribution_reports_all_resamples() -> None:
    frame = pd.DataFrame({
        "analysis_group": ["x", "x", "x"],
        "metric": [1, 3, 5],
    })
    result = distribution(frame, "x", "metric")
    assert result["resamples"] == 3
    assert result["minimum"] == 1
    assert result["median"] == 3
    assert result["maximum"] == 5
    assert np.isclose(result["q25"], 2)
