from __future__ import annotations

import pandas as pd

from summarize_contrast_atlas_complete_identity_wse_global import (
    IDENTITY_TRACK,
    relation_family_summary,
)


def test_relation_summary_does_not_count_two_directions_as_independent() -> None:
    rows = []
    for world, ranks, pvalues in (
        ("all", (1, 4), (0.01, 0.02)),
        ("known", (2, 8), (0.04, 0.20)),
    ):
        for direction, left, right, rank, pvalue in (
            ("registered", ("A", "x"), ("B", "y"), ranks[0], pvalues[0]),
            ("reverse_registered", ("B", "y"), ("A", "x"), ranks[1], pvalues[1]),
        ):
            rows.append({
                "relation_direction": direction,
                "relation_class": "prespecified_exact_intervention_family",
                "left_study_id": left[0],
                "left_contrast_id": left[1],
                "right_study_id": right[0],
                "right_contrast_id": right[1],
                "evidence_world": world,
                "evidence_track": IDENTITY_TRACK,
                "best_face_rank": rank,
                "best_face_query_maxT_p": pvalue,
            })
    result = relation_family_summary(pd.DataFrame(rows)).set_index("evidence_world")
    assert result.loc["all", "canonical_relations"] == 1
    assert result.loc["all", "directional_relations"] == 2
    assert result.loc["all", "canonical_any_direction_top1"] == 1
    assert result.loc["all", "canonical_both_directions_top1"] == 0
    assert result.loc["all", "canonical_both_directions_top5"] == 1
    assert result.loc["known", "canonical_any_direction_maxT_005"] == 1
    assert result.loc["known", "canonical_both_directions_maxT_005"] == 0
