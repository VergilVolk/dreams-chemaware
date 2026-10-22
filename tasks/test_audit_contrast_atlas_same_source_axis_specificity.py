from __future__ import annotations

import pandas as pd

from audit_contrast_atlas_same_source_axis_specificity import audit_arm


def test_same_source_axis_strict_separation() -> None:
    scores = pd.DataFrame({
        "query_study_id": ["Q"] * 4,
        "query_contrast_id": ["q"] * 4,
        "candidate_study_id": ["T"] * 4,
        "candidate_contrast_id": ["p1", "p2", "c1", "c2"],
        "evidence_world": ["all"] * 4,
        "similar_response_score": [0.7, 0.6, 0.5, 0.2],
        "opposite_response_score": [0.1, 0.2, 0.4, 0.3],
        "unsigned_response_score": [0.8, 0.8, 0.9, 0.5],
        "composition_score": [0.4] * 4,
    })
    relations = pd.DataFrame({
        "left_study_id": ["Q", "Q"],
        "left_contrast_id": ["q", "q"],
        "right_study_id": ["T", "T"],
        "right_contrast_id": ["p1", "p2"],
        "eligible_primary_relevance": ["True", "True"],
    })
    result = audit_arm(scores, relations, "observed").iloc[0]
    assert result["registered_axes"] == 2
    assert result["nonregistered_axis_controls"] == 2
    assert result["similar_response_strict_separation"]
    assert not result["unsigned_response_strict_separation"]
    assert not result["composition_strict_separation"]
