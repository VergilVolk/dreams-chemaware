from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_contrast_atlas_query_outcomes import build_outcomes


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.DataFrame([
        {"query_id": "q1", "query_study_id": "Q1", "query_contrast_id": "c", "cluster_id": "c1", "source_holdout": "s1", "instrument_holdout": "i1"},
        {"query_id": "q2", "query_study_id": "Q2", "query_contrast_id": "c", "cluster_id": "c2", "source_holdout": "s2", "instrument_holdout": "i2"},
    ])
    truth = pd.DataFrame([
        {"query_id": "q1", "target_study_id": "T1", "target_contrast_id": "a", "expected_association": "similar_response"},
        {"query_id": "q1", "target_study_id": "T2", "target_contrast_id": "b", "expected_association": "irrelevant"},
        {"query_id": "q2", "target_study_id": "T1", "target_contrast_id": "a", "expected_association": "irrelevant"},
        {"query_id": "q2", "target_study_id": "T2", "target_contrast_id": "b", "expected_association": "irrelevant"},
    ])
    truth["truth_basis"] = "synthetic_smoke"
    truth["truth_data_role"] = "synthetic"
    truth["truth_frozen_before_scoring"] = True
    truth["selection_blind_to_atlas_score"] = True
    rows = []
    for query_id, query_study in (("q1", "Q1"), ("q2", "Q2")):
        for world in ("all", "known_only", "measurement_matched_known", "unresolved_shuffled_ms2"):
            for target, contrast, score in (("T1", "a", 0.9), ("T2", "b", 0.4)):
                accepted = world == "all" and query_id == "q1" and target == "T1"
                rows.append({
                    "world": world, "query_study_id": query_study, "query_contrast_id": "c",
                    "target_study_id": target, "target_contrast_id": contrast,
                    "calibration_status": "calibrated_within_target_pattern_permutation",
                    "calibrated_association": "similar_response" if accepted else "indeterminate_after_pattern_null",
                    "q_similar": 0.01 if accepted else 0.8, "q_opposite": 0.9,
                    "observed_score": score,
                })
    return pd.DataFrame(rows), registry, truth


def test_outcome_builder_scores_match_and_no_match() -> None:
    candidates, registry, truth = fixtures()
    outcomes, accepted = build_outcomes(candidates, registry, truth)
    all_world = outcomes.loc[outcomes["world"] == "all"].set_index("query_id")
    assert bool(all_world.loc["q1", "answered"])
    assert bool(all_world.loc["q1", "correct"])
    assert all_world.loc["q1", "rank"] == 1
    assert not bool(all_world.loc["q2", "answered"])
    assert bool(all_world.loc["q2", "correct"])
    assert len(accepted) == 1


def test_outcome_builder_rejects_incomplete_truth() -> None:
    candidates, registry, truth = fixtures()
    truth = truth.iloc[:-1]
    with pytest.raises(RuntimeError, match="truth is incomplete"):
        build_outcomes(candidates, registry, truth)


def test_outcome_builder_rejects_posthoc_truth() -> None:
    candidates, registry, truth = fixtures()
    truth.loc[0, "selection_blind_to_atlas_score"] = False
    with pytest.raises(RuntimeError, match="used Atlas scores"):
        build_outcomes(candidates, registry, truth)
