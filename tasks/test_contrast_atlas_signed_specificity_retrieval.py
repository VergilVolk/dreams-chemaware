"""Focused contracts for the signed Contrast Atlas response operator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = Path(__file__).resolve().parent
if str(TASKS) not in sys.path:
    sys.path.insert(0, str(TASKS))

from run_contrast_atlas_signed_specificity_retrieval import (  # noqa: E402
    center_candidate_axis,
    endpoint_donors,
    residualise_entity_profiles,
    signed_cosine_matrix,
)
from summarize_contrast_atlas_signed_specificity_result import (  # noqa: E402
    source_decisions,
)


def test_signed_cosine_preserves_direction_and_is_bounded() -> None:
    left = np.asarray([[1.0], [-1.0]])
    same = np.asarray([[2.0], [-2.0]])
    opposite = -same
    similarity = np.ones(2)

    same_score, same_support = signed_cosine_matrix(left, same, similarity)
    opposite_score, opposite_support = signed_cosine_matrix(
        left, opposite, similarity
    )

    assert np.isclose(same_score[0, 0], 1.0)
    assert np.isclose(opposite_score[0, 0], -1.0)
    assert same_support[0, 0] == 2
    assert opposite_support[0, 0] == 2
    assert np.nanmax(np.abs(np.concatenate([same_score, opposite_score]))) <= 1.0


def test_residualisation_ignores_missing_cells_and_preserves_missingness() -> None:
    matrix = np.asarray([[1.0, 3.0, 99.0], [4.0, 8.0, 12.0]])
    observed = np.asarray([[True, True, False], [True, False, True]])

    residual = residualise_entity_profiles(matrix, observed)

    np.testing.assert_allclose(residual[0, :2], [-1.0, 1.0])
    np.testing.assert_allclose(residual[1, [0, 2]], [-4.0, 4.0])
    assert np.isnan(residual[0, 2])
    assert np.isnan(residual[1, 1])


def test_endpoint_null_is_deterministic_and_respects_strata() -> None:
    rows = np.arange(8, dtype=np.int64)
    status = np.asarray(["known"] * 4 + ["unresolved"] * 4)
    precursor_mz = np.asarray([110.0, 120.0, 210.0, 220.0] * 2)

    first = endpoint_donors(rows, status, precursor_mz, 12, 20261031, "block")
    second = endpoint_donors(rows, status, precursor_mz, 12, 20261031, "block")

    np.testing.assert_array_equal(first, second)
    original_strata = [
        (status[index], int(precursor_mz[index] // 100)) for index in rows
    ]
    for replicate in first:
        permuted_strata = [
            (status[index], int(precursor_mz[index] // 100))
            for index in replicate
        ]
        assert permuted_strata == original_strata


def test_centering_single_candidate_abstains() -> None:
    matrix = np.asarray([[0.4], [-0.2]])
    assert np.isnan(center_candidate_axis(matrix, axis=1)).all()


def test_registered_target_removal_reapplies_minimum_support() -> None:
    method = "signed_cosine"
    scores = pd.DataFrame(
        [
            {
                "query_study_id": "Q",
                "query_contrast_id": "q",
                "candidate_study_id": "T",
                "candidate_contrast_id": "registered",
                "evidence_world": "all",
                "response_method": method,
                "score_available": True,
                "absolute_score": 0.9,
                "query_max_absT_p": 0.01,
                "response_overlap_edges": 3,
            },
            {
                "query_study_id": "Q",
                "query_contrast_id": "q",
                "candidate_study_id": "D",
                "candidate_contrast_id": "unsupported",
                "evidence_world": "all",
                "response_method": method,
                "score_available": True,
                "absolute_score": 0.8,
                "query_max_absT_p": 0.01,
                "response_overlap_edges": 1,
            },
            {
                "query_study_id": "Q",
                "query_contrast_id": "q",
                "candidate_study_id": "D",
                "candidate_contrast_id": "supported",
                "evidence_world": "all",
                "response_method": method,
                "score_available": True,
                "absolute_score": 0.7,
                "query_max_absT_p": 0.20,
                "response_overlap_edges": 3,
            },
        ]
    )
    top = pd.DataFrame(
        [
            {
                "query_study_id": "Q",
                "query_contrast_id": "q",
                "candidate_study_id": "T",
                "candidate_contrast_id": "registered",
                "evidence_world": "all",
                "response_method": method,
                "result_class": "similar_response",
            }
        ]
    )
    ranks = pd.DataFrame(
        [
            {
                "query_study_id": "Q",
                "query_contrast_id": "q",
                "candidate_study_id": "T",
                "candidate_contrast_id": "registered",
                "eligible_primary_relevance": True,
            }
        ]
    )

    decision = source_decisions(scores, top, ranks).iloc[0]

    assert decision["registered_target_removal_queries"] == 1
    assert decision["post_registered_target_removal_calls"] == 0
