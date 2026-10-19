from __future__ import annotations

import pandas as pd

from evaluate_contrast_atlas_frozen_graph_exact_identity import (
    evaluate,
    orient_edges,
    strict_truth_groups,
)


def test_orientation_preserves_both_real_query_directions() -> None:
    edges = pd.DataFrame([
        {"left_study_id": "a", "right_study_id": "b", "evidence_world": "known", "direction": "left_to_right", "left_entity_id": "a1", "right_entity_id": "b1", "chemical_similarity": 0.9},
        {"left_study_id": "a", "right_study_id": "b", "evidence_world": "known", "direction": "right_to_left", "left_entity_id": "a2", "right_entity_id": "b2", "chemical_similarity": 0.8},
    ])
    observed = orient_edges(edges)
    assert observed[["query_study_id", "query_entity_id", "candidate_study_id", "candidate_entity_id"]].values.tolist() == [
        ["a", "a1", "b", "b1"], ["b", "b2", "a", "a2"]
    ]


def test_exact_truth_allows_multiple_same_identity_target_entities() -> None:
    truth = pd.DataFrame([
        {"planar_inchikey": "AAAAAAAAAAAAAA", "left_study_id": "a", "left_entity_id": "a1", "right_study_id": "b", "right_entity_id": "b1", "truth_class": "exact_planar_inchikey_identity", "truth_scope": "posttruth_spectral_entity_retrieval_calibration_only"},
        {"planar_inchikey": "AAAAAAAAAAAAAA", "left_study_id": "a", "left_entity_id": "a1", "right_study_id": "b", "right_entity_id": "b2", "truth_class": "exact_planar_inchikey_identity", "truth_scope": "posttruth_spectral_entity_retrieval_calibration_only"},
    ])
    groups = strict_truth_groups(truth, {"a", "b"})
    edges = pd.DataFrame([
        {"evidence_world": "known", "query_study_id": "a", "query_entity_id": "a1", "candidate_study_id": "b", "candidate_entity_id": "b2", "chemical_similarity": 0.9},
    ])
    summary, detail = evaluate(edges, groups)
    assert int(summary.loc[0, "top1_exact_identity_correct"]) == 1
    assert bool(detail.loc[0, "top1_exact_identity_correct"])
