from __future__ import annotations

import pandas as pd

from analyze_contrast_atlas_global_adjudication import (
    IDENTITY_LAYER,
    RELATION_LAYER,
    RELATION_TRACK,
    build_source_concentration,
    classify_queue,
    combine_two_faces,
    connection_key,
    safe_fraction,
)


def test_connection_key_is_undirected() -> None:
    forward = connection_key(
        pd.Series(["B"]), pd.Series(["b"]),
        pd.Series(["A"]), pd.Series(["a"]),
    )
    reverse = connection_key(
        pd.Series(["A"]), pd.Series(["a"]),
        pd.Series(["B"]), pd.Series(["b"]),
    )
    assert forward.iloc[0] == reverse.iloc[0] == "A::a||B::b"


def test_safe_fraction_has_explicit_zero_denominator_semantics() -> None:
    result = safe_fraction(pd.Series([2.0, 1.0]), pd.Series([4.0, 0.0]))
    assert result.tolist() == [0.5, 0.0]


def test_two_face_selection_applies_bonferroni_and_keeps_all_queries() -> None:
    rows = []
    for query_index in range(51):
        for world in ("all", "known", "unresolved"):
            for face, pvalue in (("similar", 0.02), ("opposite", 0.20)):
                rows.append({
                    "query_study_id": f"Q{query_index}",
                    "query_contrast_id": "contrast",
                    "evidence_world": world,
                    "response_face": face,
                    "evidence_track": RELATION_TRACK,
                    "mutual_edges": 5,
                    "enrichment_z": 2.0,
                    "query_maxT_p": pvalue,
                    "candidate_study_id": "TARGET",
                    "candidate_contrast_id": face,
                })
    result = combine_two_faces(pd.DataFrame(rows), RELATION_TRACK)
    assert len(result) == 153
    assert result["response_face"].eq("similar").all()
    assert result["two_face_bonferroni_p"].eq(0.04).all()
    assert result["combined_answer_class"].eq(
        "statistically_selected_hypothesis"
    ).all()


def test_adjudication_queue_never_labels_unregistered_as_false() -> None:
    queue = pd.DataFrame({
        "combined_answer_class": ["no_selected_match", "statistically_selected_hypothesis"],
        "same_parent_project": [False, False],
        "relation_evaluation_status": ["unregistered_unjudged", "unregistered_unjudged"],
        "biological_comparability_tier": [
            "cross_parent_cross_domain_exploratory",
            "cross_parent_cross_domain_exploratory",
        ],
        "cross_track_same_target_and_face": [False, False],
    })
    result = classify_queue(queue)
    assert result.tolist() == [
        "no_statistically_selected_match",
        "cross_domain_unregistered_hypothesis",
    ]
    assert not result.str.contains("false|incorrect|negative", case=False).any()


def test_source_concentration_keeps_identity_and_relation_layers_separate() -> None:
    queue = pd.DataFrame({
        "evidence_layer": [IDENTITY_LAYER, IDENTITY_LAYER, RELATION_LAYER],
        "combined_answer_class": [
            "statistically_selected_hypothesis",
            "statistically_selected_hypothesis",
            "statistically_selected_hypothesis",
        ],
        "candidate_project_dependence_group": ["A", "A", "B"],
        "query_study_id": ["q1", "q2", "q3"],
        "same_parent_project": [True, False, False],
    })
    table, summary = build_source_concentration(queue)
    assert len(table) == 2
    identity = next(row for row in summary if row["evidence_layer"] == IDENTITY_LAYER)
    relation = next(row for row in summary if row["evidence_layer"] == RELATION_LAYER)
    assert identity["selected_directions"] == 2
    assert identity["parent_project_hhi"] == 1.0
    assert relation["selected_directions"] == 1

