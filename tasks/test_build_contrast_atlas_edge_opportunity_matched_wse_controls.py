from __future__ import annotations

import pandas as pd

from build_contrast_atlas_edge_opportunity_matched_wse_controls import (
    KEY,
    MATCH_COVARIATES,
    STRATA,
    match_edges,
    mutual_edge_table,
)


def metric_rows() -> pd.DataFrame:
    rows = []
    for query_study, query_entity, candidate_study, candidate_entity, score in [
        ("A", "A::1", "B", "B::1", 0.8),
        ("B", "B::1", "A", "A::1", 0.7),
        ("A", "A::2", "B", "B::2", 0.9),
        ("B", "B::2", "A", "A::2", 0.85),
    ]:
        rows.append({
            "query_study_id": query_study,
            "query_entity_id": query_entity,
            "candidate_study_id": candidate_study,
            "candidate_entity_id": candidate_entity,
            "query_precursor_mz": 100.0 if query_study == "A" else 100.0002,
            "candidate_precursor_mz": 100.0002 if candidate_study == "B" else 100.0,
            "symmetric_ppm": 2.0,
            "physicochemical_compatibility_grade": "grade",
            "wse_score": score,
            "matched_fragments": 5,
            "world_candidate_count": 2,
            "top1_top2_margin": 0.2,
            "singleton": 0,
        })
    return pd.DataFrame(rows)


def test_mutual_edge_table_requires_both_directions() -> None:
    edges = pd.DataFrame([
        {
            "left_study_id": "A", "right_study_id": "B",
            "evidence_world": "known", "direction": "left_to_right",
            "left_entity_id": "A::1", "right_entity_id": "B::1",
            "chemical_similarity": 0.8,
        },
        {
            "left_study_id": "A", "right_study_id": "B",
            "evidence_world": "known", "direction": "right_to_left",
            "left_entity_id": "A::1", "right_entity_id": "B::1",
            "chemical_similarity": 0.7,
        },
        {
            "left_study_id": "A", "right_study_id": "B",
            "evidence_world": "known", "direction": "left_to_right",
            "left_entity_id": "A::2", "right_entity_id": "B::2",
            "chemical_similarity": 0.9,
        },
    ])
    result = mutual_edge_table(edges, metric_rows(), {"A": 2, "B": 3})
    assert len(result) == 1
    assert result.iloc[0]["left_entity_id"] == "A::1"
    assert result.iloc[0]["left_n_response_contrasts"] == 2
    assert result.iloc[0]["right_n_response_contrasts"] == 3
    assert result.iloc[0]["conservative_wse"] == 0.7


def synthetic_edge_table(prefix: str, size: int) -> pd.DataFrame:
    rows = []
    for index in range(size):
        row = {
            "left_study_id": "A",
            "right_study_id": "B",
            "left_entity_id": f"A::{prefix}{index}",
            "right_entity_id": f"B::{prefix}{index}",
            "physicochemical_compatibility_grade": "grade",
            "left_n_response_contrasts": 2,
            "right_n_response_contrasts": 3,
            "edge_id": f"{prefix}{index}",
        }
        for offset, column in enumerate(MATCH_COVARIATES):
            row[column] = float(index + offset / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_match_edges_preserves_exact_strata_and_does_not_reuse() -> None:
    known = synthetic_edge_table("k", 2)
    unresolved = synthetic_edge_table("u", 5)
    selected, matches, strata = match_edges(
        known, unresolved, base_seed=17, replicate=0
    )
    assert len(selected) == len(known) == 2
    assert selected["edge_id"].nunique() == 2
    assert len(matches) == 2
    assert len(strata) == 1
    assert strata.iloc[0]["selected_unresolved_edges"] == 2
    assert selected[STRATA].drop_duplicates().to_dict("records") == known[
        STRATA
    ].drop_duplicates().to_dict("records")
