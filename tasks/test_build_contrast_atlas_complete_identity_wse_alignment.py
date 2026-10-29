from __future__ import annotations

import numpy as np
import pandas as pd

from build_contrast_atlas_complete_identity_wse_alignment import (
    reciprocal_edge_summary,
    select_directional_top1,
)


def synthetic_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # All and known intentionally have different winners.
            {
                "query_block_index": 0,
                "query_study_id": "A",
                "query_entity_id": "A::known",
                "candidate_study_id": "B",
                "candidate_entity_id": "B::known",
                "query_identity_status": "library_supported",
                "candidate_identity_status": "library_supported",
            },
            {
                "query_block_index": 0,
                "query_study_id": "A",
                "query_entity_id": "A::known",
                "candidate_study_id": "B",
                "candidate_entity_id": "B::unresolved",
                "query_identity_status": "library_supported",
                "candidate_identity_status": "unresolved_entity",
            },
            # Reciprocal all-world decision for the mixed winner.
            {
                "query_block_index": 1,
                "query_study_id": "B",
                "query_entity_id": "B::unresolved",
                "candidate_study_id": "A",
                "candidate_entity_id": "A::known",
                "query_identity_status": "unresolved_entity",
                "candidate_identity_status": "library_supported",
            },
            # Exact top-score tie must abstain even with stable IDs.
            {
                "query_block_index": 2,
                "query_study_id": "A",
                "query_entity_id": "A::u1",
                "candidate_study_id": "C",
                "candidate_entity_id": "C::u1",
                "query_identity_status": "unresolved_entity",
                "candidate_identity_status": "unresolved_entity",
            },
            {
                "query_block_index": 2,
                "query_study_id": "A",
                "query_entity_id": "A::u1",
                "candidate_study_id": "C",
                "candidate_entity_id": "C::u2",
                "query_identity_status": "unresolved_entity",
                "candidate_identity_status": "unresolved_entity",
            },
            # A singleton with no matched spectral evidence must abstain.
            {
                "query_block_index": 3,
                "query_study_id": "C",
                "query_entity_id": "C::u3",
                "candidate_study_id": "A",
                "candidate_entity_id": "A::u2",
                "query_identity_status": "unresolved_entity",
                "candidate_identity_status": "unresolved_entity",
            },
        ]
    )


def test_worlds_are_reranked_instead_of_filtering_all_winner() -> None:
    pairs = synthetic_pairs()
    score = np.asarray([0.8, 0.9, 0.9, 0.7, 0.7, 0.0])
    matched = np.asarray([6, 7, 7, 5, 4, 0])
    available = np.ones(len(pairs), dtype=bool)

    all_edges, all_summary = select_directional_top1(
        pairs, score, matched, available, "all"
    )
    known_edges, known_summary = select_directional_top1(
        pairs, score, matched, available, "known"
    )
    unresolved_edges, unresolved_summary = select_directional_top1(
        pairs, score, matched, available, "unresolved"
    )

    a_to_b_all = all_edges.loc[all_edges["direction"].eq("left_to_right")].iloc[0]
    assert a_to_b_all["right_entity_id"] == "B::unresolved"
    assert known_edges.iloc[0]["right_entity_id"] == "B::known"
    assert known_summary["eligible_candidate_pairs"] == 1
    assert unresolved_edges.empty
    assert unresolved_summary["tied_top1_abstentions"] == 1
    assert unresolved_summary["no_positive_spectral_evidence_abstentions"] == 1
    assert all_summary["directional_top1_edges"] == 2


def test_canonical_directions_form_one_reciprocal_edge() -> None:
    pairs = synthetic_pairs().iloc[:3].copy()
    edges, _summary = select_directional_top1(
        pairs,
        np.asarray([0.8, 0.9, 0.9]),
        np.asarray([6, 7, 7]),
        np.ones(3, dtype=bool),
        "all",
    )
    assert set(edges["direction"]) == {"left_to_right", "right_to_left"}
    assert edges["left_entity_id"].nunique() == 1
    assert edges["right_entity_id"].nunique() == 1
    assert reciprocal_edge_summary(edges)["canonical_mutual_edges"] == 1
