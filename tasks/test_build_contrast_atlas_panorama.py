from __future__ import annotations

import pandas as pd

from build_contrast_atlas_panorama import (
    IDENTITY_TRACK,
    build_world_comparison,
    combined_two_face_answers,
)


def _top_rows() -> pd.DataFrame:
    rows = []
    for query_index in range(51):
        for world in ("all", "known", "unresolved"):
            for face, pvalue in (("similar", 0.02), ("opposite", 0.40)):
                rows.append({
                    "query_study_id": f"S{query_index}",
                    "query_contrast_id": "q",
                    "evidence_world": world,
                    "evidence_track": IDENTITY_TRACK,
                    "response_face": face,
                    "candidate_study_id": "T",
                    "candidate_contrast_id": "c",
                    "weighted_evidence_edges": 3,
                    "enrichment_z": 2.0,
                    "query_maxT_p": pvalue,
                })
    return pd.DataFrame(rows)


def test_combined_answer_applies_two_face_bonferroni() -> None:
    result = combined_two_face_answers(_top_rows())
    assert len(result) == 153
    assert set(result["response_face"]) == {"similar"}
    assert set(result["two_face_bonferroni_p"]) == {0.04}
    assert set(result["combined_answer_class"]) == {"calibrated_answer"}


def test_world_comparison_does_not_call_unknown_correctness() -> None:
    combined = combined_two_face_answers(_top_rows())
    known_index = combined["evidence_world"].eq("known")
    combined.loc[known_index, "candidate_study_id"] = "KNOWN_OTHER"
    result = build_world_comparison(combined)
    assert result["unresolved_supported_not_known_selection"].all()
    assert not result["known_corroborated_selection"].any()
    assert "correct" not in " ".join(result.columns).lower()


def test_insufficient_edge_support_abstains() -> None:
    top = _top_rows()
    mask = (
        top["query_study_id"].eq("S0")
        & top["evidence_world"].eq("all")
    )
    top.loc[mask, "weighted_evidence_edges"] = 2
    result = combined_two_face_answers(top)
    row = result.loc[
        result["query_study_id"].eq("S0")
        & result["evidence_world"].eq("all")
    ].iloc[0]
    assert row["combined_answer_class"] == "insufficient_evidence"
    assert row["two_face_bonferroni_p"] == 1.0
