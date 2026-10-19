from __future__ import annotations

import pandas as pd
import pytest

from build_pan_redu_open_retrieval_view import build_open_view


def row(world: str, target: str, score: float, lower: float, upper: float, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "world": world,
        "left_panel_id": "query_panel",
        "left_contrast_id": "query_contrast",
        "right_panel_id": f"target_{target}",
        "right_contrast_id": f"contrast_{target}",
        "answerable": True,
        "direction_identified": True,
        "signed_chemical_evidence_lower": lower,
        "signed_chemical_evidence_upper": upper,
        "signed_chemical_evidence_conservative_bound": score,
        "greedy_signed_chemical_evidence": score / 2,
        "greedy_intrinsic_score_available": True,
        "centroid_cosine": score / 3,
        "centroid_intrinsic_score_available": True,
        "centroid_scope_supported": True,
    }
    values.update(overrides)
    return values


def fixed_two_world_matrix() -> pd.DataFrame:
    rows = []
    for world in ("all", "known_only"):
        rows.extend([
            row(world, "positive", 0.40, 0.20, 0.50),
            row(world, "negative", -0.55, -0.70, -0.10),
            # A huge score must still be vetoed because the interval crosses 0,
            # even when the upstream answerable flag is true.
            row(world, "crosses_zero", 0.99, -0.01, 0.02),
        ])
    return pd.DataFrame(rows)


def test_truth_blind_view_ranks_both_directions_and_retains_vetoed_candidate() -> None:
    full, tops = build_open_view(fixed_two_world_matrix())
    assert len(full) == 6 * 3
    assert set(full.groupby(["world", "method"]).size()) == {3}
    vetoed = full.loc[full.right_panel_id.eq("target_crosses_zero")]
    assert not vetoed.ranking_eligible.any()
    assert set(vetoed.ineligibility_reason) == {"signed_interval_contains_zero"}
    answered = tops.loc[
        tops.world.eq("all") & tops.method.eq("transport_conservative")
    ].set_index("direction")
    assert answered.loc["similar", "target_panel_id"] == "target_positive"
    assert answered.loc["opposite", "target_panel_id"] == "target_negative"
    assert answered.status.eq("ANSWERED_UNIQUE_TOP").all()


def test_no_direction_identified_candidate_is_abstention_not_no_match() -> None:
    matrix = pd.DataFrame([
        row(
            "all", "only", 0.80, -0.10, 0.20,
            direction_identified=False,
        ),
    ])
    full, tops = build_open_view(matrix)
    assert not full.ranking_eligible.any()
    assert set(tops.status) == {"ABSTAIN_NO_DIRECTION_IDENTIFIED_CANDIDATE"}
    assert not tops.status.str.contains("NO_MATCH").any()
    assert tops.target_panel_id.eq("").all()


def test_candidate_universe_must_be_identical_across_worlds() -> None:
    matrix = fixed_two_world_matrix()
    matrix = matrix.loc[
        ~(matrix.world.eq("known_only") & matrix.right_panel_id.eq("target_negative"))
    ].copy()
    with pytest.raises(RuntimeError, match="candidate universe changes across worlds"):
        build_open_view(matrix)


def test_method_unavailability_does_not_delete_candidate() -> None:
    matrix = pd.DataFrame([
        row(
            "all", "only", 0.50, 0.20, 0.60,
            centroid_intrinsic_score_available=False,
        ),
    ])
    full, tops = build_open_view(matrix)
    centroid = full.loc[full.method.eq("centroid_same_encoder")]
    assert len(centroid) == 1
    assert not centroid.ranking_eligible.iloc[0]
    centroid_top = tops.loc[tops.method.eq("centroid_same_encoder")]
    assert set(centroid_top.status) == {"ABSTAIN_METHOD_UNAVAILABLE"}


def test_truth_and_comparison_columns_are_not_read_or_propagated() -> None:
    matrix = fixed_two_world_matrix()
    matrix["truth_class"] = ["positive", "negative", "unknown"] * 2
    matrix["expected_direction"] = "similar"
    matrix["comparison_stratum"] = "forbidden_label"
    full, tops = build_open_view(matrix)
    forbidden = {"truth_class", "expected_direction", "comparison_stratum"}
    assert not (forbidden & set(full.columns))
    assert not (forbidden & set(tops.columns))
