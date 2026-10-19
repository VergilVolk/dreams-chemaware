from __future__ import annotations

import pandas as pd
import pytest

from build_pan_redu_exact_axis_source_outcomes import (
    WORLDS,
    build_exact_axis_source_outcomes,
)
from evaluate_contrast_atlas_source_risk import validate_outcomes


def fixture(role: str = "development") -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame([
        {
            "query_panel_id": "Q", "query_contrast_id": "q1",
            "target_panel_id": "A", "target_contrast_id": "a1",
            "query_source": "query_source", "query_source_family": "query_family",
            "target_source": "target_source_a", "target_source_family": "target_family_a",
            "exact_axis_family_id": "human_ad_skin", "axis": "exact_axis",
            "target_exact_axis_family_id": "human_ad_skin", "target_axis": "exact_axis",
            "truth_class": "exact_replication",
            "expected_direction": "similar", "eligible_exact_axis": True,
            "evaluation_role": role, "truth_frozen_before_scores": True,
            "selection_blind_to_transport_scores": True,
            "candidate_universe_frozen_before_scores": True,
            "registry_complete_for_query": True,
        },
        {
            "query_panel_id": "Q", "query_contrast_id": "q1",
            "target_panel_id": "B", "target_contrast_id": "b1",
            "query_source": "query_source", "query_source_family": "query_family",
            "target_source": "target_source_b", "target_source_family": "target_family_b",
            "exact_axis_family_id": "human_ad_skin", "axis": "exact_axis",
            "target_exact_axis_family_id": "human_ad_skin", "target_axis": "mismatch_axis",
            "truth_class": "explicit_mismatch",
            "expected_direction": "not_applicable", "eligible_exact_axis": True,
            "evaluation_role": role, "truth_frozen_before_scores": True,
            "selection_blind_to_transport_scores": True,
            "candidate_universe_frozen_before_scores": True,
            "registry_complete_for_query": True,
        },
    ])
    rows = []
    scores = {
        "all": {"A": (True, 0.9), "B": (True, 0.1)},
        "known_only": {"A": (True, 0.1), "B": (True, 0.8)},
        "measurement_matched_known": {"A": (False, 0.0), "B": (False, 0.0)},
    }
    for world in WORLDS:
        for target, contrast in (("A", "a1"), ("B", "b1")):
            answerable, score = scores[world][target]
            rows.append({
                "world": world,
                "left_panel_id": "Q", "left_contrast_id": "q1",
                "right_panel_id": target, "right_contrast_id": contrast,
                "answerable": answerable,
                "direction_identified": answerable,
                "signed_chemical_evidence_conservative_bound": score,
                "greedy_signed_chemical_evidence": (
                    0.1 if target == "A" else 0.9
                ),
                "centroid_cosine": 0.8 if target == "A" else 0.2,
                "centroid_scope_supported": True,
            })
    return pd.DataFrame(rows), truth


def test_builds_fixed_universe_correct_wrong_and_abstain_for_source_gate() -> None:
    matrix, truth = fixture(role="external")
    outcomes, candidates, report = build_exact_axis_source_outcomes(matrix, truth)

    indexed = outcomes.set_index("world")
    assert bool(indexed.loc["all", "whole_query_correct"])
    assert bool(indexed.loc["known_only", "whole_query_wrong"])
    assert not bool(indexed.loc["measurement_matched_known", "whole_query_answered"])
    assert indexed.candidate_universe_id.nunique() == 1
    assert indexed.candidate_universe_sha256.nunique() == 1
    assert indexed.shared_primary_answerability_mask_sha256.notna().all()
    assert set(indexed.candidate_universe_size) == {2}
    assert indexed.candidate_universe_frozen.all()
    assert indexed.reference_atlas_frozen.all()
    assert indexed.query_reference_source_disjoint.all()
    assert indexed.reference_atlas_id.nunique() == 1
    assert set(candidates.groupby("world").size()) == {2}
    assert "shared_primary_answerability_mask_sha256" in candidates
    assert (
        candidates.groupby("world").shared_primary_answerability_mask_sha256.nunique()
        == 1
    ).all()
    assert report["evaluation_role"] == "external"
    validated, role = validate_outcomes(outcomes)
    assert len(validated) == 3 and role == "external"


def test_same_universe_baseline_can_disagree_with_primary_without_changing_mask() -> None:
    matrix, truth = fixture()
    primary, _, primary_report = build_exact_axis_source_outcomes(matrix, truth)
    greedy, greedy_candidates, greedy_report = build_exact_axis_source_outcomes(
        matrix, truth, ranking_method="greedy_same_graph",
    )
    centroid, _, centroid_report = build_exact_axis_source_outcomes(
        matrix, truth, ranking_method="centroid_same_encoder",
    )

    assert bool(primary.loc[primary.world.eq("all"), "whole_query_correct"].iloc[0])
    assert bool(greedy.loc[greedy.world.eq("all"), "whole_query_wrong"].iloc[0])
    assert bool(centroid.loc[centroid.world.eq("all"), "whole_query_correct"].iloc[0])
    assert primary_report["ranking_method"] == "transport_conservative"
    assert greedy_report["ranking_score_column"] == "greedy_signed_chemical_evidence"
    assert centroid_report["ranking_score_column"] == "centroid_cosine"
    assert set(greedy_candidates.ranking_method) == {"greedy_same_graph"}
    for world in WORLDS:
        assert (
            primary.loc[primary.world.eq(world), "candidate_universe_id"].iloc[0]
            == greedy.loc[greedy.world.eq(world), "candidate_universe_id"].iloc[0]
        )
        hashes = {
            frame.loc[
                frame.world.eq(world), "shared_primary_answerability_mask_sha256"
            ].iloc[0]
            for frame in (primary, greedy, centroid)
        }
        assert len(hashes) == 1


def test_shared_primary_answerability_hash_changes_only_with_primary_mask() -> None:
    matrix, truth = fixture()
    original, _, _ = build_exact_axis_source_outcomes(matrix, truth)
    score_changed = matrix.copy()
    score_changed["signed_chemical_evidence_conservative_bound"] = 0.42
    score_changed["greedy_signed_chemical_evidence"] = -0.37
    score_only, _, _ = build_exact_axis_source_outcomes(score_changed, truth)
    assert original.shared_primary_answerability_mask_sha256.tolist() \
        == score_only.shared_primary_answerability_mask_sha256.tolist()

    mask_changed = matrix.copy()
    changed = mask_changed.world.eq("all") & mask_changed.right_panel_id.eq("B")
    mask_changed.loc[changed, ["answerable", "direction_identified"]] = False
    altered, _, _ = build_exact_axis_source_outcomes(mask_changed, truth)
    original_hash = original.loc[
        original.world.eq("all"), "shared_primary_answerability_mask_sha256"
    ].iloc[0]
    altered_hash = altered.loc[
        altered.world.eq("all"), "shared_primary_answerability_mask_sha256"
    ].iloc[0]
    assert original_hash != altered_hash


def test_centroid_requires_global_scope_support() -> None:
    matrix, truth = fixture()
    matrix.loc[matrix.world.eq("all"), "centroid_scope_supported"] = False
    with pytest.raises(RuntimeError, match="global-scope support"):
        build_exact_axis_source_outcomes(
            matrix, truth, ranking_method="centroid_same_encoder",
        )


def test_extra_unregistered_matrix_target_is_excluded_not_called_negative() -> None:
    matrix, truth = fixture()
    extra = matrix.loc[matrix.right_panel_id.eq("B")].copy()
    extra["right_panel_id"] = "C"
    extra["right_contrast_id"] = "c1"
    extra["signed_chemical_evidence_conservative_bound"] = 99.0
    matrix = pd.concat([matrix, extra], ignore_index=True)
    outcomes, candidates, report = build_exact_axis_source_outcomes(matrix, truth)

    assert report["denominators"]["excluded_as_unadjudicated"] == 1
    assert "C" not in set(candidates.target_panel_id)
    assert "C" not in set(outcomes.top_target_panel_id)


def test_unknown_truth_never_becomes_a_negative() -> None:
    matrix, truth = fixture()
    truth.loc[truth.target_panel_id.eq("B"), "truth_class"] = "unknown"
    with pytest.raises(RuntimeError, match="unknown/unreviewed"):
        build_exact_axis_source_outcomes(matrix, truth)


def test_target_axis_must_independently_support_truth_class() -> None:
    matrix, truth = fixture()
    truth.loc[truth.truth_class.eq("exact_replication"), "target_axis"] = "different_axis"
    with pytest.raises(RuntimeError, match="exact replication must independently declare"):
        build_exact_axis_source_outcomes(matrix, truth)

    _, truth = fixture()
    truth.loc[truth.truth_class.eq("explicit_mismatch"), "target_axis"] = "exact_axis"
    with pytest.raises(RuntimeError, match="explicit mismatch cannot independently declare"):
        build_exact_axis_source_outcomes(matrix, truth)


@pytest.mark.parametrize("overlap", ["source", "family"])
def test_query_target_source_overlap_is_rejected(overlap: str) -> None:
    matrix, truth = fixture()
    column = "target_source" if overlap == "source" else "target_source_family"
    value = "query_source" if overlap == "source" else "query_family"
    truth.loc[0, column] = value
    with pytest.raises(RuntimeError, match="source.*overlap"):
        build_exact_axis_source_outcomes(matrix, truth)


def test_world_candidate_drift_is_rejected() -> None:
    matrix, truth = fixture()
    matrix = matrix.loc[~(
        matrix.world.eq("known_only") & matrix.right_panel_id.eq("B")
    )].copy()
    with pytest.raises(RuntimeError, match="registered formal candidate is absent"):
        build_exact_axis_source_outcomes(matrix, truth)


def test_query_source_cannot_reappear_as_another_querys_reference() -> None:
    matrix, truth = fixture()
    second_matrix = matrix.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
    })
    second_truth = truth.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
        "query_source": "query_source_2",
        "query_family": "query_family_2",
        "target_source_a": "query_source",
        "target_source_b": "target_source_d",
        "target_family_a": "target_family_c",
        "target_family_b": "target_family_d",
    })
    with pytest.raises(RuntimeError, match="global query/reference source-role overlap"):
        build_exact_axis_source_outcomes(
            pd.concat([matrix, second_matrix], ignore_index=True),
            pd.concat([truth, second_truth], ignore_index=True),
        )


def test_nominal_queries_cannot_duplicate_one_provenance_family() -> None:
    matrix, truth = fixture()
    second_matrix = matrix.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
    })
    second_truth = truth.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
        "query_source": "query_source_2",
        "target_source_a": "target_source_c",
        "target_source_b": "target_source_d",
        "target_family_a": "target_family_c",
        "target_family_b": "target_family_d",
    })
    with pytest.raises(RuntimeError, match="provenance family maps to multiple"):
        build_exact_axis_source_outcomes(
            pd.concat([matrix, second_matrix], ignore_index=True),
            pd.concat([truth, second_truth], ignore_index=True),
        )


def test_score_blind_complete_universe_attestations_are_required() -> None:
    matrix, truth = fixture()
    truth.loc[0, "registry_complete_for_query"] = False
    with pytest.raises(RuntimeError, match="score blind and frozen"):
        build_exact_axis_source_outcomes(matrix, truth)


def test_tie_break_uses_only_deterministic_target_ids() -> None:
    matrix, truth = fixture()
    mask = matrix.world.eq("all")
    matrix.loc[mask, "signed_chemical_evidence_conservative_bound"] = 0.5
    outcomes, _, _ = build_exact_axis_source_outcomes(matrix, truth)
    top = outcomes.loc[outcomes.world.eq("all")].iloc[0]
    assert top.top_target_panel_id == "A"
    assert bool(top.whole_query_correct)


def test_similar_expected_rejects_confident_wrong_sign_candidates() -> None:
    matrix, truth = fixture()
    mask = matrix.world.eq("all")
    matrix.loc[mask & matrix.right_panel_id.eq("A"),
               "signed_chemical_evidence_conservative_bound"] = -0.9
    matrix.loc[mask & matrix.right_panel_id.eq("B"),
               "signed_chemical_evidence_conservative_bound"] = -0.1
    outcomes, _, _ = build_exact_axis_source_outcomes(matrix, truth)
    all_outcome = outcomes.loc[outcomes.world.eq("all")].iloc[0]
    assert not bool(all_outcome.whole_query_answered)
    assert all_outcome.top_truth_class == "abstained"


def test_opposite_expected_negates_bound_before_positive_direction_gate() -> None:
    matrix, truth = fixture()
    truth.loc[truth.truth_class.eq("exact_replication"), "expected_direction"] = "opposite"
    mask = matrix.world.eq("all")
    matrix.loc[mask & matrix.right_panel_id.eq("A"),
               "signed_chemical_evidence_conservative_bound"] = -0.9
    matrix.loc[mask & matrix.right_panel_id.eq("B"),
               "signed_chemical_evidence_conservative_bound"] = 0.8
    outcomes, _, _ = build_exact_axis_source_outcomes(matrix, truth)
    all_outcome = outcomes.loc[outcomes.world.eq("all")].iloc[0]
    assert bool(all_outcome.whole_query_correct)
    assert all_outcome.top_target_panel_id == "A"
    assert all_outcome.top_signed_conservative_score == pytest.approx(0.9)


def test_exact_axis_family_is_distinct_from_provenance_family_for_weighting() -> None:
    matrix, truth = fixture()
    second_matrix = matrix.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
    })
    second_truth = truth.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
        "query_source": "query_source_2",
        "query_family": "independent_query_provenance_2",
        "target_source_a": "target_source_c",
        "target_source_b": "target_source_d",
        "target_family_a": "target_family_c",
        "target_family_b": "target_family_d",
    })
    outcomes, _, _ = build_exact_axis_source_outcomes(
        pd.concat([matrix, second_matrix], ignore_index=True),
        pd.concat([truth, second_truth], ignore_index=True),
    )

    assert set(outcomes.query_provenance_family) == {
        "query_family", "independent_query_provenance_2",
    }
    assert set(outcomes.source_family) == {"human_ad_skin"}
    from evaluate_contrast_atlas_source_risk import evaluate_source_risk

    source_result = evaluate_source_risk(outcomes)
    assert source_result.report["denominators"]["query_sources"] == 2
    assert source_result.report["denominators"]["source_families"] == 1
    assert set(source_result.family_summary.source_family) == {"human_ad_skin"}


def test_one_source_may_have_multiple_axes_but_not_multiple_exact_families() -> None:
    matrix, truth = fixture()
    second_matrix = matrix.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
    })
    second_truth = truth.replace({
        "Q": "Q2", "A": "C", "B": "D", "q1": "q2",
        "a1": "c1", "b1": "d1",
        "exact_axis": "second_axis",
        "target_source_a": "target_source_c",
        "target_source_b": "target_source_d",
        "target_family_a": "target_family_c",
        "target_family_b": "target_family_d",
    })
    combined_matrix = pd.concat([matrix, second_matrix], ignore_index=True)
    combined_truth = pd.concat([truth, second_truth], ignore_index=True)

    outcomes, _, _ = build_exact_axis_source_outcomes(combined_matrix, combined_truth)
    assert set(outcomes.axis) == {"exact_axis", "second_axis"}
    assert set(outcomes.source_family) == {"human_ad_skin"}

    invalid = combined_truth.copy()
    invalid.loc[invalid.axis.eq("second_axis"), "exact_axis_family_id"] = "other_biology"
    invalid.loc[
        invalid.axis.eq("second_axis"), "target_exact_axis_family_id"
    ] = "other_biology"
    with pytest.raises(RuntimeError, match="one query_source maps to multiple"):
        build_exact_axis_source_outcomes(combined_matrix, invalid)
