from __future__ import annotations

import json

import pandas as pd
import pytest

from evaluate_contrast_atlas_ranking_method_increment import (
    BASELINE_METHODS,
    PRIMARY_METHOD,
    evaluate_ranking_method_increment,
)


def method_outcomes(
    method: str,
    states: dict[tuple[str, str], str],
    *,
    role: str = "external",
    include_other_worlds: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_index in range(8):
        source = f"S{source_index}"
        family = f"F{source_index // 2}"
        axes = ("axis_a", "axis_b") if source_index == 0 else ("axis_a",)
        for axis in axes:
            state = states.get((source, axis), "abstain")
            base = {
                "ranking_method": method,
                "query_source": source,
                "source_family": family,
                "query_provenance_family": f"PF_{source}",
                "axis": axis,
                "evaluation_role": role,
                "whole_query_correct": state == "correct",
                "whole_query_wrong": state == "wrong",
                "whole_query_answered": state != "abstain",
                "candidate_universe_id": f"U_{source}_{axis}",
                "candidate_universe_sha256": f"UH_{source}_{axis}",
                "candidate_universe_size": 3,
                "candidate_universe_frozen": True,
                "reference_atlas_id": "REF_FROZEN",
                "reference_atlas_frozen": True,
                "query_reference_source_disjoint": True,
                "query_panel_id": f"QP_{source}",
                "query_contrast_id": f"QC_{axis}",
                "expected_direction": "similar",
                "formal_truth_registry_query_sha256": f"TRUTH_{source}_{axis}",
                "shared_primary_answerability_mask_sha256": f"MASK_{source}_{axis}",
                "answerable_candidates": 2 if state != "abstain" else 0,
            }
            rows.append({"world": "all", **base})
            if include_other_worlds:
                rows.append({
                    "world": "known_only", **base,
                    "whole_query_correct": False,
                    "whole_query_wrong": True,
                    "whole_query_answered": True,
                })
    return pd.DataFrame(rows)


def all_states(state: str) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for source_index in range(8):
        source = f"S{source_index}"
        axes = ("axis_a", "axis_b") if source_index == 0 else ("axis_a",)
        for axis in axes:
            result[(source, axis)] = state
    return result


def test_complete_two_baseline_gate_passes_with_exact_source_sign_flip() -> None:
    primary = method_outcomes(PRIMARY_METHOD, all_states("correct"), include_other_worlds=True)
    greedy = method_outcomes("greedy_same_graph", all_states("abstain"), include_other_worlds=True)
    centroid = method_outcomes("centroid_same_encoder", all_states("abstain"), include_other_worlds=True)
    result = evaluate_ranking_method_increment(primary, greedy, centroid)
    assert result.report["pass_formal_ranking_method_increment_gate"] is True
    assert result.report["denominators"] == {
        "query_source_axes": 9,
        "held_out_query_sources": 8,
        "biological_families": 4,
        "axes_per_source_distribution": {"1": 7, "2": 1},
    }
    for baseline in BASELINE_METHODS:
        comparison = result.report["comparisons"][baseline]
        assert comparison["pass_component_gate"] is True
        assert comparison["source_sign_flip"]["method"] == "exact_enumeration"
        assert comparison["source_sign_flip"]["p_value"] == pytest.approx(1 / 256)
        assert comparison["denominators"]["positive_families"] == 4
    source = result.source_summary.loc[
        result.source_summary.baseline_method.eq("greedy_same_graph")
    ]
    assert len(source) == 8
    assert source.loc[source.query_source.eq("S0"), "axes"].iloc[0] == 2
    assert source.loc[source.query_source.eq("S0"), "mean_delta_utility"].iloc[0] == 1.0


def test_iut_fails_when_only_one_fixed_baseline_is_beaten() -> None:
    result = evaluate_ranking_method_increment(
        method_outcomes(PRIMARY_METHOD, all_states("correct")),
        method_outcomes("greedy_same_graph", all_states("abstain")),
        method_outcomes("centroid_same_encoder", all_states("correct")),
    )
    assert result.report["comparisons"]["greedy_same_graph"]["pass_component_gate"] is True
    assert result.report["comparisons"]["centroid_same_encoder"]["pass_component_gate"] is False
    assert result.report["gates"]["intersection_union_both_fixed_baselines_pass"] is False
    assert result.report["pass_formal_ranking_method_increment_gate"] is False


def test_three_quarters_positive_does_not_override_one_harmful_family() -> None:
    primary_states = all_states("correct")
    centroid_states = all_states("abstain")
    for source in ("S6", "S7"):
        primary_states[(source, "axis_a")] = "abstain"
        centroid_states[(source, "axis_a")] = "correct"
    result = evaluate_ranking_method_increment(
        method_outcomes(PRIMARY_METHOD, primary_states),
        method_outcomes("greedy_same_graph", all_states("abstain")),
        method_outcomes("centroid_same_encoder", centroid_states),
    )
    comparison = result.report["comparisons"]["centroid_same_encoder"]
    assert comparison["denominators"]["positive_families"] == 3
    assert comparison["gates"]["at_least_three_quarters_families_strictly_positive"] is True
    assert comparison["gates"]["every_family_nonharmful"] is False
    assert comparison["pass_component_gate"] is False


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("candidate_universe_id", "DRIFT", "candidate_universe_id differs"),
        ("reference_atlas_id", "OTHER_REF", "reference_atlas_id differs"),
        ("evaluation_role", "development", "evaluation_role differs"),
        ("formal_truth_registry_query_sha256", "OTHER_TRUTH", "formal_truth_registry_query_sha256 differs"),
        ("shared_primary_answerability_mask_sha256", "OTHER_MASK", "shared_primary_answerability_mask_sha256 differs"),
    ],
)
def test_cross_method_invariant_drift_fails_closed(
    column: str, replacement: str, message: str,
) -> None:
    primary = method_outcomes(PRIMARY_METHOD, all_states("correct"))
    greedy = method_outcomes("greedy_same_graph", all_states("abstain"))
    centroid = method_outcomes("centroid_same_encoder", all_states("abstain"))
    centroid[column] = replacement
    with pytest.raises(RuntimeError, match=message):
        evaluate_ranking_method_increment(primary, greedy, centroid)


def test_query_axis_universe_drift_and_invalid_outcome_fail_closed() -> None:
    primary = method_outcomes(PRIMARY_METHOD, all_states("correct"))
    greedy = method_outcomes("greedy_same_graph", all_states("abstain")).iloc[1:].copy()
    centroid = method_outcomes("centroid_same_encoder", all_states("abstain"))
    with pytest.raises(RuntimeError, match="query/axis universe differs"):
        evaluate_ranking_method_increment(primary, greedy, centroid)

    greedy = method_outcomes("greedy_same_graph", all_states("abstain"))
    greedy.loc[greedy.index[0], ["whole_query_correct", "whole_query_wrong"]] = True
    with pytest.raises(RuntimeError, match="both correct and wrong"):
        evaluate_ranking_method_increment(primary, greedy, centroid)

    greedy = method_outcomes("greedy_same_graph", all_states("abstain"))
    greedy["candidate_universe_size"] = greedy.candidate_universe_size.astype(float)
    greedy.loc[greedy.index[0], "candidate_universe_size"] = 2.5
    with pytest.raises(RuntimeError, match="finite integers"):
        evaluate_ranking_method_increment(primary, greedy, centroid)

    greedy = method_outcomes("greedy_same_graph", all_states("abstain"))
    greedy.loc[greedy.query_source.eq("S1"), "query_provenance_family"] = "PF_S0"
    with pytest.raises(RuntimeError, match="one provenance family to multiple nominal"):
        evaluate_ranking_method_increment(primary, greedy, centroid)


def test_evaluator_does_not_require_equal_method_answer_or_candidate_counts() -> None:
    primary = method_outcomes(PRIMARY_METHOD, all_states("correct"))
    greedy = method_outcomes("greedy_same_graph", all_states("abstain"))
    centroid = method_outcomes("centroid_same_encoder", all_states("wrong"))
    centroid["answerable_candidates"] = 1
    result = evaluate_ranking_method_increment(primary, greedy, centroid)
    ledger = result.query_axis_ledger
    assert (ledger[f"utility__{PRIMARY_METHOD}"] == 1).all()
    assert (ledger["utility__greedy_same_graph"] == 0).all()
    assert (ledger["utility__centroid_same_encoder"] == -2).all()
    assert (ledger["answerable_candidates__greedy_same_graph"] == 0).all()
    assert (ledger["answerable_candidates__centroid_same_encoder"] == 1).all()


def test_insufficient_denominators_fail_gate_without_nonfinite_report() -> None:
    primary = method_outcomes(PRIMARY_METHOD, all_states("correct")).iloc[:2].copy()
    greedy = method_outcomes("greedy_same_graph", all_states("abstain")).iloc[:2].copy()
    centroid = method_outcomes("centroid_same_encoder", all_states("abstain")).iloc[:2].copy()
    result = evaluate_ranking_method_increment(primary, greedy, centroid)
    assert result.report["pass_formal_ranking_method_increment_gate"] is False
    assert result.report["gates"]["at_least_four_biological_families"] is False
    for baseline in BASELINE_METHODS:
        assert result.report["comparisons"][baseline][
            "leave_one_family_out_minimum_delta"
        ] is None
    json.dumps(result.report, allow_nan=False)
