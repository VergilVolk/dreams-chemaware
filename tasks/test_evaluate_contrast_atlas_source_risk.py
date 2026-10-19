from __future__ import annotations

import pandas as pd
import pytest

from evaluate_contrast_atlas_source_risk import (
    KNOWN_WORLDS,
    REQUIRED_WORLDS,
    build_query_oracle,
    evaluate_source_risk,
    source_sign_flip,
    validate_outcomes,
)


def outcome_rows(
    *,
    families: int = 4,
    sources_per_family: int = 2,
    axes: tuple[str, ...] = ("local", "field"),
    role: str = "development",
) -> pd.DataFrame:
    rows = []
    for family_index in range(families):
        family = f"family_{family_index}"
        for source_index in range(sources_per_family):
            source = f"{family}_source_{source_index}"
            for axis in axes:
                for world in REQUIRED_WORLDS:
                    correct = world == "all"
                    rows.append({
                        "query_source": source,
                        "query_provenance_family": f"provenance_{source}",
                        "source_family": family,
                        "axis": axis,
                        "world": world,
                        "evaluation_role": role,
                        "whole_query_correct": correct,
                        "whole_query_wrong": False,
                        "whole_query_answered": correct,
                        "candidate_universe_id": f"{source}__{axis}__fixed",
                        "candidate_universe_size": 17,
                        "candidate_universe_frozen": True,
                        "reference_atlas_id": "fixed_reference_atlas",
                        "reference_atlas_frozen": True,
                        "query_reference_source_disjoint": True,
                    })
    return pd.DataFrame(rows)


def set_state(
    frame: pd.DataFrame,
    source: str,
    axis: str,
    world: str,
    state: str,
) -> None:
    mask = (
        frame.query_source.eq(source) & frame.axis.eq(axis) & frame.world.eq(world)
    )
    states = {
        "correct": (True, False, True),
        "wrong": (False, True, True),
        "abstain": (False, False, False),
    }
    correct, wrong, answered = states[state]
    frame.loc[mask, [
        "whole_query_correct", "whole_query_wrong", "whole_query_answered",
    ]] = [correct, wrong, answered]


def test_complete_external_fixture_passes_exact_source_gate() -> None:
    result = evaluate_source_risk(outcome_rows(role="external"))

    assert result.report["evaluation_role"] == "external"
    assert result.report["status"].endswith("_EXTERNAL")
    assert result.report["pass_source_level_exact_axis_gate"] is True
    assert result.report["denominators"]["query_sources"] == 8
    assert result.report["denominators"]["source_families"] == 4
    assert result.report["multiplicity"]["method"] == "intersection_union"
    for comparator in KNOWN_WORLDS:
        comparison = result.report["comparisons"][comparator]
        assert comparison["family_equal_endpoint"]["mean_delta_utility"] == 1.0
        assert comparison["source_sign_flip"]["method"] == "exact_enumeration"
        assert comparison["source_sign_flip"]["p_value"] == pytest.approx(1 / 256)
        assert comparison["transitions"]["raw_source_axis_counts"] == {
            "corrected": 16,
            "lost_correct": 0,
            "wrong_rescued": 0,
            "introduced_wrong": 0,
        }
    assert set(result.sign_flip_distribution.comparator_world) == set(KNOWN_WORLDS)
    assert len(result.sign_flip_distribution) == 512


def test_nominal_sources_cannot_duplicate_one_provenance_family() -> None:
    frame = outcome_rows()
    sources = sorted(frame.query_source.unique())
    shared = frame.loc[
        frame.query_source.eq(sources[0]), "query_provenance_family"
    ].iloc[0]
    frame.loc[
        frame.query_source.eq(sources[1]), "query_provenance_family"
    ] = shared
    with pytest.raises(ValueError, match="provenance family maps to multiple"):
        validate_outcomes(frame)


def test_oracle_envelope_defines_conservative_transition_states() -> None:
    frame = outcome_rows(families=1, sources_per_family=4, axes=("local",))
    sources = sorted(frame.query_source.unique())
    # corrected: all correct, oracle known abstains
    # lost: all abstains, at least one known is correct
    set_state(frame, sources[1], "local", "all", "abstain")
    set_state(frame, sources[1], "local", "known_only", "correct")
    # rescued: all abstains, both known controls are wrong
    set_state(frame, sources[2], "local", "all", "abstain")
    for world in KNOWN_WORLDS:
        set_state(frame, sources[2], "local", world, "wrong")
    # introduced: all wrong, oracle known abstains (one other known may be wrong)
    set_state(frame, sources[3], "local", "all", "wrong")
    set_state(frame, sources[3], "local", "known_only", "wrong")

    validated, _ = validate_outcomes(frame)
    ledger = build_query_oracle(validated).set_index("query_source")

    assert bool(ledger.loc[sources[0], "corrected"])
    assert bool(ledger.loc[sources[1], "lost_correct"])
    assert bool(ledger.loc[sources[2], "wrong_rescued"])
    assert bool(ledger.loc[sources[3], "introduced_wrong"])
    assert ledger.loc[sources[1], "oracle_known_utility"] == 1
    assert ledger.loc[sources[2], "oracle_known_utility"] == -2
    assert ledger.loc[sources[3], "oracle_known_utility"] == 0
    assert bool(ledger.loc[sources[1], "lost_correct__known_only"])
    assert bool(ledger.loc[sources[2], "wrong_rescued__measurement_matched_known"])


def test_family_equal_weighting_and_lofo_do_not_reward_large_families() -> None:
    frame = outcome_rows(families=4, sources_per_family=2, axes=("local",))
    # Add four more sources to family_0.  They have delta zero, so family_0's
    # mean becomes 1/3 while the other three family means remain one.
    extra = outcome_rows(families=1, sources_per_family=4, axes=("local",))
    extra["query_source"] = extra.query_source.str.replace(
        "family_0_source_", "family_0_extra_", regex=False,
    )
    extra["query_provenance_family"] = extra.query_source.map(
        lambda source: f"provenance_{source}"
    )
    for source in extra.query_source.unique():
        set_state(extra, source, "local", "all", "abstain")
    combined = pd.concat([frame, extra], ignore_index=True)

    result = evaluate_source_risk(combined)

    family = result.family_summary.loc[
        result.family_summary.comparator_world.eq("known_only")
    ].set_index("source_family")
    assert family.loc["family_0", "mean_delta_utility"] == pytest.approx(1 / 3)
    assert result.report["comparisons"]["known_only"][
        "family_equal_endpoint"
    ]["mean_delta_utility"] == pytest.approx(
        (1 / 3 + 1 + 1 + 1) / 4
    )
    assert (result.leave_one_family_out.family_equal_mean_delta > 0).all()


def test_iut_requires_both_fixed_comparators_and_oracle_has_no_p_value() -> None:
    frame = outcome_rows()
    # all beats known_only everywhere, but ties measurement_matched_known.
    # The first fixed-comparator source test passes; the second does not.
    for source in frame.query_source.unique():
        for axis in frame.axis.unique():
            set_state(frame, source, axis, "measurement_matched_known", "correct")

    result = evaluate_source_risk(frame)

    assert result.report["comparisons"]["known_only"]["pass_comparator_gate"] is True
    assert result.report["comparisons"]["measurement_matched_known"][
        "pass_comparator_gate"
    ] is False
    assert result.report["gates"]["intersection_union_both_comparators_pass"] is False
    assert result.report["pass_source_level_exact_axis_gate"] is False
    oracle = result.report["oracle_envelope_descriptive"]
    assert oracle["status"] == "DESCRIPTIVE_ONLY_NOT_AN_INFERENTIAL_COMPARATOR"
    assert oracle["family_equal_endpoint"]["mean_delta_utility"] == 0.0
    assert "p_value" not in oracle
    assert "source_sign_flip" not in oracle


@pytest.mark.parametrize(
    "failure", ["family", "candidate", "duplicate", "reference", "role_overlap"],
)
def test_contract_rejects_pseudoreplication_and_unfixed_universes(failure: str) -> None:
    frame = outcome_rows()
    source = frame.query_source.iloc[0]
    if failure == "family":
        frame.loc[
            frame.query_source.eq(source) & frame.world.eq("known_only"), "source_family"
        ] = "different_family"
    elif failure == "candidate":
        frame.loc[
            frame.query_source.eq(source) & frame.world.eq("known_only"),
            "candidate_universe_size",
        ] = 999
    elif failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif failure == "reference":
        frame.loc[frame.query_source.eq(source), "reference_atlas_id"] = "drifted"
    elif failure == "role_overlap":
        frame.loc[frame.query_source.eq(source), "query_reference_source_disjoint"] = False
    with pytest.raises(ValueError):
        validate_outcomes(frame)


def test_heterogeneous_preregistered_axis_sets_are_averaged_within_source() -> None:
    frame = outcome_rows()
    source = sorted(frame.query_source.unique())[0]
    frame = frame.loc[~(
        frame.query_source.eq(source) & frame.axis.eq("field")
    )].copy()

    result = evaluate_source_risk(frame)

    assert result.report["denominators"]["axes_per_source_distribution"] == {
        "1": 1,
        "2": 7,
    }
    known = result.source_summary.loc[
        result.source_summary.comparator_world.eq("known_only")
    ].set_index("query_source")
    assert known.loc[source, "axes"] == 1
    assert known.loc[source, "mean_delta_utility"] == 1.0
    assert result.report["pass_source_level_exact_axis_gate"] is True


def test_more_than_twenty_sources_uses_plus_one_monte_carlo() -> None:
    frame = outcome_rows(families=7, sources_per_family=3, axes=("local",))
    result = evaluate_source_risk(frame, monte_carlo_draws=31, seed=19)
    sign_flip = result.report["comparisons"]["known_only"]["source_sign_flip"]

    assert sign_flip["method"] == "monte_carlo"
    assert sign_flip["random_sign_vectors"] == 31
    assert sign_flip["denominator"] == 32
    assert sign_flip["monte_carlo_correction"] == "plus_one"
    assert sign_flip["p_value"] >= 1 / 32


def test_sign_flip_uses_one_source_sign_after_axis_averaging() -> None:
    result = evaluate_source_risk(outcome_rows())
    known_only = result.source_summary.loc[
        result.source_summary.comparator_world.eq("known_only")
    ]
    report, distribution = source_sign_flip(
        known_only,
        monte_carlo_draws=5,
        seed=1,
        alternative="two-sided",
    )
    assert report["sources"] == 8
    assert report["random_sign_vectors"] == 256
    assert len(distribution) == 256
    assert report["p_value"] == pytest.approx(2 / 256)


def test_sign_flip_rejects_posthoc_mixture_of_comparators() -> None:
    result = evaluate_source_risk(outcome_rows())
    with pytest.raises(ValueError, match="one row per query_source"):
        source_sign_flip(
            result.source_summary,
            monte_carlo_draws=5,
            seed=1,
            alternative="greater",
        )
