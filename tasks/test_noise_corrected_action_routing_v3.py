"""Tests for the three-semantics direct-v3 action selector."""
from __future__ import annotations

import pandas as pd
import numpy as np

from noise_corrected_action_routing_v3 import (
    score_candidate_boundary,
    select_diverse_routed_actions_v3,
)
from noise_corrected_shuffled_control_v3 import source_family_shuffled_action_bank


def test_candidate_boundary_records_exact_action_switch_rows() -> None:
    rows = np.asarray([10, 11, 20, 21, 30], dtype=np.int64)
    ptr = np.asarray([0, 2, 4, 5], dtype=np.int64)
    embeddings = np.asarray([
        [0.8, 0.2], [0.9, 0.1],
        [0.7, 0.3], [0.1, 0.9],
        [0.6, 0.4],
    ], dtype=np.float32)
    score = score_candidate_boundary(
        rows, ptr, embeddings, np.asarray([0.0, 1.0], dtype=np.float32),
    )
    assert score.positive_row == 10
    assert score.hard_negative_molecule_index == 1
    assert score.hard_negative_row == 21
    assert score.rank == 3
    assert np.isclose(score.margin, -0.7)


def fixture() -> pd.DataFrame:
    rows = []
    routes = ("corrective", "harmful", "robustness_only", "uncertain")
    for route in routes:
        for family in ("N", "P"):
            for copy in range(5):
                rows.append({
                    "query_index": 3,
                    "action_id": f"{route}|{family}|{copy}",
                    "source": family,
                    "family": family,
                    "route": route,
                    "action_margin": 0.10 - 0.01 * copy,
                    "margin_change": 0.05 - 0.01 * copy,
                    "paired_advantage": 0.04 - 0.01 * copy,
                })
    return pd.DataFrame(rows)


def test_all_three_semantics_are_bounded_and_uncertain_is_audit_only() -> None:
    output = select_diverse_routed_actions_v3(
        fixture(),
        maximum_corrective_per_query=4,
        maximum_harmful_per_query=3,
        maximum_robust_per_query=2,
    )
    assert int(output.selected_corrective.sum()) == 4
    assert int(output.selected_harmful.sum()) == 3
    assert int(output.selected_robust.sum()) == 2
    uncertain = output.route.eq("uncertain")
    assert not bool(output.loc[uncertain, [
        "selected_corrective", "selected_harmful", "selected_robust",
    ]].any(axis=None))


def test_round_robin_preserves_source_family_diversity() -> None:
    output = select_diverse_routed_actions_v3(
        fixture(),
        maximum_corrective_per_query=2,
        maximum_harmful_per_query=2,
        maximum_robust_per_query=2,
    )
    for column in ("selected_corrective", "selected_harmful", "selected_robust"):
        assert set(output.loc[output[column], "family"]) == {"N", "P"}


def test_hierarchical_selector_cannot_lexically_starve_n_or_a4() -> None:
    rows = []
    for source, families in (
        ("E10B", [f"p{index}" for index in range(20)]),
        ("P_guided_original", ["prevalence_attenuation"]),
        ("N_mature", ["candidate_gradient"]),
        ("A4_exact", ["exact_peak"]),
    ):
        for family in families:
            rows.append({
                "query_index": 9,
                "action_id": f"{source}|{family}",
                "source": source,
                "family": family,
                "route": "corrective",
                "action_margin": 0.2,
                "margin_change": 0.1,
                "paired_advantage": 0.1,
            })
    output = select_diverse_routed_actions_v3(
        pd.DataFrame(rows),
        maximum_corrective_per_query=3,
        maximum_harmful_per_query=1,
        maximum_robust_per_query=1,
    )
    assert set(output.loc[output.selected_corrective, "source"]) == {
        "N_mature", "E10B", "A4_exact",
    }


def test_v4_direct_actions_share_n_mass_without_being_starved() -> None:
    rows = [
        {
            "query_index": 10,
            "action_id": f"{source}|{family}",
            "source": source,
            "family": family,
            "route": "corrective",
            "action_margin": 0.2,
            "margin_change": 0.1,
            "paired_advantage": 0.1,
        }
        for source, family in (
            ("N_mature", "candidate_gradient"),
            ("V4_gradient_path", "gradient_attenuation"),
            ("E10B", "recurrent_union_mix"),
            ("A4_exact", "exact_peak"),
        )
    ]
    output = select_diverse_routed_actions_v3(
        pd.DataFrame(rows),
        maximum_corrective_per_query=4,
        maximum_harmful_per_query=1,
        maximum_robust_per_query=1,
    )
    assert set(output.loc[output.selected_corrective, "source"]) == {
        "N_mature", "V4_gradient_path", "E10B", "A4_exact",
    }


def test_p_selector_balances_sources_then_families_before_recipe_multiplicity() -> None:
    rows = []
    for source, families in (
        ("E10B", ("recurrent_union_mix", "consensus_projection")),
        ("E11", ("recurrent_union_mix", "balanced_peak_exchange")),
        ("E12B", ("recurrent_union_mix", "transport_then_union")),
        ("P_guided_original", ("prevalence_attenuation", "matched_intensity_transport")),
    ):
        for family_index, family in enumerate(families):
            for copy in range(2):
                score = 0.20 - 0.01 * family_index - 0.001 * copy
                rows.append({
                    "query_index": 11,
                    "action_id": f"{source}|{family}|{copy}",
                    "source": source,
                    "family": family,
                    "route": "corrective",
                    "action_margin": score,
                    "margin_change": score,
                    "paired_advantage": score,
                })
    output = select_diverse_routed_actions_v3(
        pd.DataFrame(rows),
        maximum_corrective_per_query=5,
        maximum_harmful_per_query=1,
        maximum_robust_per_query=1,
    )
    selected = output.loc[output.selected_corrective]
    assert set(selected.source) == {
        "E10B", "E11", "E12B", "P_guided_original",
    }
    assert selected.family.nunique() >= 4
    assert len(selected) == 5


def test_shuffled_control_matches_stratum_and_never_keeps_same_query_action() -> None:
    actions = pd.DataFrame([
        {"query_index": query, "action_id": f"q{query}|{copy}", "source": "E10B",
         "family": "exchange", "recipe_id": "exchange|dose=0.5",
         "supervision_kind": "corrective", "action_tensor_index": index}
        for index, (query, copy) in enumerate([(1, 0), (1, 1), (2, 0), (3, 0)])
    ])
    bank = np.stack([
        np.full((2, 2), index + 1, dtype=np.float32) for index in range(len(actions))
    ])
    controls = -bank
    shuffled, report = source_family_shuffled_action_bank(
        actions, bank, controls, seed=11,
    )
    repeated, repeated_report = source_family_shuffled_action_bank(
        actions, bank, controls, seed=11,
    )
    assert np.array_equal(shuffled, repeated)
    assert (
        report["donor_tensor_index_sha256"]
        == repeated_report["donor_tensor_index_sha256"]
    )
    assert report["cross_query_rows"] == len(actions)
    query_by_value = {
        float(bank[index, 0, 0]): int(row.query_index)
        for index, row in actions.iterrows()
    }
    for index, row in actions.iterrows():
        donor_query = query_by_value[float(shuffled[index, 0, 0])]
        assert donor_query != int(row.query_index)
    assert len(set(shuffled[:, 0, 0].tolist())) == len(actions)
    assert report["cross_query_donor_rows_unique"] is True
    assert report["cross_query_donor_reuse_rows"] == 0


def test_imbalanced_stratum_uses_maximum_matching_without_donor_reuse() -> None:
    actions = pd.DataFrame([
        {
            "query_index": query,
            "action_id": f"q{query}|{copy}",
            "source": "E10B",
            "family": "exchange",
            "recipe_id": "exchange|dose=0.5",
            "supervision_kind": "corrective",
            "action_tensor_index": index,
        }
        for index, (query, copy) in enumerate(((1, 0), (1, 1), (1, 2), (2, 0)))
    ])
    bank = np.stack([
        np.full((2, 2), index + 1, dtype=np.float32)
        for index in range(len(actions))
    ])
    shuffled, report = source_family_shuffled_action_bank(
        actions, bank, -bank, seed=19,
    )
    # Multiplicity [3, 1] admits exactly two cross-query assignments.  The
    # other two majority-query recipients must use their own paired controls.
    cross_values = [float(value) for value in shuffled[:, 0, 0] if value > 0]
    fallback_values = [float(value) for value in shuffled[:, 0, 0] if value < 0]
    assert len(cross_values) == 2
    assert len(set(cross_values)) == 2
    assert len(fallback_values) == 2
    assert report["cross_query_rows"] == 2
    assert report["paired_control_fallback_rows"] == 2
    assert report["maximum_cross_query_cardinality_realized"] is True


def test_fallback_must_not_equal_targeted_action() -> None:
    actions = pd.DataFrame([{
        "query_index": 1,
        "action_id": "only",
        "source": "N",
        "family": "candidate",
        "recipe_id": "candidate|step=3|dose=0.5",
        "supervision_kind": "corrective",
        "action_tensor_index": 0,
    }])
    action = np.ones((1, 2, 2), dtype=np.float32)
    try:
        source_family_shuffled_action_bank(actions, action, action.copy(), seed=3)
    except RuntimeError as error:
        assert "fallback equals" in str(error)
    else:
        raise AssertionError("target-equal paired fallback was accepted")


def test_single_query_stratum_uses_paired_control_fallback() -> None:
    actions = pd.DataFrame([{
        "query_index": 1, "action_id": "only", "source": "N", "family": "candidate",
        "recipe_id": "candidate|step=3|dose=0.5", "supervision_kind": "corrective",
        "action_tensor_index": 0,
    }])
    action = np.ones((1, 2, 2), dtype=np.float32)
    control = -action
    shuffled, report = source_family_shuffled_action_bank(
        actions, action, control, seed=3,
    )
    assert np.array_equal(shuffled, control)
    assert report["paired_control_fallback_rows"] == 1


def test_shuffled_control_never_crosses_dose_recipe_or_supervision() -> None:
    actions = pd.DataFrame([
        {
            "query_index": query,
            "action_id": f"{kind}|{recipe}|q{query}",
            "source": "E10B",
            "family": "projection",
            "recipe_id": recipe,
            "supervision_kind": kind,
            "action_tensor_index": index,
        }
        for index, (kind, recipe, query) in enumerate((
            ("corrective", "dose=0.25", 1),
            ("corrective", "dose=0.25", 2),
            ("corrective", "dose=1.00", 1),
            ("corrective", "dose=1.00", 2),
            ("robust", "dose=0.25", 1),
            ("robust", "dose=0.25", 2),
        ))
    ])
    bank = np.stack([
        np.full((2, 2), index + 1, dtype=np.float32)
        for index in range(len(actions))
    ])
    shuffled, report = source_family_shuffled_action_bank(
        actions, bank, -bank, seed=41,
    )
    for _, block in actions.groupby(
        ["supervision_kind", "source", "family", "recipe_id"], sort=True,
    ):
        indices = block.action_tensor_index.to_numpy(np.int64)
        assert set(shuffled[indices, 0, 0]) == set(bank[indices, 0, 0])
    assert report["matching_columns"] == [
        "supervision_kind", "source", "family", "recipe_id",
    ]


def test_formal_a4_recipe_collision_is_split_by_family() -> None:
    actions = pd.DataFrame([
        {
            "query_index": query,
            "action_id": f"{family}|q{query}",
            "source": "A4_exact",
            "family": family,
            # Formal A4 recipe IDs encode token+dose but not role/rank family.
            "recipe_id": "token=137|dose=0.25",
            "supervision_kind": "corrective",
            "action_tensor_index": index,
        }
        for index, (family, query) in enumerate((
            ("exact_peak_shared|rank=1", 1),
            ("exact_peak_shared|rank=1", 2),
            ("exact_peak_unmatched|rank=1", 3),
            ("exact_peak_unmatched|rank=1", 4),
        ))
    ])
    bank = np.stack([
        np.full((2, 2), index + 1, dtype=np.float32)
        for index in range(len(actions))
    ])
    shuffled, report = source_family_shuffled_action_bank(
        actions, bank, -bank, seed=53,
    )
    family_by_value = {
        float(bank[index, 0, 0]): str(row.family)
        for index, row in actions.iterrows()
    }
    query_by_value = {
        float(bank[index, 0, 0]): int(row.query_index)
        for index, row in actions.iterrows()
    }
    for index, row in actions.iterrows():
        donor_value = float(shuffled[index, 0, 0])
        assert family_by_value[donor_value] == str(row.family)
        assert query_by_value[donor_value] != int(row.query_index)
    assert report["legacy_key_multi_family_strata"] == 1
    assert report["legacy_key_multi_family_rows"] == 4
    assert report["family_semantics_preserved"] is True


def main() -> None:
    tests = [
        test_candidate_boundary_records_exact_action_switch_rows,
        test_all_three_semantics_are_bounded_and_uncertain_is_audit_only,
        test_round_robin_preserves_source_family_diversity,
        test_hierarchical_selector_cannot_lexically_starve_n_or_a4,
        test_v4_direct_actions_share_n_mass_without_being_starved,
        test_p_selector_balances_sources_then_families_before_recipe_multiplicity,
        test_shuffled_control_matches_stratum_and_never_keeps_same_query_action,
        test_imbalanced_stratum_uses_maximum_matching_without_donor_reuse,
        test_fallback_must_not_equal_targeted_action,
        test_single_query_stratum_uses_paired_control_fallback,
        test_shuffled_control_never_crosses_dose_recipe_or_supervision,
        test_formal_a4_recipe_collision_is_split_by_family,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_action_routing_v3] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
