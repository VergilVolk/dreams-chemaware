"""Numerical-free contracts for the frozen direct best-action inventory."""
from __future__ import annotations

from noise_corrected_best_action_v5 import (
    V4_SOURCE,
    action_lineage_contract,
    registered_v4_action_recipes,
    select_registered_v4_views,
)


def test_registry_is_exact_and_excludes_failed_redundant_recipes() -> None:
    recipes = registered_v4_action_recipes()
    names = {recipe.name for recipe in recipes}
    assert len(recipes) == 9
    assert {
        "quad_attenuation_4x25", "quad_attenuation_4x50",
        "supported_boost_2x50", "adaptive_trust_up",
        "adaptive_trust_down", "adaptive_strong_down",
        "sequential_supported_boost_1x",
        "sequential_supported_boost_2x",
        "sequential_supported_boost_4x",
    } == names
    assert not names.intersection({
        "single_attenuation_1x50", "dual_attenuation_2x50",
        "signed_multiplicative_2d2u_50", "conservative_exchange_2d2u_50",
        "adaptive_strong_up", "adaptive_strong_joint",
        "sequential_supported_boost_3x", "sequential_supported_boost_5x",
        "sequential_supported_boost_6x",
    })
    assert all(recipe.recipe_id.startswith(f"{V4_SOURCE}|") for recipe in recipes)


def test_view_filter_adds_frozen_semantics_without_reordering() -> None:
    views = [
        {"recipe": "single_attenuation_1x50"},
        {"recipe": "supported_boost_2x50", "payload": 1},
        {"recipe": "quad_attenuation_4x25", "payload": 2},
    ]
    selected = select_registered_v4_views(views)
    assert [item["recipe"] for item in selected] == [
        "supported_boost_2x50", "quad_attenuation_4x25",
    ]
    assert [item["payload"] for item in selected] == [1, 2]
    assert all(item["family"] and item["evidence_role"] for item in selected)


def test_lineage_does_not_invent_e4_e8_e13_actions() -> None:
    lineage = action_lineage_contract()
    assert lineage["E4"].endswith("not_action_bank")
    assert lineage["E8"].endswith("not_action_bank")
    assert lineage["E13"].endswith("not_new_action_bank")
    assert lineage["teacher_embedding_or_distillation_target_used"] is False
    assert V4_SOURCE in lineage["mature_executable_sources"]


def main() -> None:
    tests = [
        test_registry_is_exact_and_excludes_failed_redundant_recipes,
        test_view_filter_adds_frozen_semantics_without_reordering,
        test_lineage_does_not_invent_e4_e8_e13_actions,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_best_action_v5] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
