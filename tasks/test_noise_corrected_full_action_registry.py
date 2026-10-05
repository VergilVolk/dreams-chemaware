from __future__ import annotations

from collections import Counter

from noise_corrected_full_action_registry import (
    P_MISSING_PEAK_FAMILIES,
    registered_p_recipes,
)


def main() -> None:
    recipes = registered_p_recipes()
    assert len(recipes) == 66
    assert len({item.recipe_id for item in recipes}) == 66
    assert Counter(item.source for item in recipes) == {
        "P_guided_original": 6, "E10B": 19, "E11": 16, "E12B": 25,
    }
    assert {item.reference_policy for item in recipes} == {
        "top3", "farthest3", "maxmin6", "condition6", "maxmin12",
    }
    assert {item.family for item in recipes} >= {
        "consensus_projection", "prevalence_attenuation", "recurrent_union_mix",
        "matched_intensity_transport",
        "recurrent_peak_graft", "balanced_peak_exchange", "transport_then_union",
        "consensus_then_union",
    }
    original_intensity = {
        (item.family, item.dose)
        for item in recipes
        if item.reference_policy == "top3"
        and item.family in {
            "matched_intensity_transport", "prevalence_attenuation", "consensus_projection",
        }
    }
    assert original_intensity == {
        (family, dose)
        for family in (
            "matched_intensity_transport", "prevalence_attenuation", "consensus_projection",
        )
        for dose in (0.25, 0.50, 0.75, 1.00)
    }
    missing_context_keys = {
        (
            item.reference_policy,
            item.minimum_reference_prevalence,
            item.maximum_transferred_peaks,
        )
        for item in recipes
        if item.family in P_MISSING_PEAK_FAMILIES
    }
    # Five reference policies each use the same three recurrence settings.
    # Direction is deliberately outside this set and doubles the runtime cache.
    assert len(missing_context_keys) == 15
    print("PASS=1")


if __name__ == "__main__":
    main()
