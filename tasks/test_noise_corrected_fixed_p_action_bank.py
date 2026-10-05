"""Contract tests for the fixed P1 action bank."""
from __future__ import annotations

from build_noise_corrected_fixed_p_action_bank import RECIPE


def test_recipe_is_historical_fixed_p1() -> None:
    assert RECIPE == {
        "selector": "p_top3_transport_then_union",
        "family": "transport_then_union",
        "dose": 1.0,
        "auxiliary_dose": 0.5,
        "minimum_reference_prevalence": 0.67,
        "maximum_transferred_peaks": 5,
        "positive_references": 3,
    }


if __name__ == "__main__":
    test_recipe_is_historical_fixed_p1()
    print("PASS=1")
