"""CPU tests for minimal-dose direct noise actions."""
from __future__ import annotations

import numpy as np
import torch

from noise_corrected_trust_region_action_v4 import (
    apply_trust_region_action,
    minimal_supported_margin_action,
)
from noise_v3_core import CONFOUNDER_ONLY, IDENTITY_ONLY, SHARED


def fixture() -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    clean = torch.tensor([
        [500.0, 1.0],
        [100.0, 1.0],
        [120.0, 0.8],
        [140.0, 0.5],
        [160.0, 0.4],
        [0.0, 0.0],
    ])
    gradient = np.asarray([0.0, -0.20, -0.10, 0.30, 0.10, 99.0])
    roles = np.asarray([SHARED, CONFOUNDER_ONLY, SHARED, IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED])
    return clean, gradient, roles


def test_minimal_dose_reaches_requested_first_order_gain() -> None:
    clean, gradient, roles = fixture()
    action = minimal_supported_margin_action(
        clean, gradient, roles, target_gain=0.20,
        maximum_fraction_per_peak=0.75, maximum_total_fraction=2.0,
    )
    assert action.target_reached
    assert abs(action.predicted_gain - 0.20) < 1e-8
    assert action.total_fractional_dose < 2.0


def test_identity_peak_is_never_attenuated_and_confounder_is_never_boosted() -> None:
    clean, gradient, roles = fixture()
    action = minimal_supported_margin_action(clean, gradient, roles, target_gain=1.0)
    assert all(roles[token] != IDENTITY_ONLY for token in action.attenuation_tokens)
    assert all(roles[token] in {IDENTITY_ONLY, SHARED} for token in action.boost_tokens)


def test_action_preserves_precursor_and_normalization() -> None:
    clean, gradient, roles = fixture()
    action = minimal_supported_margin_action(clean, gradient, roles, target_gain=0.20)
    output = apply_trust_region_action(clean, action)
    assert torch.equal(output[0], clean[0])
    assert float(output[1:, 1].max()) == 1.0
    assert torch.isfinite(output).all()


def test_unreachable_target_is_reported_without_exceeding_budget() -> None:
    clean, gradient, roles = fixture()
    action = minimal_supported_margin_action(
        clean, gradient, roles, target_gain=10.0,
        maximum_peaks=2, maximum_fraction_per_peak=0.25,
        maximum_total_fraction=0.40,
    )
    assert not action.target_reached
    assert action.total_fractional_dose <= 0.40 + 1e-12
    assert len(action.attenuation_tokens) + len(action.boost_tokens) <= 2


def test_direction_restriction_produces_isolated_actions() -> None:
    clean, gradient, roles = fixture()
    down = minimal_supported_margin_action(
        clean, gradient, roles, target_gain=0.10,
        directions=frozenset({"down"}),
    )
    up = minimal_supported_margin_action(
        clean, gradient, roles, target_gain=0.10,
        directions=frozenset({"up"}),
    )
    assert down.attenuation_tokens and not down.boost_tokens
    assert up.boost_tokens and not up.attenuation_tokens


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_trust_region_action_v4] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
