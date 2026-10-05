"""Deterministic tests for bounded multi-peak noise actions."""
from __future__ import annotations

import numpy as np
import torch

from noise_corrected_action_expansion_v4 import (
    attenuate_tokens_and_renormalize,
    boost_tokens_and_renormalize,
    conservative_intensity_exchange,
    mass_matched_monotone_transfer_delta,
    mass_matched_monotone_transfer_by_group,
    rank_attenuation_tokens,
    rank_supported_boost_tokens,
    signed_multiplicative_action,
    smooth_monotone_transfer_delta,
)
from noise_v3_core import CONFOUNDER_ONLY, IDENTITY_ONLY, SHARED, UNMATCHED


def spectrum() -> torch.Tensor:
    return torch.tensor([
        [500.0, 0.25],
        [50.0, 1.00],
        [75.0, 0.80],
        [100.0, 0.50],
        [125.0, 0.20],
        [0.0, 0.0],
    ])


def test_rankers_obey_direction_and_peak_roles() -> None:
    clean = spectrum()
    gradient = np.asarray([0.0, -0.1, -0.8, 0.9, 0.7, 0.0])
    roles = np.asarray([-1, IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED, UNMATCHED, -1])
    assert rank_attenuation_tokens(clean, gradient, roles, 3).tolist() == [2]
    assert rank_supported_boost_tokens(clean, gradient, roles, 3).tolist() == [3]


def test_joint_actions_preserve_layout_and_normalization() -> None:
    clean = spectrum()
    outputs = [
        attenuate_tokens_and_renormalize(clean, [1, 2], 0.5),
        boost_tokens_and_renormalize(clean, [3, 4], 0.5),
        signed_multiplicative_action(
            clean, [1, 2], [3, 4], attenuation=0.5, boost=0.5,
        ),
        conservative_intensity_exchange(clean, [1, 2], [3, 4], attenuation=0.5),
    ]
    for output in outputs:
        assert torch.equal(output[0], clean[0])
        assert torch.equal(output[:, 0], clean[:, 0])
        assert torch.isclose(output[1:, 1].max(), torch.tensor(1.0))
        assert torch.all(output[1:, 1] >= 0)
    assert outputs[0][2, 1] / outputs[0][3, 1] < clean[2, 1] / clean[3, 1]
    assert outputs[1][3, 1] > clean[3, 1]


def test_exchange_conserves_pre_normalization_intensity_ratios() -> None:
    clean = spectrum()
    output = conservative_intensity_exchange(clean, [1, 2], [3, 4], attenuation=0.25)
    # Max-normalization can change the common scale, but relative total to the
    # unmodified transform is recoverable because the output maximum is one.
    raw = clean.clone()
    removed = 0.25 * (raw[1, 1] + raw[2, 1])
    raw[1, 1] *= 0.75
    raw[2, 1] *= 0.75
    weights = raw[[3, 4], 1] / raw[[3, 4], 1].sum()
    raw[[3, 4], 1] += removed * weights
    raw[1:, 1] /= raw[1:, 1].max()
    assert torch.allclose(output, raw)


def test_smooth_transfer_is_strictly_ordered_and_bounded() -> None:
    values = torch.tensor([0.0, 0.01, 0.10, 1.00])
    transformed = smooth_monotone_transfer_delta(values, scale=0.10, asymptote=0.20)
    assert transformed[0] == 0
    assert torch.all(transformed[1:] > transformed[:-1])
    assert float(transformed.max()) < 0.20


def test_mass_matched_transfer_removes_hard_cap_ties_without_more_dose() -> None:
    values = torch.tensor([0.0, 0.02, 0.10, 0.20, 0.50], requires_grad=True)
    active = torch.tensor([False, True, True, True, True])
    old = values.clamp_max(0.10) * active
    transformed = mass_matched_monotone_transfer_delta(
        values, active, hard_cap=0.10, maximum_cap_factor=2.0,
    )
    assert transformed[0] == 0
    assert torch.isclose(transformed.sum(), old.sum(), atol=1e-7)
    assert float(transformed.max()) <= 0.20
    assert torch.all(transformed[1:] > transformed[:-1])
    assert transformed.requires_grad is False


def test_grouped_transfer_preserves_every_source_family_cell() -> None:
    values = torch.tensor([
        [0.02, 0.20, 0.40],
        [0.01, 0.30, 0.50],
        [0.03, 0.10, 0.60],
    ])
    active = values > 0
    groups = ["P::E10B|union", "P::E10B|union", "N::N_mature|gradient"]
    transformed = mass_matched_monotone_transfer_by_group(
        values, active, groups, hard_cap=0.10,
    )
    old = values.clamp_max(0.10)
    assert torch.isclose(transformed[:2].sum(), old[:2].sum(), atol=1e-7)
    assert torch.isclose(transformed[2].sum(), old[2].sum(), atol=1e-7)
    assert transformed[0, 2] > transformed[0, 1]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_action_expansion_v4] PASS tests={len(tests)}", flush=True)
