"""CPU contracts for norm-neutral optimizer update rotation."""
from __future__ import annotations

import math

import torch

from noise_corrected_directional_update_v4 import rotate_update_toward_action


def test_reported_v3_like_alignment_needs_only_a_small_norm_neutral_rotation() -> None:
    cosine = 0.1380011700676673
    update = [torch.tensor([cosine, math.sqrt(1 - cosine**2)], dtype=torch.float64)]
    action = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = rotate_update_toward_action(
        update, action,
        minimum_action_alignment=0.25,
        maximum_action_coefficient=0.50,
    )
    assert result.target_reached
    assert abs(result.action_alignment_after - 0.25) < 1e-8
    assert 0 < result.action_coefficient < 0.20
    assert abs(result.update_norm_after - result.update_norm_before) < 1e-6


def test_nonconflicting_action_cannot_reduce_protective_alignment() -> None:
    update = [torch.tensor([0.2, 1.0], dtype=torch.float64)]
    action = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    risk = [torch.tensor([0.5, 1.0], dtype=torch.float64)]
    result = rotate_update_toward_action(
        update, action,
        risk_gradients=risk,
        minimum_action_alignment=0.40,
    )
    assert result.risk_alignment_after >= result.risk_alignment_before - 1e-8
    assert abs(result.update_norm_after - result.update_norm_before) < 1e-6


def test_conflicting_action_is_projected_before_rotation() -> None:
    update = [torch.tensor([0.0, 1.0], dtype=torch.float64)]
    action = [torch.tensor([-1.0, 1.0], dtype=torch.float64)]
    risk = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = rotate_update_toward_action(
        update, action,
        risk_gradients=risk,
        minimum_action_alignment=0.80,
    )
    assert result.target_reached
    assert result.risk_alignment_after >= result.risk_alignment_before - 1e-8


def test_identity_when_alignment_is_already_sufficient() -> None:
    update = [torch.tensor([1.0, 0.0])]
    action = [torch.tensor([1.0, 0.0])]
    result = rotate_update_toward_action(
        update, action, minimum_action_alignment=0.25,
    )
    assert result.action_coefficient == 0.0
    assert torch.equal(result.updates[0], update[0])


def test_risk_constraint_declines_an_unsafe_orthogonal_rotation() -> None:
    update = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    action = [torch.tensor([0.0, 1.0], dtype=torch.float64)]
    risk = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = rotate_update_toward_action(
        update, action,
        risk_gradients=risk,
        minimum_action_alignment=0.30,
    )
    assert result.risk_constraint_active
    assert not result.target_reached
    assert result.risk_alignment_after >= result.risk_alignment_before - 1e-7


def test_metric_only_mode_avoids_materializing_full_update_tensors() -> None:
    update = [torch.tensor([0.2, 1.0], dtype=torch.float64)]
    action = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    full = rotate_update_toward_action(
        update, action, minimum_action_alignment=0.40,
    )
    metrics = rotate_update_toward_action(
        update, action, minimum_action_alignment=0.40,
        materialize_updates=False,
    )
    assert metrics.updates == []
    assert abs(metrics.action_coefficient - full.action_coefficient) < 1e-12
    assert abs(metrics.action_alignment_after - full.action_alignment_after) < 1e-12


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_directional_update_v4] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
