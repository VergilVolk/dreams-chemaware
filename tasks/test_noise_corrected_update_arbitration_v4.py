"""CPU contract tests for optimizer-space action restoration."""
from __future__ import annotations

import math

import torch

from noise_corrected_update_arbitration_v4 import (
    arbitrate_corrective_optimizer_updates,
    arbitrate_corrective_optimizer_updates_by_group,
    arbitrate_optimizer_updates,
    arbitrate_optimizer_updates_by_group,
    materialize_descent_updates_,
    reconcile_adamw_first_moments_to_materialized_updates_,
)
from train_noise_corrected_routed_direct import _virtual_adamw_descent_updates


def test_identity_when_action_fraction_is_already_sufficient() -> None:
    combined = [torch.tensor([1.0, 1.0])]
    risk = [torch.tensor([1.0, 0.0])]
    result = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.50,
    )
    assert result.action_gain == 1.0
    assert torch.equal(result.updates[0], combined[0])


def test_reaches_target_without_increasing_total_update_norm() -> None:
    combined = [torch.tensor([1.0, 0.2])]
    risk = [torch.tensor([1.0, 0.0])]
    result = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.40,
        maximum_action_gain=4.0,
    )
    assert result.target_reached
    assert abs(result.final_attributable_fraction - 0.40) < 1e-8
    assert abs(result.final_update_norm - result.original_update_norm) < 1e-6
    assert result.action_gain > 1


def test_conflicting_residual_is_projected_from_risk_direction() -> None:
    combined = [torch.tensor([0.8, 0.1])]
    risk = [torch.tensor([1.0, 0.0])]
    result = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.30,
        maximum_action_gain=4.0,
    )
    assert result.action_risk_cosine_before < 0
    assert abs(result.action_risk_cosine_after) < 1e-7
    assert result.target_reached
    assert abs(result.final_update_norm - result.original_update_norm) < 1e-6


def test_gain_cap_is_explicit_when_target_is_unreachable() -> None:
    combined = [torch.tensor([1.0, 0.01])]
    risk = [torch.tensor([1.0, 0.0])]
    result = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.80,
        maximum_action_gain=2.0,
    )
    assert result.gain_cap_hit
    assert not result.target_reached
    assert result.action_gain == 2.0


def test_none_layout_and_dtype_are_preserved() -> None:
    combined = [None, torch.tensor([1.0, 0.1], dtype=torch.float64)]
    risk = [None, torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.20,
        maximum_action_gain=4.0,
    )
    assert result.updates[0] is None
    assert result.updates[1].dtype == torch.float64


def test_restores_a_real_momentum_dominated_adamw_counterfactual() -> None:
    generator = torch.Generator().manual_seed(17)
    parameter = torch.nn.Parameter(torch.randn(4096, generator=generator))
    optimizer = torch.optim.AdamW(
        [parameter], lr=1e-4, weight_decay=1e-4, foreach=True,
    )
    risk_gradient = torch.ones_like(parameter) * 1e-3
    for _ in range(20):
        parameter.grad = risk_gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    action_gradient = torch.randn(4096, generator=generator) * 1e-3
    combined = _virtual_adamw_descent_updates(
        optimizer, [parameter], [risk_gradient + action_gradient],
    )
    risk = _virtual_adamw_descent_updates(
        optimizer, [parameter], [risk_gradient],
    )
    result = arbitrate_optimizer_updates(
        combined, risk,
        minimum_attributable_fraction=0.25,
        maximum_action_gain=4.0,
    )
    assert result.target_reached
    assert result.original_attributable_fraction < 0.10
    assert result.final_attributable_fraction >= 0.25 - 1e-10
    assert abs(result.final_update_norm - result.original_update_norm) < 1e-6


def test_metric_only_mode_avoids_materializing_full_update_tensors() -> None:
    combined = [torch.tensor([1.0, 0.2])]
    risk = [torch.tensor([1.0, 0.0])]
    full = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.40,
        maximum_action_gain=4.0,
    )
    metrics = arbitrate_optimizer_updates(
        combined, risk, minimum_attributable_fraction=0.40,
        maximum_action_gain=4.0, materialize_updates=False,
    )
    assert metrics.updates == []
    assert abs(metrics.action_gain - full.action_gain) < 1e-12
    assert abs(metrics.final_attributable_fraction - full.final_attributable_fraction) < 1e-12


def test_registered_risk_component_floor_is_explicit_and_respected() -> None:
    combined = [torch.tensor([1.0, 0.05], dtype=torch.float64)]
    risk = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = arbitrate_optimizer_updates(
        combined, risk,
        minimum_attributable_fraction=0.50,
        maximum_action_gain=20.0,
        minimum_risk_component_retention=0.90,
    )
    assert result.risk_constraint_active
    assert result.risk_component_retention >= 0.90 - 1e-9
    assert abs(result.final_update_norm - result.original_update_norm) < 1e-8


def test_unreachable_risk_floor_fails_closed_even_when_action_fraction_is_large() -> None:
    combined = [torch.tensor([0.5, 1.0], dtype=torch.float64)]
    risk = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = arbitrate_optimizer_updates(
        combined,
        risk,
        minimum_attributable_fraction=0.25,
        maximum_action_gain=4.0,
        minimum_risk_component_retention=0.90,
    )
    assert result.original_attributable_fraction >= 0.25
    assert result.risk_constraint_active
    assert result.risk_component_retention < 0.90
    assert not result.target_reached
    assert torch.equal(result.updates[0], combined[0])


def test_groupwise_restoration_prevents_head_from_hiding_backbone() -> None:
    combined = [
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        torch.tensor([1.0, 0.02], dtype=torch.float64),
    ]
    risk = [
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    ]
    result = arbitrate_optimizer_updates_by_group(
        combined, risk, {"head": [0], "backbone": [1]},
        minimum_attributable_fraction=0.10,
        maximum_action_gain=8.0,
        minimum_risk_component_retention=0.90,
    )
    assert result.parameter_groups["head"].target_reached
    assert result.parameter_groups["backbone"].target_reached
    assert result.parameter_groups["backbone"].final_attributable_fraction >= 0.10 - 1e-9
    assert result.minimum_group_risk_component_retention >= 0.90 - 1e-9
    assert abs(result.final_update_norm - result.original_update_norm) < 1e-8


def test_materialization_changes_parameters_but_not_adamw_state_progress() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=1e-4)
    gradient = torch.tensor([0.2, -0.1], dtype=torch.float64)
    parameter.grad = gradient.clone()
    combined = _virtual_adamw_descent_updates(optimizer, [parameter], [gradient])
    before = [parameter.detach().clone()]
    optimizer.step()
    state_step = int(optimizer.state[parameter]["step"].item())
    desired = [combined[0] * 0.5]
    materialize_descent_updates_([parameter], before, desired)
    assert state_step == 1
    assert torch.allclose(before[0] - parameter.detach(), desired[0], rtol=0, atol=1e-12)


def test_groupwise_restoration_survives_a_real_adamw_step() -> None:
    head = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
    backbone = torch.nn.Parameter(torch.tensor([0.5, -0.5], dtype=torch.float64))
    parameters = [head, backbone]
    optimizer = torch.optim.AdamW([
        {"params": [head], "lr": 1e-3, "group_name": "head"},
        {"params": [backbone], "lr": 2e-4, "group_name": "backbone"},
    ], weight_decay=1e-4)
    risk = [
        torch.tensor([0.20, 0.0], dtype=torch.float64),
        torch.tensor([0.10, 0.0], dtype=torch.float64),
    ]
    action = [
        torch.tensor([0.0, 0.02], dtype=torch.float64),
        torch.tensor([0.0, 0.01], dtype=torch.float64),
    ]
    combined_gradients = [left + right for left, right in zip(risk, action)]
    combined_updates = _virtual_adamw_descent_updates(
        optimizer, parameters, combined_gradients,
    )
    risk_updates = _virtual_adamw_descent_updates(
        optimizer, parameters, risk,
    )
    restored = arbitrate_optimizer_updates_by_group(
        combined_updates,
        risk_updates,
        {"head": [0], "backbone": [1]},
        minimum_attributable_fraction=0.10,
        maximum_action_gain=4.0,
        minimum_risk_component_retention=0.90,
    )
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter, gradient in zip(parameters, combined_gradients):
        parameter.grad = gradient
    optimizer.step()
    materialize_descent_updates_(parameters, before, restored.updates)
    for index, parameter in enumerate(parameters):
        assert torch.allclose(
            before[index] - parameter.detach(), restored.updates[index],
            rtol=0, atol=1e-12,
        )
        assert int(optimizer.state[parameter]["step"].item()) == 1
    assert restored.all_groups_target_reached
    assert restored.minimum_group_risk_component_retention >= 0.90 - 1e-9


def test_corrective_only_restoration_does_not_amplify_auxiliary_baseline() -> None:
    baseline = [torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)]
    protective = [torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)]
    combined = [torch.tensor([1.0, 0.5, 0.05], dtype=torch.float64)]
    result = arbitrate_corrective_optimizer_updates(
        combined,
        baseline,
        protective,
        minimum_attributable_fraction=0.20,
        maximum_corrective_gain=8.0,
        minimum_protective_component_retention=0.80,
    )
    assert result.target_reached
    assert result.action_gain > 1.0
    # The protective and auxiliary baseline coordinates receive the same global
    # norm rescale.  Their ratio therefore remains fixed; only z is amplified.
    assert abs(float(result.updates[0][1] / result.updates[0][0]) - 0.5) < 1e-12
    assert abs(result.final_update_norm - result.original_update_norm) < 1e-10
    assert result.risk_component_retention >= 0.80 - 1e-10


def test_corrective_only_zero_residual_falls_back_without_crashing() -> None:
    combined = [torch.tensor([1.0, 0.5], dtype=torch.float64)]
    result = arbitrate_corrective_optimizer_updates(
        combined,
        combined,
        [torch.tensor([1.0, 0.0], dtype=torch.float64)],
        minimum_attributable_fraction=0.20,
        maximum_corrective_gain=4.0,
        minimum_protective_component_retention=0.90,
    )
    assert not result.target_reached
    assert result.final_attributable_fraction == 0.0
    assert torch.equal(result.updates[0], combined[0])


def test_corrective_groupwise_restoration_and_adamw_moment_reconciliation() -> None:
    head = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
    backbone = torch.nn.Parameter(torch.tensor([0.5, -0.5], dtype=torch.float64))
    parameters = [head, backbone]
    optimizer = torch.optim.AdamW([
        {"params": [head], "lr": 1e-3, "group_name": "head"},
        {"params": [backbone], "lr": 2e-4, "group_name": "backbone"},
    ], weight_decay=1e-4)
    protective = [
        torch.tensor([0.20, 0.0], dtype=torch.float64),
        torch.tensor([0.10, 0.0], dtype=torch.float64),
    ]
    auxiliary = [
        torch.tensor([0.0, 0.01], dtype=torch.float64),
        torch.tensor([0.0, 0.005], dtype=torch.float64),
    ]
    corrective = [
        torch.tensor([0.01, 0.02], dtype=torch.float64),
        torch.tensor([0.005, 0.01], dtype=torch.float64),
    ]
    baseline_gradients = [
        left + right for left, right in zip(protective, auxiliary)
    ]
    combined_gradients = [
        left + right for left, right in zip(baseline_gradients, corrective)
    ]
    combined_updates = _virtual_adamw_descent_updates(
        optimizer, parameters, combined_gradients,
    )
    baseline_updates = _virtual_adamw_descent_updates(
        optimizer, parameters, baseline_gradients,
    )
    protective_updates = _virtual_adamw_descent_updates(
        optimizer, parameters, protective,
    )
    restored = arbitrate_corrective_optimizer_updates_by_group(
        combined_updates,
        baseline_updates,
        protective_updates,
        {"head": [0], "backbone": [1]},
        minimum_attributable_fraction=0.10,
        maximum_corrective_gain=4.0,
        minimum_protective_component_retention=0.90,
    )
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter, gradient in zip(parameters, combined_gradients):
        parameter.grad = gradient
    optimizer.step()
    materialize_descent_updates_(parameters, before, restored.updates)
    report = reconcile_adamw_first_moments_to_materialized_updates_(
        optimizer, parameters, before, restored.updates,
    )
    assert report["gate_passed"]
    assert report["same_step_reconstruction_relative_error"] <= 1e-6
    assert math.isfinite(report["materialized_update_relative_error"])
    assert math.isfinite(report["same_step_fp32_parameter_replay_relative_error"])
    assert report["reconstruction_target"] == (
        "realized_materialized_parameter_displacement"
    )
    assert report["verification_arithmetic"] == (
        "stable_decay_plus_adaptive_displacement"
    )
    assert report["second_moment_source"] == "actual_combined_gradient"
    for parameter in parameters:
        assert int(optimizer.state[parameter]["step"].item()) == 1
        assert torch.isfinite(optimizer.state[parameter]["exp_avg"]).all()
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for parameter in parameters:
        assert int(optimizer.state[parameter]["step"].item()) == 2
        assert torch.isfinite(parameter).all()


def test_fp32_reconciliation_does_not_fail_on_parameter_scale_cancellation() -> None:
    parameter = torch.nn.Parameter(
        torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "lr": 2e-6, "weight_decay": 0.0}]
    )
    before = [parameter.detach().clone()]
    parameter.grad = torch.linspace(0.01, 0.02, 4096, dtype=torch.float32)
    optimizer.step()
    requested = [torch.full_like(parameter, 1.5e-6)]
    materialize_descent_updates_([parameter], before, requested)
    report = reconcile_adamw_first_moments_to_materialized_updates_(
        optimizer, [parameter], before, requested,
    )
    assert report["gate_passed"]
    assert report["same_step_reconstruction_relative_error"] <= 1e-6
    assert report["reconstruction_target"] == (
        "realized_materialized_parameter_displacement"
    )
    assert math.isfinite(report["same_step_fp32_parameter_replay_relative_error"])


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_corrected_update_arbitration_v4] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
