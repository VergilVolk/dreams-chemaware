"""Golden integration tests for historical E4 gradients and Injector V1."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_action_injector_v1 import ActionInjectorV1, ActionInjectorV1Config
from noise_e4_action_injector_v1_bridge import (
    E4ActionInjectorV1Bridge,
    summarize_e4_action_injector_v1_steps,
)


def _case(*, warm: bool = True):
    head = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32))
    backbone = torch.nn.Parameter(torch.tensor([0.3, 1.2, -0.7], dtype=torch.float32))
    optimizer = torch.optim.AdamW([
        {
            "params": [head], "lr": 1e-3, "weight_decay": 0.01,
            "group_name": "head",
        },
        {
            "params": [backbone], "lr": 2e-4, "weight_decay": 0.02,
            "group_name": "backbone",
        },
    ])
    if warm:
        # Reproduce the mature warm-state condition after the first update.
        head.grad = torch.tensor([0.12, -0.05, 0.08])
        backbone.grad = torch.tensor([-0.07, 0.11, 0.03])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return [head, backbone], optimizer


def _assign(parameters, values):
    for parameter, value in zip(parameters, values):
        parameter.grad = value.detach().clone()


def _add(parameters, values):
    for parameter, value in zip(parameters, values):
        parameter.grad.add_(value)


def test_bridge_matches_direct_injector() -> None:
    action = [
        torch.tensor([0.04, -0.05, 0.02]),
        torch.tensor([-0.04, 0.06, 0.005]),
    ]
    protective = [
        torch.tensor([0.02, -0.03, 0.02]),
        torch.tensor([-0.01, 0.03, 0.015]),
    ]
    bridge_parameters, bridge_optimizer = _case()
    direct_parameters, direct_optimizer = _case()

    bridge = E4ActionInjectorV1Bridge(
        bridge_optimizer, bridge_parameters,
        verify_frozen_dependencies=False,
    )
    _assign(bridge_parameters, action)
    bridge.capture_corrective_()
    _add(bridge_parameters, protective)
    observed = bridge.step_and_inject_(maximum_gradient_norm=1.0)

    combined = [left + right for left, right in zip(action, protective)]
    combined_norm = torch.linalg.vector_norm(torch.cat(combined))
    clip = min(1.0, 1.0 / (float(combined_norm) + 1e-6))
    _assign(direct_parameters, [value * clip for value in combined])
    injector = ActionInjectorV1(
        ActionInjectorV1Config(0.25, 0.90, 1.50),
        verify_frozen_dependencies=False,
    )
    prepared = injector.prepare(
        direct_optimizer,
        direct_parameters,
        clipped_combined_gradients=[value * clip for value in combined],
        clipped_noncorrective_gradients=[value * clip for value in protective],
        clipped_protective_gradients=[value * clip for value in protective],
        parameter_group_positions={"head": [0], "backbone": [1]},
    )
    expected = injector.step_and_inject_(
        direct_optimizer, direct_parameters, prepared,
    )

    for left, right in zip(bridge_parameters, direct_parameters):
        if not torch.equal(left, right):
            raise AssertionError("E4 bridge changed the frozen V1 parameter result")
    for left, right in zip(bridge_parameters, direct_parameters):
        left_state = bridge_optimizer.state[left]
        right_state = direct_optimizer.state[right]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if not torch.equal(left_state[key], right_state[key]):
                raise AssertionError(f"E4 bridge changed AdamW state: {key}")
    if observed.maximum_fraction_abs_error > 2e-6:
        raise AssertionError("E4 bridge missed exact optimizer action dose")
    if min(observed.protective_component_retention_by_group.values()) + 1e-6 < 0.90:
        raise AssertionError("E4 bridge missed the protective floor")
    if not all(
        abs(value - 0.25) <= 2e-6
        for value in observed.optimizer_action_fraction_by_group.values()
    ):
        raise AssertionError("E4 bridge action fractions are not exact")
    if expected.semantic_contract_sha256 != bridge.injector.audit_manifest()[
        "semantic_contract_sha256"
    ]:
        raise AssertionError("E4 bridge did not use the frozen Injector V1 contract")

    summary = summarize_e4_action_injector_v1_steps([observed])
    if summary.get("gate_passed") is not True or summary.get("steps") != 1:
        raise AssertionError("E4 bridge exhaustive summary gate failed")


def test_bridge_rejects_missing_boundaries() -> None:
    parameters, optimizer = _case()
    bridge = E4ActionInjectorV1Bridge(
        optimizer, parameters, verify_frozen_dependencies=False,
    )
    try:
        bridge.step_and_inject_(maximum_gradient_norm=1.0)
    except RuntimeError as error:
        if "not captured" not in str(error):
            raise
    else:
        raise AssertionError("E4 bridge accepted a missing corrective capture")

    _assign(parameters, [torch.ones_like(value) for value in parameters])
    bridge.capture_corrective_()
    try:
        bridge.capture_corrective_()
    except RuntimeError as error:
        if "captured twice" not in str(error):
            raise
    else:
        raise AssertionError("E4 bridge accepted a duplicate corrective capture")


def test_bridge_materializes_the_cold_adamw_first_step() -> None:
    """The E8 weights are warm, but this E4 optimizer state starts empty."""
    # AdamW's first update is sign-like. A three-coordinate toy group can make
    # an exact 0.25 vector fraction geometrically infeasible even though the
    # real head/backbone groups contain millions of coordinates. Keep this
    # fixture high-dimensional enough to test the actual groupwise contract.
    torch.manual_seed(7)
    head = torch.nn.Parameter(torch.randn(256, dtype=torch.float32))
    backbone = torch.nn.Parameter(torch.randn(256, dtype=torch.float32))
    parameters = [head, backbone]
    optimizer = torch.optim.AdamW([
        {
            "params": [head], "lr": 1e-5, "weight_decay": 1e-4,
            "group_name": "head",
        },
        {
            "params": [backbone], "lr": 2e-6, "weight_decay": 0.0,
            "group_name": "backbone",
        },
    ])
    action = [torch.randn(256) * 0.002, torch.randn(256) * 0.002]
    protective = [torch.randn(256) * 0.003, torch.randn(256) * 0.003]
    bridge = E4ActionInjectorV1Bridge(
        optimizer, parameters, verify_frozen_dependencies=False,
    )
    _assign(parameters, action)
    bridge.capture_corrective_()
    _add(parameters, protective)
    observed = bridge.step_and_inject_(maximum_gradient_norm=1.0)
    if any(int(optimizer.state[value]["step"].item()) != 1 for value in parameters):
        raise AssertionError("E4 cold AdamW state did not advance exactly once")
    if observed.first_moment_reconstruction_relative_error > 1e-6:
        raise AssertionError("E4 cold AdamW first moment was not reconstructed")
    if summarize_e4_action_injector_v1_steps([observed]).get("gate_passed") is not True:
        raise AssertionError("E4 cold AdamW Injector V1 gate failed")


def main() -> None:
    tests = [
        test_bridge_matches_direct_injector,
        test_bridge_rejects_missing_boundaries,
        test_bridge_materializes_the_cold_adamw_first_step,
    ]
    for test in tests:
        test()
    print(f"[test_noise_e4_action_injector_v1_bridge] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
