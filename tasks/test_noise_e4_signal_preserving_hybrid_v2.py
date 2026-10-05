"""Golden tests for the E4-base plus later-action semantic bridge."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_e4_signal_preserving_hybrid_v2 import (  # noqa: E402
    E4SignalPreservingInjectorBridgeV2,
    bounded_semantic_microbatches,
    build_identity_equal_action_bag_plan,
    flatten_weighted_action_bags,
    query_local_e4_semantic_residual,
    summarize_signal_preserving_v2_steps,
)


def test_all_actions_enter_identity_equal_bags_without_multiplicity_dose() -> None:
    identities = ["A"] * 40 + ["B"] * 2 + ["C"] * 16
    action_ids = [f"a{index}" for index in range(len(identities))]
    policies = [f"p{index % 3}" for index in range(len(identities))]
    plan = build_identity_equal_action_bag_plan(
        action_ids,
        identities,
        policies,
        epochs=4,
        bags_per_identity_per_epoch=4,
        optimizer_steps_per_epoch=12,
        seed=17,
    )
    seen: set[int] = set()
    bag_counts: dict[tuple[str, int], int] = {}
    identity_effective: dict[tuple[str, int], float] = {}
    family_effective: dict[tuple[str, str, int], float] = {}
    for epoch, steps in enumerate(plan.epoch_steps):
        if len(steps) != 12:
            raise AssertionError("semantic plan changed the historical optimizer-step count")
        for bags in steps:
            indices, weights = flatten_weighted_action_bags(bags)
            if not 0 < sum(weights) <= 1.0 + 1e-12:
                raise AssertionError("step semantic weights are not bounded")
            seen.update(indices)
            cursor = 0
            for bag in bags:
                bag_counts[(bag.identity, epoch)] = bag_counts.get(
                    (bag.identity, epoch), 0,
                ) + 1
                identity_effective[(bag.identity, epoch)] = identity_effective.get(
                    (bag.identity, epoch), 0.0,
                ) + bag.effective_weight
                local = weights[cursor:cursor + len(bag.action_indices)]
                cursor += len(bag.action_indices)
                for index, weight in zip(bag.action_indices, local):
                    key = (bag.identity, policies[index], epoch)
                    family_effective[key] = family_effective.get(key, 0.0) + weight
                if not np.isclose(
                    sum(local), bag.effective_weight, rtol=0.0, atol=1e-12,
                ):
                    raise AssertionError("one identity bag lost its solved effective mass")
    if seen != set(range(len(action_ids))):
        raise AssertionError("complete later-action coverage failed")
    if set(bag_counts.values()) != {4}:
        raise AssertionError(f"identity epoch bag dose drifted: {bag_counts}")
    for epoch in range(4):
        values = [identity_effective[(identity, epoch)] for identity in ("A", "B", "C")]
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise AssertionError(f"identity effective dose differs in epoch {epoch}: {values}")
        for identity in ("A", "B", "C"):
            family_values = [
                value for (observed_identity, _, observed_epoch), value
                in family_effective.items()
                if observed_identity == identity and observed_epoch == epoch
            ]
            if not np.allclose(
                family_values, family_values[0], rtol=0.0, atol=1e-12,
            ):
                raise AssertionError(
                    f"source/family dose differs for {identity} in epoch {epoch}: "
                    f"{family_values}"
                )
    if plan.physical_action_exposures != 128:
        raise AssertionError("source/family-local coverage/fill count drifted")
    if plan.maximum_actions_per_bag != 3:
        raise AssertionError("large identity was not averaged into the fixed bag budget")
    if not np.isclose(
        plan.minimum_identity_effective_weight_per_epoch, 4.0,
        rtol=0.0, atol=0.0,
    ) or not np.isclose(
        plan.maximum_identity_effective_weight_per_epoch, 4.0,
        rtol=0.0, atol=0.0,
    ):
        raise AssertionError("identity semantic dose was equal but not four bags per epoch")


def test_corrected_geometry_has_four_distinct_steps_per_later_identity() -> None:
    identities = [f"identity_{index}" for index in range(1536)]
    plan = build_identity_equal_action_bag_plan(
        [f"action_{index}" for index in range(1536)],
        identities,
        ["later"] * 1536,
        epochs=1,
        bags_per_identity_per_epoch=4,
        optimizer_steps_per_epoch=7624,
        seed=31,
    )
    if plan.active_optimizer_steps_per_epoch != 6144:
        raise AssertionError("later semantic active-step count drifted")
    if plan.zero_semantic_optimizer_steps_per_epoch != 1480:
        raise AssertionError("corrected E4-only step count drifted")
    if sum(bool(step) for step in plan.epoch_steps[0]) != 6144:
        raise AssertionError("real schedule active mask drifted")
    if any(
        len(step) != 1 for step in plan.epoch_steps[0] if step
    ):
        raise AssertionError("multiple semantic bags were collapsed into one step")


def test_semantic_microbatching_preserves_weights_and_accumulated_gradient() -> None:
    action_indices = [0, 1, 2, 3, 4]
    action_weights = [0.10, 0.15, 0.25, 0.20, 0.30]
    spectra_per_action = [14, 28, 7, 21, 14]
    batches = bounded_semantic_microbatches(
        action_indices,
        action_weights,
        spectra_per_action,
        maximum_spectra_per_forward=42,
    )
    if any(sum(spectra_per_action[index] for index in indices) > 42
           for indices, _ in batches):
        raise AssertionError("semantic forward exceeded the spectra cap")
    parameter_full = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    losses = torch.tensor([0.3, -0.5, 1.2, 0.8, -0.9], dtype=torch.float64)
    sum(
        action_weights[index] * (parameter_full * losses[index]).square()
        for index in action_indices
    ).backward()
    parameter_split = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    for indices, weights in batches:
        sum(
            weight * (parameter_split * losses[index]).square()
            for index, weight in zip(indices, weights)
        ).backward()
    torch.testing.assert_close(
        parameter_split.grad, parameter_full.grad, rtol=0.0, atol=0.0,
    )


def test_query_local_semantic_residual_has_no_reference_gradient() -> None:
    torch.manual_seed(23)
    encoded = torch.nn.functional.normalize(
        torch.randn(8, 5, dtype=torch.float64), dim=1,
    ).detach().requires_grad_(True)
    layouts = [
        {"clean": 0, "action": 1, "positive": [2, 3], "negative": [4, 5]},
        {"clean": 6, "action": 7, "positive": [2, 3], "negative": [4, 5]},
    ]
    loss, report = query_local_e4_semantic_residual(
        encoded,
        layouts,
        [0.5, 0.5],
        rank_margin=0.05,
        temperature=0.10,
        lambda_clean_rank=1.0,
        lambda_aug_rank=1.0,
        lambda_consistency=0.25,
    )
    loss.backward()
    if report["semantic_reference_gradient_detached"] is not True:
        raise AssertionError("semantic residual did not certify reference detachment")
    torch.testing.assert_close(encoded.grad[2:6], torch.zeros_like(encoded.grad[2:6]))
    if encoded.grad[[0, 1, 6, 7]].norm().item() <= 0:
        raise AssertionError("clean/action semantic paths are not live")
    if not 0.0 <= report["semantic_action_margin_pass_fraction"] <= 1.0:
        raise AssertionError("semantic action margin pass is not a fraction")


def test_semantic_residual_targets_clean_boundary_and_gates_satisfied_action() -> None:
    encoded = torch.tensor([
        [0.0, 1.0],   # clean: wrong against the selected boundary
        [1.0, 0.0],   # action: robustly correct
        [1.0, 0.0],   # positive reference
        [-1.0, 0.0],  # negative reference
    ], dtype=torch.float64, requires_grad=True)
    loss, report = query_local_e4_semantic_residual(
        encoded,
        [{"clean": 0, "action": 1, "positive": [2], "negative": [3]}],
        [1.0],
        rank_margin=0.05,
        temperature=0.10,
        lambda_clean_rank=1.0,
        lambda_aug_rank=1.0,
        lambda_consistency=0.0,
    )
    loss.backward()
    if report["semantic_clean_boundary_rank_active"] is not True:
        raise AssertionError("later residual omitted the clean molecule boundary")
    if report["semantic_satisfied_action_rank_gradient_gated"] is not True:
        raise AssertionError("later residual did not gate an already-satisfied action")
    if report["semantic_action_rank_active_fraction"] != 0.0:
        raise AssertionError("robust action retained rank pressure past the E4 margin")
    if encoded.grad[0].norm().item() <= 0:
        raise AssertionError("clean boundary/transfer gradient did not reach the clean view")
    torch.testing.assert_close(encoded.grad[1], torch.zeros_like(encoded.grad[1]))
    torch.testing.assert_close(encoded.grad[2:], torch.zeros_like(encoded.grad[2:]))


def test_symmetric_action_content_changes_the_clean_consistency_direction() -> None:
    def clean_gradient(action: list[float]) -> torch.Tensor:
        encoded = torch.tensor([
            [0.0, 1.0], action, [1.0, 0.0], [-1.0, 0.0],
        ], dtype=torch.float64, requires_grad=True)
        loss, _ = query_local_e4_semantic_residual(
            encoded,
            [{"clean": 0, "action": 1, "positive": [2], "negative": [3]}],
            [1.0],
            rank_margin=0.05,
            temperature=0.10,
            lambda_clean_rank=1.0,
            lambda_aug_rank=0.0,
            lambda_consistency=0.25,
        )
        loss.backward()
        return encoded.grad[0].detach().clone()

    targeted = clean_gradient([1.0, 0.0])
    shuffled = clean_gradient([-1.0, 0.0])
    if torch.equal(targeted, shuffled):
        raise AssertionError("targeted/shuffled action content does not reach clean gradient")


def _run_bridge_step() -> tuple[
    list[torch.Tensor], list[torch.Tensor], object, dict[str, object]
]:
    torch.manual_seed(29)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 1e-3, "weight_decay": 1e-4,
         "group_name": "head"},
        {"params": [parameters[1]], "lr": 2e-4, "weight_decay": 0.0,
         "group_name": "backbone"},
    ])
    bridge = E4SignalPreservingInjectorBridgeV2(
        optimizer,
        parameters,
        target_attributable_fraction=0.25,
        minimum_historical_e4_retention=0.90,
        maximum_historical_e4_update_norm_ratio=1.50,
        verify_frozen_dependencies=False,
    )
    before = [parameter.detach().clone() for parameter in parameters]
    x = torch.tensor([[1.0, -0.5, 0.25], [-0.2, 0.8, 0.4]], dtype=torch.float64)
    semantic = model(x)[0].square().sum() + 0.3 * model(x)[1].sum()
    semantic.backward()
    bridge.capture_semantic_corrective_()
    optimizer.zero_grad(set_to_none=True)
    historical = (model(x) - torch.tensor([[0.2, -0.1], [0.1, 0.4]])).square().mean()
    historical.backward()
    step = bridge.step_and_inject_(maximum_gradient_norm=1.0)
    after = [parameter.detach().clone() for parameter in parameters]
    return before, after, step, bridge.audit_manifest()


def test_bridge_preserves_absolute_e4_update_and_exact_action_fraction() -> None:
    before, after, step, manifest = _run_bridge_step()
    if not any(not torch.equal(left, right) for left, right in zip(before, after)):
        raise AssertionError("signal-preserving bridge did not update model parameters")
    for value in step.optimizer_action_fraction_by_group.values():
        if not np.isclose(value, 0.25, rtol=0.0, atol=2e-6):
            raise AssertionError(f"optimizer action fraction drifted: {value}")
    if min(step.historical_e4_component_retention_by_group.values()) < 0.90 - 1e-6:
        raise AssertionError("historical E4 protective component was not retained")
    if step.final_to_historical_e4_update_norm_ratio < 0.90 - 1e-6:
        raise AssertionError("absolute E4 update collapsed")
    if manifest["historical_e4_baseline_is_protective_axis"] is not True:
        raise AssertionError("bridge manifest misdeclared its protective axis")
    summary = summarize_signal_preserving_v2_steps([step])
    if summary["gate_passed"] is not True:
        raise AssertionError(f"valid signal-preserving step failed: {summary}")


def test_bridge_is_deterministic_from_same_adamw_state() -> None:
    first = _run_bridge_step()
    second = _run_bridge_step()
    for left, right in zip(first[1], second[1]):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    if first[2] != second[2]:
        raise AssertionError("signal-preserving bridge receipt is not deterministic")


def test_zero_semantic_is_bitwise_historical_e4_for_64_adamw_steps() -> None:
    torch.manual_seed(41)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    oracle = copy.deepcopy(model)
    parameters = list(model.parameters())
    oracle_parameters = list(oracle.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [parameters[0]], "lr": 1e-3, "weight_decay": 1e-4,
         "group_name": "head"},
        {"params": [parameters[1]], "lr": 2e-4, "weight_decay": 0.0,
         "group_name": "backbone"},
    ])
    oracle_optimizer = torch.optim.AdamW([
        {"params": [oracle_parameters[0]], "lr": 1e-3, "weight_decay": 1e-4,
         "group_name": "head"},
        {"params": [oracle_parameters[1]], "lr": 2e-4, "weight_decay": 0.0,
         "group_name": "backbone"},
    ])
    bridge = E4SignalPreservingInjectorBridgeV2(
        optimizer, parameters, verify_frozen_dependencies=False,
    )
    for step_index in range(64):
        x = torch.tensor([
            [1.0 + 0.01 * step_index, -0.5, 0.25],
            [-0.2, 0.8 - 0.005 * step_index, 0.4],
        ], dtype=torch.float64)
        scale = 100.0 if step_index % 2 == 0 else 0.01

        optimizer.zero_grad(set_to_none=True)
        (model(x).sum() * 0.0).backward()
        bridge.capture_semantic_corrective_(allow_zero=True)
        optimizer.zero_grad(set_to_none=True)
        (scale * model(x).square().mean()).backward()

        oracle_optimizer.zero_grad(set_to_none=True)
        (scale * oracle(x).square().mean()).backward()
        torch.nn.utils.clip_grad_norm_(oracle_parameters, 1.0)
        oracle_optimizer.step()

        receipt = bridge.step_and_inject_(maximum_gradient_norm=1.0)
        if receipt.zero_semantic_exact_e4_equivalence is not True:
            raise AssertionError(f"zero semantic diverged internally at step {step_index}")
        for observed, expected in zip(parameters, oracle_parameters):
            if not torch.equal(observed, expected):
                raise AssertionError(f"zero semantic changed E4 parameters at step {step_index}")
        for position, (observed, expected) in enumerate(zip(parameters, oracle_parameters)):
            left_state = optimizer.state[observed]
            right_state = oracle_optimizer.state[expected]
            shadow_state = bridge.shadow_optimizer.state[
                bridge.shadow_parameters[position]
            ]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                if not torch.equal(left_state[key], right_state[key]):
                    raise AssertionError(
                        f"zero semantic changed AdamW {key} at step {step_index}"
                    )
                if not torch.equal(shadow_state[key], right_state[key]):
                    raise AssertionError(
                        f"shadow E4 {key} diverged from oracle at step {step_index}"
                    )


def test_alternating_nonzero_zero_steps_preserve_independent_shadow_for_64_steps() -> None:
    torch.manual_seed(43)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).double()
    parameters = list(model.parameters())
    unused = torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
    parameters_with_unused = [parameters[0], unused, parameters[1]]
    optimizer = torch.optim.AdamW([
        {"params": [parameters[0], unused], "lr": 1e-3, "weight_decay": 1e-4,
         "group_name": "head"},
        {"params": [parameters[1]], "lr": 2e-4, "weight_decay": 0.0,
         "group_name": "backbone"},
    ])
    bridge = E4SignalPreservingInjectorBridgeV2(
        optimizer,
        parameters_with_unused,
        target_attributable_fraction=0.25,
        minimum_historical_e4_retention=0.90,
        maximum_historical_e4_update_norm_ratio=1.50,
        verify_frozen_dependencies=False,
    )
    oracle_parameters = [
        torch.nn.Parameter(value.detach().clone(), requires_grad=False)
        for value in parameters_with_unused
    ]
    oracle_optimizer = torch.optim.AdamW([
        {"params": [oracle_parameters[0], oracle_parameters[1]],
         "lr": 1e-3, "weight_decay": 1e-4, "group_name": "head"},
        {"params": [oracle_parameters[2]], "lr": 2e-4,
         "weight_decay": 0.0, "group_name": "backbone"},
    ])
    receipts = []
    for step_index in range(64):
        x = torch.tensor([
            [1.0 + 0.01 * step_index, -0.5, 0.25],
            [-0.2, 0.8 - 0.005 * step_index, 0.4],
        ], dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        if step_index % 2 == 0:
            sign = 1.0 if step_index % 4 == 0 else -1.0
            semantic = sign * 1000.0 * (
                model(x)[0].square().sum() + 0.1 * model(x)[1].sum()
            )
            semantic.backward()
            bridge.capture_semantic_corrective_()
        else:
            (model(x).sum() * 0.0).backward()
            bridge.capture_semantic_corrective_(allow_zero=True)

        optimizer.zero_grad(set_to_none=True)
        ((1.0 + step_index % 5) * model(x).square().mean()).backward()
        historical_raw = [
            None if value.grad is None else value.grad.detach().clone()
            for value in parameters_with_unused
        ]
        for live, oracle_parameter, gradient in zip(
            parameters_with_unused, oracle_parameters, historical_raw
        ):
            oracle_parameter.data.copy_(live.detach())
            oracle_parameter.grad = (
                None if gradient is None else gradient.detach().clone()
            )
        torch.nn.utils.clip_grad_norm_(oracle_parameters, 1.0)
        oracle_optimizer.step()
        oracle_optimizer.zero_grad(set_to_none=True)

        receipt = bridge.step_and_inject_(maximum_gradient_norm=1.0)
        receipts.append(receipt)
        for position, (observed, expected) in enumerate(zip(
            bridge.shadow_parameters, oracle_parameters
        )):
            if not torch.equal(observed, expected):
                raise AssertionError(
                    f"independent shadow parameter diverged at {step_index}/{position}"
                )
            left_state = bridge.shadow_optimizer.state.get(observed, {})
            right_state = oracle_optimizer.state.get(expected, {})
            if set(left_state) != set(right_state):
                raise AssertionError("unused-parameter shadow state layout drifted")
            for key in left_state:
                left = left_state[key]
                right = right_state[key]
                if torch.is_tensor(left) and not torch.equal(left, right):
                    raise AssertionError(
                        f"shadow {key} diverged at {step_index}/{position}"
                    )
        if min(receipt.final_to_shadow_e4_update_norm_ratio_by_group.values()) < 0.90 - 1e-6:
            raise AssertionError("adversarial sequence collapsed absolute E4 update")
        if min(receipt.historical_e4_component_retention_by_group.values()) < 0.90 - 1e-6:
            raise AssertionError("adversarial sequence consumed the E4 axis")
        if receipt.semantic_active:
            if any(abs(value - 0.25) > 2e-6
                   for value in receipt.optimizer_action_fraction_by_group.values()):
                raise AssertionError("active semantic step missed exact 25% attribution")
        elif receipt.zero_semantic_shadow_update_materialized is not True:
            raise AssertionError("mixed-history zero step reused contaminated AdamW")
    summary = summarize_signal_preserving_v2_steps(receipts)
    if summary["gate_passed"] is not True:
        raise AssertionError(f"adversarial 64-step summary failed: {summary}")
    json.dumps(summary)


def test_fp32_bridge_survives_alternating_clipped_history() -> None:
    """Exercise the exact production dtype that broke restored AdamW in V7."""
    torch.manual_seed(47)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 7, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(7, 3, bias=False),
    ).float()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": parameters[:2], "lr": 1e-5, "weight_decay": 1e-4,
         "group_name": "head"},
        {"params": parameters[2:], "lr": 2e-6, "weight_decay": 0.0,
         "group_name": "backbone"},
    ])
    bridge = E4SignalPreservingInjectorBridgeV2(
        optimizer, parameters, verify_frozen_dependencies=False,
    )
    receipts = []
    for step_index in range(128):
        generator = torch.Generator().manual_seed(1000 + step_index)
        x = torch.randn(6, 5, generator=generator, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        if step_index % 3:
            semantic_scale = 500.0 if step_index % 2 else 0.02
            semantic = semantic_scale * (
                model(x)[0].square().sum() + 0.2 * model(x)[1].sum()
            )
            semantic.backward()
            bridge.capture_semantic_corrective_()
        else:
            (model(x).sum() * 0.0).backward()
            bridge.capture_semantic_corrective_(allow_zero=True)
        optimizer.zero_grad(set_to_none=True)
        historical_scale = 100.0 if step_index % 4 == 0 else 0.05
        (historical_scale * model(x).square().mean()).backward()
        receipts.append(bridge.step_and_inject_(maximum_gradient_norm=1.0))
    summary = summarize_signal_preserving_v2_steps(receipts)
    if summary["gate_passed"] is not True:
        raise AssertionError(f"fp32 bridge stress gate failed: {summary}")
    json.dumps(summary)


def main() -> None:
    tests = (
        test_all_actions_enter_identity_equal_bags_without_multiplicity_dose,
        test_corrected_geometry_has_four_distinct_steps_per_later_identity,
        test_semantic_microbatching_preserves_weights_and_accumulated_gradient,
        test_query_local_semantic_residual_has_no_reference_gradient,
        test_semantic_residual_targets_clean_boundary_and_gates_satisfied_action,
        test_symmetric_action_content_changes_the_clean_consistency_direction,
        test_bridge_preserves_absolute_e4_update_and_exact_action_fraction,
        test_bridge_is_deterministic_from_same_adamw_state,
        test_zero_semantic_is_bitwise_historical_e4_for_64_adamw_steps,
        test_alternating_nonzero_zero_steps_preserve_independent_shadow_for_64_steps,
        test_fp32_bridge_survives_alternating_clipped_history,
    )
    for test in tests:
        test()
    print(f"[test_noise_e4_signal_preserving_hybrid_v2] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
