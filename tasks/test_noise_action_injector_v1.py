"""CPU-only golden and lifecycle tests for the frozen ActionInjectorV1."""
from __future__ import annotations

import inspect
from pathlib import Path

import torch

from noise_action_injector_v1 import (
    ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256,
    ActionInjectorV1,
    ActionInjectorV1Config,
    action_injector_v1_contract_manifest,
    validate_action_injector_v1_frozen_dependencies,
    virtual_adamw_descent_updates_v1,
)
from noise_corrected_update_arbitration_v4 import (
    materialize_descent_updates_,
    reconcile_adamw_first_moments_to_materialized_updates_,
)
from noise_corrected_update_arbitration_v10 import (
    compose_safe_exact_corrective_updates_by_group,
)
from train_noise_corrected_routed_direct import (
    _virtual_adamw_descent_updates as legacy_virtual_adamw_descent_updates,
)


EXPECTED_SEMANTIC_CONTRACT_SHA256 = (
    "e7a742a9a8a695a439d2437f502e52a77c9f6bf8e82c793c6e618fd097bce579"
)


def _optimizer_fixture() -> tuple[
    list[torch.nn.Parameter], torch.optim.AdamW,
]:
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5])),
        torch.nn.Parameter(torch.tensor([0.3, 1.2, -0.7])),
    ]
    optimizer = torch.optim.AdamW([
        {
            "params": [parameters[0]], "lr": 1e-3, "weight_decay": 0.01,
            "group_name": "head",
        },
        {
            "params": [parameters[1]], "lr": 2e-4, "weight_decay": 0.02,
            "group_name": "backbone",
        },
    ], betas=(0.9, 0.999), eps=1e-8)
    # Populate unequal first/second moments before the compared injection step.
    parameters[0].grad = torch.tensor([0.12, -0.05, 0.08])
    parameters[1].grad = torch.tensor([-0.07, 0.11, 0.03])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return parameters, optimizer


def _gradient_ledgers() -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
]:
    combined = [
        torch.tensor([0.06, -0.08, 0.04]),
        torch.tensor([-0.05, 0.09, 0.02]),
    ]
    noncorrective = [
        torch.tensor([0.03, -0.04, 0.01]),
        torch.tensor([-0.02, 0.04, 0.01]),
    ]
    protective = [
        torch.tensor([0.02, -0.03, 0.02]),
        torch.tensor([-0.01, 0.03, 0.015]),
    ]
    return combined, noncorrective, protective


def _assert_ledger_equal(
    left: list[torch.Tensor | None], right: list[torch.Tensor | None],
) -> None:
    assert len(left) == len(right)
    for one, two in zip(left, right):
        assert (one is None) == (two is None)
        if one is not None:
            assert two is not None and torch.equal(one, two)


def _optimizer_state_snapshot(
    optimizer: torch.optim.AdamW,
    parameters: list[torch.nn.Parameter],
) -> list[dict[str, torch.Tensor | int]]:
    output = []
    for parameter in parameters:
        values = {}
        for name, value in optimizer.state[parameter].items():
            values[name] = value.detach().clone() if torch.is_tensor(value) else value
        output.append(values)
    return output


def test_contract_and_numerical_dependencies_are_frozen() -> None:
    assert ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256 == (
        EXPECTED_SEMANTIC_CONTRACT_SHA256
    )
    manifest = action_injector_v1_contract_manifest()
    assert manifest["semantic_contract_sha256"] == EXPECTED_SEMANTIC_CONTRACT_SHA256
    assert manifest["action_semantics"] == "opaque_upstream_any_action_family"
    observed = validate_action_injector_v1_frozen_dependencies()
    assert observed == manifest["frozen_dependency_sha256"]


def test_public_prepare_boundary_is_action_type_agnostic() -> None:
    parameters = set(inspect.signature(ActionInjectorV1.prepare).parameters)
    assert parameters == {
        "self", "optimizer", "parameters", "clipped_combined_gradients",
        "clipped_noncorrective_gradients", "clipped_protective_gradients",
        "parameter_group_positions",
    }


def test_v1_virtual_adamw_is_tensor_exact_to_v10_trainer_oracle() -> None:
    parameters, optimizer = _optimizer_fixture()
    combined, _, _ = _gradient_ledgers()
    expected = legacy_virtual_adamw_descent_updates(
        optimizer, parameters, combined,
    )
    observed = virtual_adamw_descent_updates_v1(
        optimizer, parameters, combined,
    )
    _assert_ledger_equal(observed, expected)
    assert all(parameter.grad is None for parameter in parameters)


def test_complete_injection_is_tensor_exact_to_v10_legacy_sequence() -> None:
    legacy_parameters, legacy_optimizer = _optimizer_fixture()
    v1_parameters, v1_optimizer = _optimizer_fixture()
    combined, noncorrective, protective = _gradient_ledgers()
    groups = {"head": [0], "backbone": [1]}

    legacy_combined = legacy_virtual_adamw_descent_updates(
        legacy_optimizer, legacy_parameters, combined,
    )
    legacy_noncorrective = legacy_virtual_adamw_descent_updates(
        legacy_optimizer, legacy_parameters, noncorrective,
    )
    legacy_protective = legacy_virtual_adamw_descent_updates(
        legacy_optimizer, legacy_parameters, protective,
    )
    legacy_composition = compose_safe_exact_corrective_updates_by_group(
        legacy_combined,
        legacy_noncorrective,
        legacy_protective,
        groups,
        target_attributable_fraction=0.25,
        minimum_protective_component_retention=0.90,
        materialize_updates=True,
    )
    legacy_before = [parameter.detach().clone() for parameter in legacy_parameters]
    for parameter, gradient in zip(legacy_parameters, combined):
        parameter.grad = gradient.clone()
    legacy_optimizer.step()
    legacy_standard = [
        before - parameter.detach()
        for before, parameter in zip(legacy_before, legacy_parameters)
    ]
    materialize_descent_updates_(
        legacy_parameters, legacy_before, legacy_composition.updates,
    )
    legacy_reconciliation = (
        reconcile_adamw_first_moments_to_materialized_updates_(
            legacy_optimizer,
            legacy_parameters,
            legacy_before,
            legacy_composition.updates,
        )
    )

    injector = ActionInjectorV1(ActionInjectorV1Config(
        target_attributable_fraction=0.25,
        minimum_protective_component_retention=0.90,
        maximum_update_norm_ratio_to_original=1.50,
    ))
    prepared = injector.prepare(
        v1_optimizer,
        v1_parameters,
        clipped_combined_gradients=combined,
        clipped_noncorrective_gradients=noncorrective,
        clipped_protective_gradients=protective,
        parameter_group_positions=groups,
    )
    for parameter, gradient in zip(v1_parameters, combined):
        parameter.grad = gradient.clone()
    applied = injector.step_and_inject_(
        v1_optimizer, v1_parameters, prepared,
    )

    _assert_ledger_equal(prepared.virtual_combined_updates, legacy_combined)
    _assert_ledger_equal(
        prepared.virtual_noncorrective_updates, legacy_noncorrective,
    )
    _assert_ledger_equal(prepared.virtual_protective_updates, legacy_protective)
    _assert_ledger_equal(prepared.composition.updates, legacy_composition.updates)
    _assert_ledger_equal(
        prepared.composition.counterfactual_baseline_updates,
        legacy_composition.counterfactual_baseline_updates,
    )
    _assert_ledger_equal(applied.standard_adamw_updates, legacy_standard)
    for left, right in zip(v1_parameters, legacy_parameters):
        assert torch.equal(left, right)

    v1_state = _optimizer_state_snapshot(v1_optimizer, v1_parameters)
    legacy_state = _optimizer_state_snapshot(legacy_optimizer, legacy_parameters)
    assert [set(value) for value in v1_state] == [
        set(value) for value in legacy_state
    ]
    for observed, expected in zip(v1_state, legacy_state):
        for name in observed:
            left, right = observed[name], expected[name]
            if torch.is_tensor(left):
                assert torch.is_tensor(right) and torch.equal(left, right), name
            else:
                assert left == right
    assert applied.reconciliation == legacy_reconciliation
    assert applied.virtual_adamw_relative_error == 0.0


def test_prepared_receipt_cannot_be_reused_after_optimizer_state_advances() -> None:
    parameters, optimizer = _optimizer_fixture()
    combined, noncorrective, protective = _gradient_ledgers()
    injector = ActionInjectorV1(ActionInjectorV1Config(0.25, 0.90, 1.50))
    prepared = injector.prepare(
        optimizer,
        parameters,
        clipped_combined_gradients=combined,
        clipped_noncorrective_gradients=noncorrective,
        clipped_protective_gradients=protective,
        parameter_group_positions={"head": [0], "backbone": [1]},
    )
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient.clone()
    optimizer.step()
    try:
        injector.step_and_inject_(optimizer, parameters, prepared)
    except RuntimeError as error:
        assert "state advanced" in str(error)
    else:
        raise AssertionError("stale ActionInjectorV1 receipt did not fail closed")


def test_parameter_groups_and_prepared_configuration_fail_closed() -> None:
    parameters, optimizer = _optimizer_fixture()
    combined, noncorrective, protective = _gradient_ledgers()
    injector = ActionInjectorV1(ActionInjectorV1Config(0.25, 0.90, 1.50))
    try:
        injector.prepare(
            optimizer,
            parameters,
            clipped_combined_gradients=combined,
            clipped_noncorrective_gradients=noncorrective,
            clipped_protective_gradients=protective,
            parameter_group_positions={"head": [1], "backbone": [0]},
        )
    except ValueError as error:
        assert "differ from AdamW groups" in str(error)
    else:
        raise AssertionError("mismatched AdamW group partition did not fail closed")

    prepared = injector.prepare(
        optimizer,
        parameters,
        clipped_combined_gradients=combined,
        clipped_noncorrective_gradients=noncorrective,
        clipped_protective_gradients=protective,
        parameter_group_positions={"head": [0], "backbone": [1]},
    )
    another = ActionInjectorV1(ActionInjectorV1Config(0.30, 0.90, 1.50))
    try:
        another.step_and_inject_(optimizer, parameters, prepared)
    except RuntimeError as error:
        assert "another configuration" in str(error)
    else:
        raise AssertionError("cross-configuration receipt did not fail closed")


def test_trainer_routes_safe_exact_scope_only_through_v1() -> None:
    source = Path(__file__).with_name(
        "train_noise_corrected_routed_direct.py"
    ).read_text(encoding="utf-8")
    assert "ActionInjectorV1(ActionInjectorV1Config(" in source
    assert "safe_exact_injector.prepare(" in source
    assert "safe_exact_injector.step_and_inject_(" in source
    assert "compose_safe_exact_corrective_updates_by_group(" not in source
    safe_path = source[
        source.index("if safe_exact_restoration:", source.index(
            "prepared_action_injection = None"
        )):source.index("else:", source.index(
            "if safe_exact_restoration:", source.index(
                "prepared_action_injection = None"
            )
        ))
    ]
    assert ".counterfactual_baseline_updates" in safe_path
    assert "virtual_attribution_baseline = virtual_noncorrective" not in safe_path
    assert '"action_injector_v1_sha256"' in source
    assert '"versioned_action_injector_v1_boundary_used"' in source


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"[test_noise_action_injector_v1] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
