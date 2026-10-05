"""Versioned, action-agnostic packaging of the validated V10 injector.

``ActionInjectorV1`` owns the optimizer-boundary mechanism only.  Upstream
code may construct its corrective, auxiliary, and protective gradients from
any spectrum action family; the injector never inspects action identities or
spectra.  It receives three *already clipped* gradient ledgers evaluated at
the same AdamW state:

* the combined training gradient;
* the noncorrective (protective + admitted auxiliary) counterfactual; and
* the protective counterfactual that defines the hard safety axis.

The numerical path is deliberately the V10 path, not a redesign: construct
three virtual AdamW updates, apply groupwise hard-safe exact-dose composition,
execute the ordinary AdamW step once, materialize the composed displacement,
and reconcile only AdamW's first moment.  The actual combined gradient remains
the source of the second moment and optimizer step number.

This module is versioned rather than edited in place.  A behavior change must
be introduced as a new injector version and compared with the golden V10
implementation; otherwise frozen dependency checks fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from noise_corrected_update_arbitration_v4 import (
    materialize_descent_updates_,
    reconcile_adamw_first_moments_to_materialized_updates_,
)
from noise_corrected_update_arbitration_v10 import (
    SafeExactGroupwiseUpdateResult,
    compose_safe_exact_corrective_updates_by_group,
)


ACTION_INJECTOR_V1_NAME = "noise_action_injector_v1"
ACTION_INJECTOR_V1_VERSION = 1
ACTION_INJECTOR_V1_FROZEN_DEPENDENCY_SHA256 = {
    "noise_corrected_update_arbitration_v10.py": (
        "00b587506b6ac5755c6ba3f1750c879eca993981ad191555951dd8d0552a3193"
    ),
    "noise_corrected_update_arbitration_v4.py": (
        "6c11387102eb5edb54783d3a0f0ea1ed7ed3b1d63976249d0cde51e6039c3473"
    ),
}

# These are the exact non-configurable numerical choices in the completed V10
# execution.  They are public for audit, not knobs for callers.
ACTION_INJECTOR_V1_NUMERICAL_CONSTANTS = {
    "protective_floor_float32_guard": 1e-7,
    "protective_floor_acceptance_tolerance": 1e-6,
    "exact_fraction_acceptance_tolerance": 2e-6,
    "update_norm_ratio_acceptance_tolerance": 1e-7,
    "adamw_reconciliation_relative_error_maximum": 1e-6,
    "virtual_adamw_relative_error_maximum": 1e-3,
}

ACTION_INJECTOR_V1_SEMANTIC_CONTRACT = {
    "name": ACTION_INJECTOR_V1_NAME,
    "version": ACTION_INJECTOR_V1_VERSION,
    "input_boundary": "three_aligned_already_clipped_gradient_ledgers",
    "action_semantics": "opaque_upstream_any_action_family",
    "optimizer": "torch.optim.AdamW",
    "counterfactuals": {
        "combined": "actual_combined_gradient_same_pre_step_adamw_state",
        "noncorrective": "protective_plus_admitted_auxiliary_without_corrective",
        "protective": "protective_gradient_only",
    },
    "composition": [
        "repair_noncorrective_baseline_to_hard_protective_floor",
        "remove_only_corrective_opposition_to_protective_axis",
        "solve_analytic_per_group_corrective_coefficient_for_exact_fraction",
        "fail_if_any_group_misses_floor_or_fraction_or_norm_cap",
    ],
    "state_transition": [
        "snapshot_parameters",
        "one_ordinary_adamw_step_on_actual_combined_gradient",
        "measure_ordinary_step_against_virtual_combined_update",
        "materialize_composed_descent_update",
        "reconcile_adamw_first_moment_to_realized_materialized_displacement",
    ],
    "second_moment_source": "actual_combined_gradient",
    "step_number_policy": "ordinary_step_preserved_no_extra_optimizer_step",
    "numerical_constants": ACTION_INJECTOR_V1_NUMERICAL_CONSTANTS,
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256 = _canonical_sha256(
    ACTION_INJECTOR_V1_SEMANTIC_CONTRACT
)


def action_injector_v1_contract_manifest() -> dict[str, object]:
    """Return a JSON-safe, detached description of the frozen V1 contract."""
    return json.loads(json.dumps({
        **ACTION_INJECTOR_V1_SEMANTIC_CONTRACT,
        "semantic_contract_sha256": ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256,
        "frozen_dependency_sha256": ACTION_INJECTOR_V1_FROZEN_DEPENDENCY_SHA256,
    }, sort_keys=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_action_injector_v1_frozen_dependencies() -> dict[str, str]:
    """Fail if either numerical dependency differs from the validated V10."""
    root = Path(__file__).resolve().parent
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for name, expected in ACTION_INJECTOR_V1_FROZEN_DEPENDENCY_SHA256.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"ActionInjectorV1 dependency is missing: {path}")
        actual = _sha256_file(path)
        observed[name] = actual
        if actual != expected:
            mismatches[name] = {"observed": actual, "expected": expected}
    if mismatches:
        raise RuntimeError(
            "ActionInjectorV1 frozen numerical dependency drifted; create a new "
            "injector version instead of changing V1: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


@dataclass(frozen=True)
class ActionInjectorV1Config:
    """The three registered V10 decision thresholds.

    Numerical tolerances are intentionally absent: they are part of the
    versioned implementation contract and cannot be relaxed by a caller.
    """

    target_attributable_fraction: float
    minimum_protective_component_retention: float
    maximum_update_norm_ratio_to_original: float

    def __post_init__(self) -> None:
        values = (
            self.target_attributable_fraction,
            self.minimum_protective_component_retention,
            self.maximum_update_norm_ratio_to_original,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("ActionInjectorV1 configuration must be finite")
        if not 0 < self.target_attributable_fraction < 1:
            raise ValueError("target attributable fraction must be in (0, 1)")
        if not 0 <= self.minimum_protective_component_retention <= 1:
            raise ValueError("minimum protective retention must be in [0, 1]")
        if self.maximum_update_norm_ratio_to_original < 1:
            raise ValueError("maximum update norm ratio must be at least one")

    def as_dict(self) -> dict[str, float]:
        return {
            "target_attributable_fraction": float(
                self.target_attributable_fraction
            ),
            "minimum_protective_component_retention": float(
                self.minimum_protective_component_retention
            ),
            "maximum_update_norm_ratio_to_original": float(
                self.maximum_update_norm_ratio_to_original
            ),
        }


@dataclass(frozen=True)
class PreparedActionInjectionV1:
    """Immutable lifecycle receipt produced before the ordinary AdamW step."""

    virtual_combined_updates: list[torch.Tensor | None]
    virtual_noncorrective_updates: list[torch.Tensor | None]
    virtual_protective_updates: list[torch.Tensor | None]
    composition: SafeExactGroupwiseUpdateResult
    parameter_ids: tuple[int, ...]
    optimizer_step_signature: tuple[int, ...]
    parameter_group_positions: dict[str, tuple[int, ...]]
    semantic_contract_sha256: str
    configuration_sha256: str


@dataclass(frozen=True)
class AppliedActionInjectionV1:
    """Receipt after the one ordinary step and V1 materialization."""

    updates: list[torch.Tensor | None]
    counterfactual_baseline_updates: list[torch.Tensor | None]
    standard_adamw_updates: list[torch.Tensor | None]
    virtual_adamw_relative_error: float
    composition: SafeExactGroupwiseUpdateResult
    reconciliation: dict[str, float | int | bool | str]
    semantic_contract_sha256: str
    configuration_sha256: str


def _ledger_norm(values: list[torch.Tensor | None]) -> float:
    """Match the trainer's V10 diagnostic norm arithmetic exactly."""
    terms = [
        torch.sum(value.detach().float() ** 2).double()
        for value in values if value is not None
    ]
    total = float(torch.stack(terms).sum().item()) if terms else 0.0
    return float(np.sqrt(total))


def _optimizer_step_signature(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
) -> tuple[int, ...]:
    output: list[int] = []
    for parameter in parameters:
        raw_step = optimizer.state.get(parameter, {}).get("step", 0)
        output.append(
            int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
        )
    return tuple(output)


def _validate_parameter_group_partition(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    parameter_group_positions: dict[str, list[int]],
) -> dict[str, tuple[int, ...]]:
    position = {id(parameter): index for index, parameter in enumerate(parameters)}
    expected_ids = set(position)
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if (
        set(optimizer_parameter_ids) != expected_ids
        or len(optimizer_parameter_ids) != len(expected_ids)
    ):
        raise RuntimeError(
            "ActionInjectorV1 requires the full optimizer parameter set"
        )
    expected_groups: dict[str, tuple[int, ...]] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", f"group_{group_index}"))
        if name in expected_groups:
            raise RuntimeError(
                f"duplicate optimizer parameter-group name: {name}"
            )
        expected_groups[name] = tuple(
            position[id(parameter)] for parameter in group["params"]
        )
    observed = [
        int(position)
        for positions in parameter_group_positions.values()
        for position in positions
    ]
    if (
        not parameter_group_positions
        or len(observed) != len(parameters)
        or set(observed) != set(range(len(parameters)))
    ):
        raise ValueError(
            "ActionInjectorV1 parameter groups must partition every parameter "
            "exactly once"
        )
    observed_groups = {
        str(name): tuple(map(int, positions))
        for name, positions in parameter_group_positions.items()
    }
    if observed_groups != expected_groups:
        raise ValueError(
            "ActionInjectorV1 named parameter groups differ from AdamW groups"
        )
    return observed_groups


def virtual_adamw_descent_updates_v1(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    gradients: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Evaluate the exact V10 virtual AdamW descent update without mutation."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("ActionInjectorV1 requires torch.optim.AdamW")
    if len(parameters) != len(gradients):
        raise ValueError("virtual AdamW parameters and gradients must align")
    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if set(groups) != {id(parameter) for parameter in parameters}:
        raise RuntimeError(
            "virtual AdamW did not receive the full optimizer parameter set"
        )

    updates: list[torch.Tensor | None] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            updates.append(None)
            continue
        group = groups[id(parameter)]
        beta1, beta2 = map(float, group["betas"])
        if group.get("differentiable", False):
            raise RuntimeError("differentiable AdamW is unsupported by V1")
        grad = gradient.detach()
        if group.get("maximize", False):
            grad = -grad
        state = optimizer.state.get(parameter, {})
        raw_step = state.get("step", 0)
        step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
        step += 1
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(parameter)
        if exp_avg_sq is None:
            exp_avg_sq = torch.zeros_like(parameter)
        next_avg = exp_avg.detach().clone()
        next_avg.lerp_(grad, 1.0 - beta1)
        next_sq = exp_avg_sq.detach().clone()
        next_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)
        if group.get("amsgrad", False):
            maximum = state.get("max_exp_avg_sq")
            if maximum is None:
                maximum = torch.zeros_like(parameter)
            denominator_sq = torch.maximum(maximum.detach(), next_sq)
        else:
            denominator_sq = next_sq
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denominator = denominator_sq.sqrt() / np.sqrt(bias2)
        denominator.add_(float(group["eps"]))
        after = parameter.detach().clone()
        after.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        after.addcdiv_(
            next_avg,
            denominator,
            value=-(float(group["lr"]) / bias1),
        )
        updates.append(parameter.detach() - after)
    return updates


class ActionInjectorV1:
    """Frozen V10 optimizer-boundary injector, independent of action type."""

    def __init__(
        self,
        config: ActionInjectorV1Config,
        *,
        verify_frozen_dependencies: bool = True,
    ) -> None:
        self._config = config
        self._configuration_sha256 = _canonical_sha256(config.as_dict())
        self._dependency_sha256 = (
            validate_action_injector_v1_frozen_dependencies()
            if verify_frozen_dependencies else {}
        )

    @property
    def config(self) -> ActionInjectorV1Config:
        return self._config

    def audit_manifest(self) -> dict[str, object]:
        return {
            **action_injector_v1_contract_manifest(),
            "configuration": self.config.as_dict(),
            "configuration_sha256": self._configuration_sha256,
            "validated_dependency_sha256": dict(self._dependency_sha256),
        }

    def prepare(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        *,
        clipped_combined_gradients: list[torch.Tensor | None],
        clipped_noncorrective_gradients: list[torch.Tensor | None],
        clipped_protective_gradients: list[torch.Tensor | None],
        parameter_group_positions: dict[str, list[int]],
    ) -> PreparedActionInjectionV1:
        """Construct all same-state V10 counterfactuals and fail closed."""
        if not (
            len(parameters)
            == len(clipped_combined_gradients)
            == len(clipped_noncorrective_gradients)
            == len(clipped_protective_gradients)
        ):
            raise ValueError("ActionInjectorV1 gradient ledgers must align")
        frozen_groups = _validate_parameter_group_partition(
            optimizer, parameters, parameter_group_positions,
        )
        virtual_combined = virtual_adamw_descent_updates_v1(
            optimizer, parameters, clipped_combined_gradients,
        )
        virtual_noncorrective = virtual_adamw_descent_updates_v1(
            optimizer, parameters, clipped_noncorrective_gradients,
        )
        virtual_protective = virtual_adamw_descent_updates_v1(
            optimizer, parameters, clipped_protective_gradients,
        )
        composition = compose_safe_exact_corrective_updates_by_group(
            virtual_combined,
            virtual_noncorrective,
            virtual_protective,
            {name: list(positions) for name, positions in frozen_groups.items()},
            target_attributable_fraction=(
                self.config.target_attributable_fraction
            ),
            minimum_protective_component_retention=(
                self.config.minimum_protective_component_retention
            ),
            materialize_updates=True,
        )
        if not composition.all_groups_protective_floor_enforced:
            raise RuntimeError("V10 hard protective floor was not enforced")
        if not composition.all_groups_target_reached:
            raise RuntimeError(
                "V10 exact optimizer corrective fraction was not reached in "
                "every parameter group"
            )
        norm_tolerance = ACTION_INJECTOR_V1_NUMERICAL_CONSTANTS[
            "update_norm_ratio_acceptance_tolerance"
        ]
        if (
            composition.maximum_group_update_norm_ratio_to_original
            > self.config.maximum_update_norm_ratio_to_original + norm_tolerance
        ):
            raise RuntimeError(
                "V10 hard-safe update exceeds the registered norm-ratio cap: "
                f"{composition.maximum_group_update_norm_ratio_to_original:.6f} "
                f"> {self.config.maximum_update_norm_ratio_to_original:.6f}"
            )
        return PreparedActionInjectionV1(
            virtual_combined_updates=virtual_combined,
            virtual_noncorrective_updates=virtual_noncorrective,
            virtual_protective_updates=virtual_protective,
            composition=composition,
            parameter_ids=tuple(id(parameter) for parameter in parameters),
            optimizer_step_signature=_optimizer_step_signature(
                optimizer, parameters,
            ),
            parameter_group_positions=frozen_groups,
            semantic_contract_sha256=(
                ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256
            ),
            configuration_sha256=self._configuration_sha256,
        )

    def step_and_inject_(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        prepared: PreparedActionInjectionV1,
    ) -> AppliedActionInjectionV1:
        """Execute exactly one ordinary step, then materialize and reconcile V1."""
        if prepared.semantic_contract_sha256 != (
            ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256
        ):
            raise RuntimeError("ActionInjectorV1 prepared receipt has contract drift")
        if prepared.configuration_sha256 != self._configuration_sha256:
            raise RuntimeError(
                "ActionInjectorV1 prepared receipt belongs to another configuration"
            )
        if tuple(id(parameter) for parameter in parameters) != prepared.parameter_ids:
            raise RuntimeError("ActionInjectorV1 prepared for different parameters")
        if _optimizer_step_signature(optimizer, parameters) != (
            prepared.optimizer_step_signature
        ):
            raise RuntimeError(
                "AdamW state advanced between ActionInjectorV1 prepare and apply"
            )
        _validate_parameter_group_partition(
            optimizer,
            parameters,
            {
                name: list(positions)
                for name, positions in prepared.parameter_group_positions.items()
            },
        )

        before_parameters = [
            parameter.detach().clone() for parameter in parameters
        ]
        optimizer.step()
        standard_updates = [
            before - parameter.detach()
            for before, parameter in zip(before_parameters, parameters)
        ]
        virtual_residual = [
            None if actual is None or expected is None else actual - expected
            for actual, expected in zip(
                standard_updates, prepared.virtual_combined_updates,
            )
        ]
        standard_norm = _ledger_norm(standard_updates)
        virtual_error = (
            _ledger_norm(virtual_residual) / standard_norm
            if standard_norm > 0 else 0.0
        )

        materialize_descent_updates_(
            parameters,
            before_parameters,
            prepared.composition.updates,
        )
        reconciliation = reconcile_adamw_first_moments_to_materialized_updates_(
            optimizer,
            parameters,
            before_parameters,
            prepared.composition.updates,
        )
        if not reconciliation["gate_passed"]:
            raise RuntimeError(
                "restored AdamW first-moment reconciliation failed: "
                f"{reconciliation}"
            )
        return AppliedActionInjectionV1(
            updates=prepared.composition.updates,
            counterfactual_baseline_updates=(
                prepared.composition.counterfactual_baseline_updates
            ),
            standard_adamw_updates=standard_updates,
            virtual_adamw_relative_error=float(virtual_error),
            composition=prepared.composition,
            reconciliation=reconciliation,
            semantic_contract_sha256=(
                ACTION_INJECTOR_V1_SEMANTIC_CONTRACT_SHA256
            ),
            configuration_sha256=self._configuration_sha256,
        )
