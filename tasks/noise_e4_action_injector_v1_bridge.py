"""Thin E4-to-ActionInjectorV1 optimizer-boundary adapter.

The historical E4 trainer owns spectra, action selection, the symmetric/shared
loss and its schedule.  ``ActionInjectorV1`` owns only the final AdamW update.
This module joins those already-existing boundaries without interpreting,
filtering, averaging or otherwise changing an action.

The caller performs the unchanged E4 forward/backward sequence:

1. backward the complete E4 action loss;
2. snapshot that corrective gradient with :meth:`capture_corrective_`;
3. backward the matched E4 safety loss into the same ``parameter.grad``;
4. call :meth:`step_and_inject_` once.

The bridge derives the protective gradient by subtracting the captured E4
gradient from the accumulated combined gradient.  The same global clipping
coefficient is then applied to all three ledgers, exactly as required by the
frozen Injector V1 interface.  No second optimizer step is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from noise_action_injector_v1 import (
    ActionInjectorV1,
    ActionInjectorV1Config,
    AppliedActionInjectionV1,
)


@dataclass(frozen=True)
class E4ActionInjectorV1Step:
    """JSON-safe diagnostics from one materialized E4 optimizer step."""

    gradient_norm_before_clip: float
    global_clip_retention: float
    action_gradient_norm_before_clip: float
    protective_gradient_norm_before_clip: float
    optimizer_action_fraction_by_group: dict[str, float]
    protective_component_retention_by_group: dict[str, float]
    action_gain_by_group: dict[str, float]
    maximum_fraction_abs_error: float
    maximum_update_norm_ratio: float
    virtual_adamw_relative_error: float
    first_moment_reconstruction_relative_error: float
    fp32_parameter_replay_relative_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _clone_gradients(
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _subtract_gradients(
    combined: Sequence[torch.Tensor | None],
    corrective: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    if len(combined) != len(corrective):
        raise ValueError("E4 combined/corrective gradient ledgers differ")
    output: list[torch.Tensor | None] = []
    for total, action in zip(combined, corrective):
        if total is None and action is None:
            output.append(None)
        elif total is None:
            raise RuntimeError("E4 corrective gradient vanished from combined ledger")
        elif action is None:
            output.append(total.detach().clone())
        else:
            output.append(total.detach() - action.detach())
    return output


def _scaled(
    gradients: Sequence[torch.Tensor | None], scale: float,
) -> list[torch.Tensor | None]:
    return [
        None if gradient is None else gradient.detach() * float(scale)
        for gradient in gradients
    ]


def _norm(gradients: Sequence[torch.Tensor | None]) -> float:
    terms = [
        torch.sum(gradient.detach().float() ** 2).double()
        for gradient in gradients if gradient is not None
    ]
    return float(np.sqrt(float(torch.stack(terms).sum()))) if terms else 0.0


class E4ActionInjectorV1Bridge:
    """Apply frozen Injector V1 to the unchanged E4 action/safety gradients."""

    def __init__(
        self,
        optimizer: torch.optim.AdamW,
        parameters: Sequence[torch.nn.Parameter],
        *,
        target_attributable_fraction: float = 0.25,
        minimum_protective_component_retention: float = 0.90,
        maximum_update_norm_ratio_to_original: float = 1.50,
        verify_frozen_dependencies: bool = True,
    ) -> None:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("E4 ActionInjectorV1 bridge requires AdamW")
        self.optimizer = optimizer
        self.parameters = list(parameters)
        if not self.parameters or len({id(value) for value in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("E4 injector parameters must be non-empty and unique")
        position = {id(parameter): index for index, parameter in enumerate(self.parameters)}
        groups: dict[str, list[int]] = {}
        for index, group in enumerate(optimizer.param_groups):
            name = str(group.get("group_name", f"group_{index}"))
            if name in groups:
                raise ValueError(f"duplicate E4 optimizer group name: {name}")
            try:
                groups[name] = [position[id(parameter)] for parameter in group["params"]]
            except KeyError as error:
                raise ValueError("E4 injector did not receive every optimizer parameter") from error
        observed = [value for values in groups.values() for value in values]
        if len(observed) != len(self.parameters) or set(observed) != set(
            range(len(self.parameters))
        ):
            raise ValueError("E4 optimizer groups do not exactly partition parameters")
        self.parameter_group_positions = groups
        self.injector = ActionInjectorV1(
            ActionInjectorV1Config(
                target_attributable_fraction=target_attributable_fraction,
                minimum_protective_component_retention=(
                    minimum_protective_component_retention
                ),
                maximum_update_norm_ratio_to_original=(
                    maximum_update_norm_ratio_to_original
                ),
            ),
            verify_frozen_dependencies=verify_frozen_dependencies,
        )
        self._corrective_gradients: list[torch.Tensor | None] | None = None

    def audit_manifest(self) -> dict[str, object]:
        return {
            "bridge": "historical_E4_action_gradient_to_ActionInjectorV1",
            "action_selection_or_tensor_mutation": False,
            "historical_e4_loss_mutation": False,
            "same_global_clip_scale_for_all_ledgers": True,
            "one_real_adamw_step": True,
            "parameter_group_positions": {
                name: list(positions)
                for name, positions in self.parameter_group_positions.items()
            },
            "injector": self.injector.audit_manifest(),
        }

    def capture_corrective_(self) -> None:
        """Capture the complete E4 action gradient before safety accumulation."""
        if self._corrective_gradients is not None:
            raise RuntimeError("E4 corrective gradient was captured twice")
        captured = _clone_gradients(self.parameters)
        if _norm(captured) <= 0:
            raise RuntimeError("historical E4 action gradient is zero")
        self._corrective_gradients = captured

    def step_and_inject_(self, *, maximum_gradient_norm: float) -> E4ActionInjectorV1Step:
        """Clip, prepare, execute and materialize exactly one Injector V1 step."""
        if self._corrective_gradients is None:
            raise RuntimeError("E4 corrective gradient was not captured")
        if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm <= 0:
            raise ValueError("maximum E4 gradient norm must be finite and positive")
        combined_raw = _clone_gradients(self.parameters)
        protective_raw = _subtract_gradients(
            combined_raw, self._corrective_gradients,
        )
        action_norm = _norm(self._corrective_gradients)
        protective_norm = _norm(protective_raw)
        if protective_norm <= 0:
            raise RuntimeError("historical E4 protective gradient is zero")

        raw_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.parameters, float(maximum_gradient_norm),
        )
        raw_norm = float(raw_norm_tensor)
        if not math.isfinite(raw_norm):
            raise RuntimeError("historical E4 combined gradient is non-finite")
        # This is PyTorch clip_grad_norm_'s exact public coefficient formula.
        clip_retention = min(
            1.0, float(maximum_gradient_norm) / (raw_norm + 1e-6),
        )
        combined_clipped = _clone_gradients(self.parameters)
        protective_clipped = _scaled(protective_raw, clip_retention)
        prepared = self.injector.prepare(
            self.optimizer,
            self.parameters,
            clipped_combined_gradients=combined_clipped,
            clipped_noncorrective_gradients=protective_clipped,
            clipped_protective_gradients=protective_clipped,
            parameter_group_positions=self.parameter_group_positions,
        )
        applied: AppliedActionInjectionV1 = self.injector.step_and_inject_(
            self.optimizer, self.parameters, prepared,
        )
        composition = applied.composition
        group_reports = composition.parameter_groups
        step = E4ActionInjectorV1Step(
            gradient_norm_before_clip=raw_norm,
            global_clip_retention=float(clip_retention),
            action_gradient_norm_before_clip=action_norm,
            protective_gradient_norm_before_clip=protective_norm,
            optimizer_action_fraction_by_group={
                name: float(report.final_attributable_fraction)
                for name, report in group_reports.items()
            },
            protective_component_retention_by_group={
                name: float(report.risk_component_retention)
                for name, report in group_reports.items()
            },
            action_gain_by_group={
                name: float(report.action_gain)
                for name, report in group_reports.items()
            },
            maximum_fraction_abs_error=float(
                composition.maximum_group_attributable_fraction_abs_error
            ),
            maximum_update_norm_ratio=float(
                composition.maximum_group_update_norm_ratio_to_original
            ),
            virtual_adamw_relative_error=float(
                applied.virtual_adamw_relative_error
            ),
            first_moment_reconstruction_relative_error=float(
                applied.reconciliation["same_step_reconstruction_relative_error"]
            ),
            fp32_parameter_replay_relative_error=float(
                applied.reconciliation[
                    "same_step_fp32_parameter_replay_relative_error"
                ]
            ),
        )
        self._corrective_gradients = None
        return step


def summarize_e4_action_injector_v1_steps(
    steps: Sequence[E4ActionInjectorV1Step],
) -> dict[str, object]:
    """Summarize every active step; no sampled tail can hide a failure."""
    if not steps:
        return {"enabled": False, "steps": 0}

    groups = sorted(steps[0].optimizer_action_fraction_by_group)
    if any(sorted(step.optimizer_action_fraction_by_group) != groups for step in steps):
        raise RuntimeError("E4 Injector V1 parameter-group ledger drifted")
    fraction = {
        group: np.asarray([
            step.optimizer_action_fraction_by_group[group] for step in steps
        ], dtype=np.float64)
        for group in groups
    }
    retention = {
        group: np.asarray([
            step.protective_component_retention_by_group[group] for step in steps
        ], dtype=np.float64)
        for group in groups
    }
    report = {
        "enabled": True,
        "steps": int(len(steps)),
        "attribution_sampling": "every_optimizer_step",
        "optimizer_action_fraction_p10_by_group": {
            group: float(np.quantile(values, 0.10))
            for group, values in fraction.items()
        },
        "optimizer_action_fraction_max_abs_error": float(max(
            step.maximum_fraction_abs_error for step in steps
        )),
        "minimum_protective_component_retention_by_group": {
            group: float(np.min(values)) for group, values in retention.items()
        },
        "minimum_global_clip_retention": float(min(
            step.global_clip_retention for step in steps
        )),
        "maximum_update_norm_ratio": float(max(
            step.maximum_update_norm_ratio for step in steps
        )),
        "maximum_virtual_adamw_relative_error": float(max(
            step.virtual_adamw_relative_error for step in steps
        )),
        "maximum_first_moment_reconstruction_relative_error": float(max(
            step.first_moment_reconstruction_relative_error for step in steps
        )),
        "maximum_fp32_parameter_replay_relative_error": float(max(
            step.fp32_parameter_replay_relative_error for step in steps
        )),
    }
    report["gate_passed"] = bool(
        report["optimizer_action_fraction_max_abs_error"] <= 2e-6
        and min(report["minimum_protective_component_retention_by_group"].values())
        + 1e-6 >= 0.90
        and report["maximum_update_norm_ratio"] <= 1.50 + 1e-7
        and report["maximum_virtual_adamw_relative_error"] <= 1e-3
        and report["maximum_first_moment_reconstruction_relative_error"] <= 1e-6
    )
    return report
