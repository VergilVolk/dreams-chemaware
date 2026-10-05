"""Hard-safe, exact-dose optimizer composition for noise direct fine-tuning.

V7--V9 measured a corrective AdamW residual correctly, but two properties were
only audited rather than enforced: the protective component could remain below
its registered floor, and causal arms could deliver different optimizer-space
corrective fractions after matching only their pre-optimizer gradient norms.

This module keeps the existing AdamW counterfactuals and changes only their
parameter-space composition.  For every optimizer parameter group it:

1. projects the noncorrective counterfactual onto the registered protective
   half-space;
2. removes only opposition between the corrective residual and the protective
   update; and
3. solves analytically for the corrective coefficient whose *materialized*
   attributable-update fraction equals the registered target.

The protective projection is part of the noncorrective counterfactual, so it is
never misreported as action-attributable signal.  No teacher, distillation
target, extra optimizer step, learning-rate scan, or new action is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from noise_corrected_update_arbitration_v4 import (
    UpdateArbitrationResult,
    _aligned,
    _cosine,
    _stats,
)


@dataclass(frozen=True)
class SafeExactGroupwiseUpdateResult:
    updates: list[torch.Tensor | None]
    counterfactual_baseline_updates: list[torch.Tensor | None]
    parameter_groups: dict[str, UpdateArbitrationResult]
    original_update_norm: float
    final_update_norm: float
    all_groups_target_reached: bool
    all_groups_protective_floor_enforced: bool
    minimum_group_risk_component_retention: float
    maximum_group_attributable_fraction_abs_error: float
    maximum_group_update_norm_ratio_to_original: float


def _clone(values: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [None if value is None else value.clone() for value in values]


def _add_scaled(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
    scale: float,
) -> list[torch.Tensor | None]:
    output: list[torch.Tensor | None] = []
    for base, direction in zip(left, right):
        if base is None:
            output.append(None)
        elif direction is None:
            output.append(base.clone())
        else:
            output.append(base + float(scale) * direction)
    return output


def _exact_fraction_coefficient(
    *,
    baseline_sq: float,
    direction_sq: float,
    baseline_direction_dot: float,
    target_fraction: float,
) -> float:
    """Solve ||a*d|| / ||b + a*d|| = q for the positive coefficient a."""
    q = float(target_fraction)
    if baseline_sq <= 0 or direction_sq <= 0:
        raise ValueError("exact corrective composition requires non-zero vectors")
    q2 = q * q
    discriminant = (
        q2 * q2 * baseline_direction_dot * baseline_direction_dot
        + q2 * (1.0 - q2) * direction_sq * baseline_sq
    )
    coefficient = (
        q2 * baseline_direction_dot + math.sqrt(max(discriminant, 0.0))
    ) / ((1.0 - q2) * direction_sq)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise RuntimeError("failed to solve a positive exact-dose coefficient")
    return float(coefficient)


def compose_safe_exact_corrective_updates(
    combined_updates: list[torch.Tensor | None],
    noncorrective_baseline_updates: list[torch.Tensor | None],
    protective_updates: list[torch.Tensor | None],
    *,
    target_attributable_fraction: float,
    minimum_protective_component_retention: float,
    materialize_updates: bool = True,
) -> tuple[
    UpdateArbitrationResult,
    list[torch.Tensor | None],
    bool,
    float,
]:
    """Compose one parameter group with a hard floor and exact action dose.

    The fourth return value is the final/original update-norm ratio.  Unlike the
    legacy norm-neutral restoration, V10 gives the hard protective constraint
    priority.  The norm change is explicit and gated by the caller rather than
    hidden behind an unsafe fallback.
    """
    if not 0 < target_attributable_fraction < 1:
        raise ValueError("target attributable fraction must be in (0, 1)")
    if not 0 <= minimum_protective_component_retention <= 1:
        raise ValueError("minimum protective retention must be in [0, 1]")
    combined, baseline = _aligned(
        combined_updates, noncorrective_baseline_updates,
    )
    _, protective = _aligned(combined_updates, protective_updates)
    corrective = [
        None if value is None else value - reference
        for value, reference in zip(combined, baseline)
    ]
    combined_sq, _, _ = _stats(combined, combined)
    corrective_sq, _, _ = _stats(corrective, corrective)
    _, protective_sq, corrective_protective_dot = _stats(
        corrective, protective,
    )
    baseline_sq, _, baseline_protective_dot = _stats(baseline, protective)
    _, _, combined_protective_dot = _stats(combined, protective)
    if combined_sq <= 0 or baseline_sq <= 0:
        raise ValueError("combined and noncorrective optimizer updates must be non-zero")

    original_fraction = float(math.sqrt(max(corrective_sq, 0.0) / combined_sq))
    original_retention = (
        float(combined_protective_dot / protective_sq)
        if protective_sq > 0 else 1.0
    )

    # The safety repair belongs to the noncorrective counterfactual.  A tiny
    # numerical guard keeps float32 materialization on the safe side of the
    # inclusive registered floor without creating a meaningful extra dose.
    requested_floor = float(minimum_protective_component_retention)
    guarded_floor = min(1.0, requested_floor + 1e-7)
    baseline_retention = (
        float(baseline_protective_dot / protective_sq)
        if protective_sq > 0 else 1.0
    )
    protective_coefficient = max(0.0, guarded_floor - baseline_retention)
    safe_baseline = _add_scaled(baseline, protective, protective_coefficient)
    safe_baseline_sq, _, safe_baseline_protective_dot = _stats(
        safe_baseline, protective,
    )

    # Corrective semantics may never consume the guaranteed protective axis.
    opposition = (
        corrective_protective_dot / protective_sq
        if protective_sq > 0 and corrective_protective_dot < 0 else 0.0
    )
    safe_corrective = _add_scaled(corrective, protective, -opposition)
    safe_corrective_sq, _, safe_corrective_protective_dot = _stats(
        safe_corrective, protective,
    )
    _, _, safe_baseline_corrective_dot = _stats(
        safe_baseline, safe_corrective,
    )

    if safe_corrective_sq <= 0:
        final = safe_baseline
        final_sq, _, final_protective_dot = _stats(final, protective)
        final_retention = (
            float(final_protective_dot / protective_sq)
            if protective_sq > 0 else 1.0
        )
        report = UpdateArbitrationResult(
            updates=_clone(final) if materialize_updates else [],
            action_gain=0.0,
            norm_rescale=1.0,
            original_update_norm=float(math.sqrt(combined_sq)),
            final_update_norm=float(math.sqrt(final_sq)),
            original_attributable_fraction=original_fraction,
            final_attributable_fraction=0.0,
            action_risk_cosine_before=_cosine(
                corrective_sq, protective_sq, corrective_protective_dot,
            ),
            action_risk_cosine_after=None,
            target_reached=False,
            gain_cap_hit=False,
            risk_component_before=combined_protective_dot,
            risk_component_after=final_protective_dot,
            risk_component_retention=final_retention,
            risk_constraint_active=bool(protective_coefficient > 0 or opposition < 0),
        )
        return (
            report,
            _clone(safe_baseline) if materialize_updates else [],
            bool(final_retention + 1e-6 >= requested_floor),
            float(math.sqrt(final_sq / combined_sq)),
        )

    coefficient = _exact_fraction_coefficient(
        baseline_sq=safe_baseline_sq,
        direction_sq=safe_corrective_sq,
        baseline_direction_dot=safe_baseline_corrective_dot,
        target_fraction=target_attributable_fraction,
    )
    final = _add_scaled(safe_baseline, safe_corrective, coefficient)
    attributable = [
        None if value is None else value - reference
        for value, reference in zip(final, safe_baseline)
    ]
    final_sq, _, final_protective_dot = _stats(final, protective)
    attributable_sq, _, _ = _stats(attributable, attributable)
    final_fraction = float(math.sqrt(attributable_sq / final_sq))
    final_retention = (
        float(final_protective_dot / protective_sq)
        if protective_sq > 0 else 1.0
    )
    fraction_tolerance = 2e-6
    floor_tolerance = 1e-6
    fraction_reached = bool(
        abs(final_fraction - target_attributable_fraction) <= fraction_tolerance
    )
    floor_enforced = bool(final_retention + floor_tolerance >= requested_floor)
    if not floor_enforced:
        raise RuntimeError(
            "hard protective floor failed after materialization: "
            f"{final_retention:.9f} < {requested_floor:.9f}"
        )
    report = UpdateArbitrationResult(
        updates=_clone(final) if materialize_updates else [],
        action_gain=coefficient,
        norm_rescale=1.0,
        original_update_norm=float(math.sqrt(combined_sq)),
        final_update_norm=float(math.sqrt(final_sq)),
        original_attributable_fraction=original_fraction,
        final_attributable_fraction=final_fraction,
        action_risk_cosine_before=_cosine(
            corrective_sq, protective_sq, corrective_protective_dot,
        ),
        action_risk_cosine_after=_cosine(
            safe_corrective_sq,
            protective_sq,
            safe_corrective_protective_dot,
        ),
        target_reached=fraction_reached and floor_enforced,
        gain_cap_hit=False,
        risk_component_before=combined_protective_dot,
        risk_component_after=final_protective_dot,
        risk_component_retention=final_retention,
        risk_constraint_active=bool(protective_coefficient > 0 or opposition < 0),
    )
    return (
        report,
        _clone(safe_baseline) if materialize_updates else [],
        floor_enforced,
        float(math.sqrt(final_sq / combined_sq)),
    )


def compose_safe_exact_corrective_updates_by_group(
    combined_updates: list[torch.Tensor | None],
    noncorrective_baseline_updates: list[torch.Tensor | None],
    protective_updates: list[torch.Tensor | None],
    parameter_group_positions: dict[str, list[int]],
    *,
    target_attributable_fraction: float,
    minimum_protective_component_retention: float,
    materialize_updates: bool = True,
) -> SafeExactGroupwiseUpdateResult:
    """Apply V10 composition independently to every optimizer group."""
    if not (
        len(combined_updates)
        == len(noncorrective_baseline_updates)
        == len(protective_updates)
    ):
        raise ValueError("safe exact update layouts must align")
    expected = set(range(len(combined_updates)))
    observed = [
        int(position)
        for positions in parameter_group_positions.values()
        for position in positions
    ]
    if (
        not parameter_group_positions
        or set(observed) != expected
        or len(observed) != len(expected)
    ):
        raise ValueError("parameter groups must partition every update exactly once")

    output: list[torch.Tensor | None] = [None] * len(combined_updates)
    counterfactual: list[torch.Tensor | None] = [None] * len(combined_updates)
    reports: dict[str, UpdateArbitrationResult] = {}
    floor_results: list[bool] = []
    norm_ratios: list[float] = []
    for name in sorted(parameter_group_positions):
        positions = list(map(int, parameter_group_positions[name]))
        report, safe_baseline, floor_enforced, norm_ratio = (
            compose_safe_exact_corrective_updates(
                [combined_updates[position] for position in positions],
                [noncorrective_baseline_updates[position] for position in positions],
                [protective_updates[position] for position in positions],
                target_attributable_fraction=target_attributable_fraction,
                minimum_protective_component_retention=(
                    minimum_protective_component_retention
                ),
                materialize_updates=materialize_updates,
            )
        )
        reports[name] = report
        floor_results.append(floor_enforced)
        norm_ratios.append(norm_ratio)
        if materialize_updates:
            for position, update, baseline_update in zip(
                positions, report.updates, safe_baseline,
            ):
                output[position] = update
                counterfactual[position] = baseline_update

    original_norm = float(math.sqrt(sum(
        report.original_update_norm**2 for report in reports.values()
    )))
    final_norm = float(math.sqrt(sum(
        report.final_update_norm**2 for report in reports.values()
    )))
    return SafeExactGroupwiseUpdateResult(
        updates=output if materialize_updates else [],
        counterfactual_baseline_updates=(
            counterfactual if materialize_updates else []
        ),
        parameter_groups=reports,
        original_update_norm=original_norm,
        final_update_norm=final_norm,
        all_groups_target_reached=all(
            report.target_reached for report in reports.values()
        ),
        all_groups_protective_floor_enforced=all(floor_results),
        minimum_group_risk_component_retention=min(
            report.risk_component_retention for report in reports.values()
        ),
        maximum_group_attributable_fraction_abs_error=max(
            abs(
                report.final_attributable_fraction
                - target_attributable_fraction
            )
            for report in reports.values()
        ),
        maximum_group_update_norm_ratio_to_original=max(norm_ratios),
    )
