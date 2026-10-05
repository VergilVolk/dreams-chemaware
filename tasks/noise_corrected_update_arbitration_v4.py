"""Parameter-update arbitration for direct noise fine-tuning.

The v3 trainer protects the corrective gradient before AdamW, but AdamW is a
non-linear map because it mixes the current gradient with historical first and
second moments.  Consequently a well-preserved action gradient can still leave
only a small action-attributable parameter update.  This module operates on
*virtual AdamW descent updates* and constructs a norm-neutral v4 candidate:

1. decompose the combined update into the risk-only counterfactual plus the
   action-attributable residual;
2. remove only a component of that residual that directly opposes the
   risk-only update;
3. amplify the remaining residual just enough to reach a registered minimum
   attributable fraction, subject to a hard gain cap; and
4. rescale the complete update to the original combined-update norm.

It does not define a teacher, distillation target, learning-rate increase or
extra optimizer dose.  The legacy v6-restored contract restores the complete
action residual relative to risk alone.  The corrective-only successor instead
uses ``protective + projected auxiliary`` as its counterfactual baseline and a
separate protective update as the safety reference, so robust/harmful updates
cannot consume the restored corrective budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class UpdateArbitrationResult:
    updates: list[torch.Tensor | None]
    action_gain: float
    norm_rescale: float
    original_update_norm: float
    final_update_norm: float
    original_attributable_fraction: float
    final_attributable_fraction: float
    action_risk_cosine_before: float | None
    action_risk_cosine_after: float | None
    target_reached: bool
    gain_cap_hit: bool
    risk_component_before: float
    risk_component_after: float
    risk_component_retention: float
    risk_constraint_active: bool


@dataclass(frozen=True)
class GroupwiseUpdateArbitrationResult:
    updates: list[torch.Tensor | None]
    parameter_groups: dict[str, UpdateArbitrationResult]
    original_update_norm: float
    final_update_norm: float
    all_groups_target_reached: bool
    minimum_group_risk_component_retention: float


def _aligned(
    combined: list[torch.Tensor | None],
    risk: list[torch.Tensor | None],
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    if len(combined) != len(risk):
        raise ValueError("combined and risk-only updates must align")
    combined_out: list[torch.Tensor | None] = []
    risk_out: list[torch.Tensor | None] = []
    for left, right in zip(combined, risk):
        if left is None and right is None:
            combined_out.append(None)
            risk_out.append(None)
            continue
        if left is None:
            raise ValueError("risk-only update exists where combined update is absent")
        if not torch.isfinite(left).all():
            raise ValueError("combined update contains non-finite values")
        if right is None:
            right = torch.zeros_like(left)
        if right.shape != left.shape or right.device != left.device:
            raise ValueError("combined and risk-only update tensors must align")
        if not torch.isfinite(right).all():
            raise ValueError("risk-only update contains non-finite values")
        combined_out.append(left.detach())
        risk_out.append(right.detach())
    if not any(value is not None for value in combined_out):
        raise ValueError("at least one optimizer update is required")
    return combined_out, risk_out


def _stats(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> tuple[float, float, float]:
    contributions: list[torch.Tensor] = []
    for x, y in zip(left, right):
        if x is None or y is None:
            continue
        xf = x.float()
        yf = y.float()
        contributions.append(torch.stack((
            torch.sum(xf * xf),
            torch.sum(yf * yf),
            torch.sum(xf * yf),
        )).double())
    if not contributions:
        return 0.0, 0.0, 0.0
    # One device-to-host synchronization per complete parameter group.  The
    # old per-tensor float conversions were acceptable for 64 audit steps but
    # would dominate an exhaustive 14,032-step formal restoration.
    totals = torch.stack(contributions).sum(dim=0).cpu().tolist()
    return tuple(map(float, totals))


def _cosine(left_sq: float, right_sq: float, dot: float) -> float | None:
    if left_sq <= 0 or right_sq <= 0:
        return None
    return float(dot / np.sqrt(left_sq * right_sq))


def _candidate_stats(
    *,
    original_combined_sq: float,
    action_sq: float,
    risk_sq: float,
    dot: float,
    gain: float,
    norm_neutral: bool,
) -> tuple[float, float, float, float]:
    """Return rescale, final norm, true counterfactual fraction and risk dot.

    The action-attributable quantity used by the trainer is the materialized
    update minus the *unscaled* risk-only AdamW counterfactual.  When the
    candidate is made norm neutral, merely reporting ``gain * action`` would
    omit the ``(rescale - 1) * risk`` term and overstate what the signal gate
    will observe.  This helper keeps the optimizer-space design and the gate on
    exactly the same definition.
    """
    gain = float(gain)
    candidate_sq = risk_sq + gain**2 * action_sq + 2 * gain * dot
    if candidate_sq <= 0:
        raise ValueError("optimizer restoration candidate has zero norm")
    rescale = (
        float(np.sqrt(original_combined_sq / candidate_sq))
        if norm_neutral else 1.0
    )
    final_sq = rescale**2 * candidate_sq
    risk_coefficient = rescale - 1.0
    action_coefficient = rescale * gain
    attributable_sq = (
        risk_coefficient**2 * risk_sq
        + action_coefficient**2 * action_sq
        + 2 * risk_coefficient * action_coefficient * dot
    )
    fraction = float(np.sqrt(max(attributable_sq, 0.0) / final_sq))
    risk_component = float(rescale * (risk_sq + gain * dot))
    return rescale, float(np.sqrt(final_sq)), fraction, risk_component


def arbitrate_optimizer_updates(
    combined_updates: list[torch.Tensor | None],
    risk_only_updates: list[torch.Tensor | None],
    *,
    minimum_attributable_fraction: float,
    maximum_action_gain: float = 2.0,
    norm_neutral: bool = True,
    minimum_risk_component_retention: float = 0.0,
    materialize_updates: bool = True,
) -> UpdateArbitrationResult:
    """Return a bounded optimizer-space action restoration candidate.

    ``combined_updates`` and ``risk_only_updates`` are descent directions
    (parameter value before minus parameter value after) evaluated from the
    same AdamW state.  The returned tensors have the same layout and dtype.
    """
    if not 0 < minimum_attributable_fraction < 1:
        raise ValueError("minimum attributable fraction must be in (0, 1)")
    if maximum_action_gain < 1:
        raise ValueError("maximum action gain must be at least one")
    if not 0 <= minimum_risk_component_retention <= 1:
        raise ValueError("minimum risk-component retention must be in [0, 1]")
    combined, risk = _aligned(combined_updates, risk_only_updates)
    action = [
        None if left is None else left - right
        for left, right in zip(combined, risk)
    ]
    combined_sq, _, _ = _stats(combined, combined)
    action_sq, risk_sq, action_risk_dot = _stats(action, risk)
    if combined_sq <= 0 or action_sq <= 0:
        raise ValueError("combined and action-attributable updates must be non-zero")
    original_fraction = float(np.sqrt(action_sq / combined_sq))
    original_cosine = _cosine(action_sq, risk_sq, action_risk_dot)
    original_risk_component = float(risk_sq + action_risk_dot)
    required_risk_component = float(
        minimum_risk_component_retention * risk_sq
    )
    risk_tolerance = 1e-10 * max(1.0, abs(required_risk_component))
    if (
        original_fraction >= minimum_attributable_fraction
        and original_risk_component >= required_risk_component - risk_tolerance
    ):
        return UpdateArbitrationResult(
            updates=(
                [None if value is None else value.clone() for value in combined]
                if materialize_updates else []
            ),
            action_gain=1.0,
            norm_rescale=1.0,
            original_update_norm=float(np.sqrt(combined_sq)),
            final_update_norm=float(np.sqrt(combined_sq)),
            original_attributable_fraction=original_fraction,
            final_attributable_fraction=original_fraction,
            action_risk_cosine_before=original_cosine,
            action_risk_cosine_after=original_cosine,
            target_reached=True,
            gain_cap_hit=False,
            risk_component_before=original_risk_component,
            risk_component_after=original_risk_component,
            risk_component_retention=(
                float(original_risk_component / risk_sq)
                if risk_sq > 0 else 1.0
            ),
            risk_constraint_active=False,
        )

    # Only remove direct opposition to the protective counterfactual.  This is
    # the update-space analogue of v3's gradient-space PCGrad and cannot reduce
    # the first-order protective descent component.
    projection = action_risk_dot / risk_sq if risk_sq > 0 and action_risk_dot < 0 else 0.0
    safe_action = [
        None if left is None else left - float(projection) * right
        for left, right in zip(action, risk)
    ]
    safe_action_sq, risk_sq, safe_dot = _stats(safe_action, risk)
    if safe_action_sq <= 0:
        raise ValueError("action-attributable update vanished after risk projection")

    target = float(minimum_attributable_fraction)
    high = float(maximum_action_gain)

    def candidate_stats(gain: float) -> tuple[float, float, float, float]:
        return _candidate_stats(
            original_combined_sq=combined_sq,
            action_sq=safe_action_sq,
            risk_sq=risk_sq,
            dot=safe_dot,
            gain=gain,
            norm_neutral=norm_neutral,
        )

    if candidate_stats(high)[2] < target:
        gain = high
        cap_hit = True
    else:
        low = 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if candidate_stats(middle)[2] >= target:
                high = middle
            else:
                low = middle
        gain = high
        cap_hit = bool(np.isclose(gain, maximum_action_gain))

    # A registered restoration may reserve a minimum fraction of the risk-only
    # AdamW descent projection.  Exact preservation would make any
    # norm-neutral amplification of an orthogonal action mathematically
    # impossible, so the caller chooses the explicit floor (V6R uses 0.90).
    # Gain one is the original combined update for non-conflicting residuals;
    # if the requested gain crosses the floor, retain the largest safe gain.
    rescale, final_norm, final_fraction, final_risk_component = candidate_stats(gain)
    risk_constraint_active = False
    if final_risk_component < required_risk_component - risk_tolerance:
        base = candidate_stats(1.0)
        if base[3] < required_risk_component - risk_tolerance:
            # Extremely pathological conflict: the risk-projected path itself
            # is unsafe after norm neutralization.  Preserve the exact original
            # update instead of inventing a weaker protective step.
            return UpdateArbitrationResult(
                updates=(
                    [None if value is None else value.clone() for value in combined]
                    if materialize_updates else []
                ),
                action_gain=1.0,
                norm_rescale=1.0,
                original_update_norm=float(np.sqrt(combined_sq)),
                final_update_norm=float(np.sqrt(combined_sq)),
                original_attributable_fraction=original_fraction,
                final_attributable_fraction=original_fraction,
                action_risk_cosine_before=original_cosine,
                action_risk_cosine_after=original_cosine,
                target_reached=False,
                gain_cap_hit=False,
                risk_component_before=original_risk_component,
                risk_component_after=original_risk_component,
                risk_component_retention=(
                    float(original_risk_component / risk_sq)
                    if risk_sq > 0 else 1.0
                ),
                risk_constraint_active=True,
            )
        low, high = 1.0, gain
        for _ in range(64):
            middle = 0.5 * (low + high)
            if candidate_stats(middle)[3] >= required_risk_component - risk_tolerance:
                low = middle
            else:
                high = middle
        gain = low
        rescale, final_norm, final_fraction, final_risk_component = candidate_stats(gain)
        risk_constraint_active = True
    output = (
        [
            None if left is None else (
                right + float(gain) * left
            ).mul(rescale).to(dtype=original.dtype)
            for left, right, original in zip(safe_action, risk, combined)
        ]
        if materialize_updates else []
    )
    safe_cosine = _cosine(safe_action_sq, risk_sq, safe_dot)
    return UpdateArbitrationResult(
        updates=output,
        action_gain=float(gain),
        norm_rescale=rescale,
        original_update_norm=float(np.sqrt(combined_sq)),
        final_update_norm=final_norm,
        original_attributable_fraction=original_fraction,
        final_attributable_fraction=final_fraction,
        action_risk_cosine_before=original_cosine,
        action_risk_cosine_after=safe_cosine,
        target_reached=bool(final_fraction >= target - 1e-10),
        gain_cap_hit=cap_hit,
        risk_component_before=original_risk_component,
        risk_component_after=final_risk_component,
        risk_component_retention=(
            float(final_risk_component / risk_sq) if risk_sq > 0 else 1.0
        ),
        risk_constraint_active=risk_constraint_active,
    )


def arbitrate_optimizer_updates_by_group(
    combined_updates: list[torch.Tensor | None],
    risk_only_updates: list[torch.Tensor | None],
    parameter_group_positions: dict[str, list[int]],
    *,
    minimum_attributable_fraction: float,
    maximum_action_gain: float,
    minimum_risk_component_retention: float,
    materialize_updates: bool = True,
) -> GroupwiseUpdateArbitrationResult:
    """Restore action attribution independently in every optimizer group.

    Head and backbone use different learning rates and AdamW moments.  A single
    global gain can therefore hide a weak backbone behind a healthy projection
    head.  Per-group norm neutrality preserves both registered update budgets,
    while the complete partition also preserves the global update norm.
    """
    if len(combined_updates) != len(risk_only_updates):
        raise ValueError("combined and risk-only update layouts must align")
    expected = set(range(len(combined_updates)))
    observed: list[int] = [
        int(position)
        for positions in parameter_group_positions.values()
        for position in positions
    ]
    if not parameter_group_positions or set(observed) != expected or len(observed) != len(expected):
        raise ValueError("parameter groups must partition every update exactly once")

    output: list[torch.Tensor | None] = [None] * len(combined_updates)
    reports: dict[str, UpdateArbitrationResult] = {}
    for name in sorted(parameter_group_positions):
        positions = list(map(int, parameter_group_positions[name]))
        report = arbitrate_optimizer_updates(
            [combined_updates[position] for position in positions],
            [risk_only_updates[position] for position in positions],
            minimum_attributable_fraction=minimum_attributable_fraction,
            maximum_action_gain=maximum_action_gain,
            norm_neutral=True,
            minimum_risk_component_retention=minimum_risk_component_retention,
            materialize_updates=materialize_updates,
        )
        reports[name] = report
        if materialize_updates:
            for position, update in zip(positions, report.updates):
                output[position] = update

    original_norm = float(np.sqrt(sum(
        report.original_update_norm**2 for report in reports.values()
    )))
    final_norm = float(np.sqrt(sum(
        report.final_update_norm**2 for report in reports.values()
    )))
    if not np.isclose(original_norm, final_norm, rtol=1e-6, atol=1e-10):
        raise RuntimeError("groupwise restoration changed the global update norm")
    return GroupwiseUpdateArbitrationResult(
        updates=output if materialize_updates else [],
        parameter_groups=reports,
        original_update_norm=original_norm,
        final_update_norm=final_norm,
        all_groups_target_reached=all(
            report.target_reached for report in reports.values()
        ),
        minimum_group_risk_component_retention=min(
            report.risk_component_retention for report in reports.values()
        ),
    )


def _corrective_candidate_stats(
    *,
    original_combined_sq: float,
    baseline_sq: float,
    corrective_sq: float,
    baseline_corrective_dot: float,
    baseline_protective_dot: float,
    corrective_protective_dot: float,
    protective_sq: float,
    gain: float,
    norm_neutral: bool,
) -> tuple[float, float, float, float]:
    """Return candidate statistics for a corrective-only residual.

    The attributable fraction is measured against the *unscaled* noncorrective
    baseline.  Safety, however, is measured against the independent protective
    update.  Keeping these two references separate is the essential distinction
    from the legacy composite-action restoration.
    """
    gain = float(gain)
    candidate_sq = (
        baseline_sq
        + gain**2 * corrective_sq
        + 2.0 * gain * baseline_corrective_dot
    )
    if candidate_sq <= 0:
        raise ValueError("corrective optimizer restoration candidate has zero norm")
    rescale = (
        float(np.sqrt(original_combined_sq / candidate_sq))
        if norm_neutral else 1.0
    )
    final_sq = rescale**2 * candidate_sq
    baseline_coefficient = rescale - 1.0
    corrective_coefficient = rescale * gain
    attributable_sq = (
        baseline_coefficient**2 * baseline_sq
        + corrective_coefficient**2 * corrective_sq
        + 2.0
        * baseline_coefficient
        * corrective_coefficient
        * baseline_corrective_dot
    )
    fraction = float(np.sqrt(max(attributable_sq, 0.0) / final_sq))
    protective_component = float(
        rescale
        * (
            baseline_protective_dot
            + gain * corrective_protective_dot
        )
    )
    return rescale, float(np.sqrt(final_sq)), fraction, protective_component


def arbitrate_corrective_optimizer_updates(
    combined_updates: list[torch.Tensor | None],
    noncorrective_baseline_updates: list[torch.Tensor | None],
    protective_updates: list[torch.Tensor | None],
    *,
    minimum_attributable_fraction: float,
    maximum_corrective_gain: float = 2.0,
    norm_neutral: bool = True,
    minimum_protective_component_retention: float = 0.0,
    materialize_updates: bool = True,
) -> UpdateArbitrationResult:
    """Restore only the optimizer increment caused by adding corrective loss.

    ``combined - noncorrective_baseline`` is the only residual eligible for
    amplification.  ``protective_updates`` is used solely as the safety axis.
    A zero corrective residual is an observed failed target, not a fatal runtime
    error: the original combined update is returned unchanged so held evaluation
    can still complete.
    """
    if not 0 < minimum_attributable_fraction < 1:
        raise ValueError("minimum attributable fraction must be in (0, 1)")
    if maximum_corrective_gain < 1:
        raise ValueError("maximum corrective gain must be at least one")
    if not 0 <= minimum_protective_component_retention <= 1:
        raise ValueError("minimum protective-component retention must be in [0, 1]")

    combined, baseline = _aligned(
        combined_updates, noncorrective_baseline_updates,
    )
    _, protective = _aligned(combined_updates, protective_updates)
    corrective = [
        None if left is None else left - right
        for left, right in zip(combined, baseline)
    ]
    combined_sq, _, _ = _stats(combined, combined)
    corrective_sq, baseline_sq, corrective_baseline_dot = _stats(
        corrective, baseline,
    )
    _, protective_sq, corrective_protective_dot = _stats(
        corrective, protective,
    )
    _, _, baseline_protective_dot = _stats(baseline, protective)
    _, _, combined_protective_dot = _stats(combined, protective)
    original_fraction = (
        float(np.sqrt(corrective_sq / combined_sq))
        if combined_sq > 0 else 0.0
    )
    original_cosine = _cosine(
        corrective_sq, protective_sq, corrective_protective_dot,
    )
    original_retention = (
        float(combined_protective_dot / protective_sq)
        if protective_sq > 0 else 1.0
    )
    if combined_sq <= 0:
        raise ValueError("combined optimizer update must be non-zero")
    if corrective_sq <= 0:
        return UpdateArbitrationResult(
            updates=(
                [None if value is None else value.clone() for value in combined]
                if materialize_updates else []
            ),
            action_gain=1.0,
            norm_rescale=1.0,
            original_update_norm=float(np.sqrt(combined_sq)),
            final_update_norm=float(np.sqrt(combined_sq)),
            original_attributable_fraction=0.0,
            final_attributable_fraction=0.0,
            action_risk_cosine_before=None,
            action_risk_cosine_after=None,
            target_reached=False,
            gain_cap_hit=False,
            risk_component_before=combined_protective_dot,
            risk_component_after=combined_protective_dot,
            risk_component_retention=original_retention,
            risk_constraint_active=False,
        )

    # Only the corrective residual may be modified.  Remove direct opposition
    # to the independent protective reference without altering the auxiliary
    # baseline itself.
    projection = (
        corrective_protective_dot / protective_sq
        if protective_sq > 0 and corrective_protective_dot < 0 else 0.0
    )
    safe_corrective = [
        None if left is None else left - float(projection) * right
        for left, right in zip(corrective, protective)
    ]
    safe_sq, baseline_sq, safe_baseline_dot = _stats(
        safe_corrective, baseline,
    )
    _, protective_sq, safe_protective_dot = _stats(
        safe_corrective, protective,
    )
    if safe_sq <= 0:
        return UpdateArbitrationResult(
            updates=(
                [None if value is None else value.clone() for value in combined]
                if materialize_updates else []
            ),
            action_gain=1.0,
            norm_rescale=1.0,
            original_update_norm=float(np.sqrt(combined_sq)),
            final_update_norm=float(np.sqrt(combined_sq)),
            original_attributable_fraction=original_fraction,
            final_attributable_fraction=original_fraction,
            action_risk_cosine_before=original_cosine,
            action_risk_cosine_after=None,
            target_reached=False,
            gain_cap_hit=False,
            risk_component_before=combined_protective_dot,
            risk_component_after=combined_protective_dot,
            risk_component_retention=original_retention,
            risk_constraint_active=True,
        )

    def candidate_stats(gain: float) -> tuple[float, float, float, float]:
        return _corrective_candidate_stats(
            original_combined_sq=combined_sq,
            baseline_sq=baseline_sq,
            corrective_sq=safe_sq,
            baseline_corrective_dot=safe_baseline_dot,
            baseline_protective_dot=baseline_protective_dot,
            corrective_protective_dot=safe_protective_dot,
            protective_sq=protective_sq,
            gain=gain,
            norm_neutral=norm_neutral,
        )

    target = float(minimum_attributable_fraction)
    high = float(maximum_corrective_gain)
    if candidate_stats(high)[2] < target:
        gain = high
        cap_hit = True
    else:
        low = 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if candidate_stats(middle)[2] >= target:
                high = middle
            else:
                low = middle
        gain = high
        cap_hit = bool(np.isclose(gain, maximum_corrective_gain))

    required_protective_component = float(
        minimum_protective_component_retention * protective_sq
    )
    tolerance = 1e-10 * max(1.0, abs(required_protective_component))
    rescale, final_norm, final_fraction, final_protective_component = (
        candidate_stats(gain)
    )
    risk_constraint_active = False
    if final_protective_component < required_protective_component - tolerance:
        base = candidate_stats(1.0)
        if base[3] < required_protective_component - tolerance:
            return UpdateArbitrationResult(
                updates=(
                    [None if value is None else value.clone() for value in combined]
                    if materialize_updates else []
                ),
                action_gain=1.0,
                norm_rescale=1.0,
                original_update_norm=float(np.sqrt(combined_sq)),
                final_update_norm=float(np.sqrt(combined_sq)),
                original_attributable_fraction=original_fraction,
                final_attributable_fraction=original_fraction,
                action_risk_cosine_before=original_cosine,
                action_risk_cosine_after=_cosine(
                    safe_sq, protective_sq, safe_protective_dot,
                ),
                target_reached=False,
                gain_cap_hit=False,
                risk_component_before=combined_protective_dot,
                risk_component_after=combined_protective_dot,
                risk_component_retention=original_retention,
                risk_constraint_active=True,
            )
        low, high = 1.0, gain
        for _ in range(64):
            middle = 0.5 * (low + high)
            if candidate_stats(middle)[3] >= required_protective_component - tolerance:
                low = middle
            else:
                high = middle
        gain = low
        rescale, final_norm, final_fraction, final_protective_component = (
            candidate_stats(gain)
        )
        risk_constraint_active = True

    output = (
        [
            None if left is None else (
                right + float(gain) * left
            ).mul(rescale).to(dtype=original.dtype)
            for left, right, original in zip(
                safe_corrective, baseline, combined,
            )
        ]
        if materialize_updates else []
    )
    return UpdateArbitrationResult(
        updates=output,
        action_gain=float(gain),
        norm_rescale=rescale,
        original_update_norm=float(np.sqrt(combined_sq)),
        final_update_norm=final_norm,
        original_attributable_fraction=original_fraction,
        final_attributable_fraction=final_fraction,
        action_risk_cosine_before=original_cosine,
        action_risk_cosine_after=_cosine(
            safe_sq, protective_sq, safe_protective_dot,
        ),
        target_reached=bool(final_fraction >= target - 1e-10),
        gain_cap_hit=cap_hit,
        risk_component_before=combined_protective_dot,
        risk_component_after=final_protective_component,
        risk_component_retention=(
            float(final_protective_component / protective_sq)
            if protective_sq > 0 else 1.0
        ),
        risk_constraint_active=risk_constraint_active,
    )


def arbitrate_corrective_optimizer_updates_by_group(
    combined_updates: list[torch.Tensor | None],
    noncorrective_baseline_updates: list[torch.Tensor | None],
    protective_updates: list[torch.Tensor | None],
    parameter_group_positions: dict[str, list[int]],
    *,
    minimum_attributable_fraction: float,
    maximum_corrective_gain: float,
    minimum_protective_component_retention: float,
    materialize_updates: bool = True,
) -> GroupwiseUpdateArbitrationResult:
    """Apply corrective-only restoration independently to every optimizer group."""
    if not (
        len(combined_updates)
        == len(noncorrective_baseline_updates)
        == len(protective_updates)
    ):
        raise ValueError("corrective restoration update layouts must align")
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
    reports: dict[str, UpdateArbitrationResult] = {}
    for name in sorted(parameter_group_positions):
        positions = list(map(int, parameter_group_positions[name]))
        report = arbitrate_corrective_optimizer_updates(
            [combined_updates[position] for position in positions],
            [noncorrective_baseline_updates[position] for position in positions],
            [protective_updates[position] for position in positions],
            minimum_attributable_fraction=minimum_attributable_fraction,
            maximum_corrective_gain=maximum_corrective_gain,
            minimum_protective_component_retention=(
                minimum_protective_component_retention
            ),
            materialize_updates=materialize_updates,
        )
        reports[name] = report
        if materialize_updates:
            for position, update in zip(positions, report.updates):
                output[position] = update
    original_norm = float(np.sqrt(sum(
        report.original_update_norm**2 for report in reports.values()
    )))
    final_norm = float(np.sqrt(sum(
        report.final_update_norm**2 for report in reports.values()
    )))
    if not np.isclose(original_norm, final_norm, rtol=1e-6, atol=1e-10):
        raise RuntimeError("groupwise corrective restoration changed global update norm")
    return GroupwiseUpdateArbitrationResult(
        updates=output if materialize_updates else [],
        parameter_groups=reports,
        original_update_norm=original_norm,
        final_update_norm=final_norm,
        all_groups_target_reached=all(
            report.target_reached for report in reports.values()
        ),
        minimum_group_risk_component_retention=min(
            report.risk_component_retention for report in reports.values()
        ),
    )


@torch.no_grad()
def materialize_descent_updates_(
    parameters: list[torch.nn.Parameter],
    parameters_before_step: list[torch.Tensor],
    descent_updates: list[torch.Tensor | None],
) -> None:
    """Overwrite only parameter values with a registered descent direction.

    AdamW moments have already advanced under the real combined gradient.  This
    function changes the current parameter displacement, not optimizer state,
    learning rates, weight decay, gradient dose or the next step number.
    """
    if not (
        len(parameters) == len(parameters_before_step) == len(descent_updates)
    ):
        raise ValueError("parameter materialization layouts must align")
    for parameter, before, update in zip(
        parameters, parameters_before_step, descent_updates,
    ):
        if before.shape != parameter.shape or before.device != parameter.device:
            raise ValueError("parameter snapshot does not align with live parameter")
        if update is None:
            parameter.copy_(before)
            continue
        if update.shape != parameter.shape or update.device != parameter.device:
            raise ValueError("descent update does not align with live parameter")
        if not torch.isfinite(update).all():
            raise ValueError("descent update contains non-finite values")
        parameter.copy_(before - update.to(dtype=parameter.dtype))


@torch.no_grad()
def reconcile_adamw_first_moments_to_materialized_updates_(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    parameters_before_step: list[torch.Tensor],
    descent_updates: list[torch.Tensor | None],
) -> dict[str, float | int | bool | str]:
    """Make AdamW's stored first moment reproduce the applied displacement.

    The ordinary AdamW step has already advanced ``step`` and the second moment
    from the real combined gradient.  A post-step corrective restoration changes
    the parameter displacement.  Leaving ``exp_avg`` untouched would make the
    optimizer memory describe a different update from the one actually applied.
    This function solves the AdamW update equation for ``exp_avg`` while keeping
    the observed second moment and step number fixed, then verifies the same-step
    reconstruction.  The second moment deliberately remains evidence from the
    real combined gradient rather than being silently rewritten.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("first-moment reconciliation currently requires AdamW")
    if not (
        len(parameters) == len(parameters_before_step) == len(descent_updates)
    ):
        raise ValueError("AdamW reconciliation layouts must align")
    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if set(groups) != {id(parameter) for parameter in parameters}:
        raise RuntimeError("AdamW reconciliation requires every optimizer parameter")

    statistics: list[torch.Tensor] = []
    reconciled = 0
    for parameter, before, update in zip(
        parameters, parameters_before_step, descent_updates,
    ):
        if update is None:
            continue
        group = groups[id(parameter)]
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        raw_step = state.get("step")
        if exp_avg is None or exp_avg_sq is None or raw_step is None:
            raise RuntimeError("AdamW state was not initialized before reconciliation")
        if group.get("differentiable", False) or group.get("maximize", False):
            raise RuntimeError("unsupported AdamW mode for first-moment reconciliation")
        step = int(raw_step.item()) if torch.is_tensor(raw_step) else int(raw_step)
        if step < 1:
            raise RuntimeError("AdamW reconciliation requires an advanced step")
        beta1, beta2 = map(float, group["betas"])
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        if group.get("amsgrad", False):
            denominator_sq = state.get("max_exp_avg_sq")
            if denominator_sq is None:
                raise RuntimeError("AMSGrad state lacks max_exp_avg_sq")
        else:
            denominator_sq = exp_avg_sq
        denominator = denominator_sq.detach().sqrt() / np.sqrt(bias2)
        denominator = denominator.add(float(group["eps"]))
        # Reconcile to the displacement that was actually representable after
        # materialization, rather than to the pre-rounding requested tensor.
        # Both the AdamW decay and adaptive terms are update-scale quantities.
        # Forming ``decayed_parameter - desired_parameter`` subtracts two
        # parameter-scale FP32 tensors and produced a 3.6e-4 relative error on
        # the full model even though the moment change itself was healthy.
        requested_update = update.detach().to(dtype=before.dtype)
        realized_update = before.detach() - parameter.detach()
        decay_descent = before.detach() * (
            float(group["lr"]) * float(group["weight_decay"])
        )
        adaptive_descent = realized_update - decay_descent
        desired_moment = (
            adaptive_descent
            * denominator
            * (bias1 / float(group["lr"]))
        ).to(dtype=exp_avg.dtype)
        original_value = torch.sum(
            exp_avg.detach().float() * exp_avg.detach().float()
        ).double()
        moment_delta = desired_moment.detach().float() - exp_avg.detach().float()
        changed_value = torch.sum(moment_delta * moment_delta).double()
        exp_avg.copy_(desired_moment)

        # Verify the small descent equation directly.  This avoids the same
        # catastrophic cancellation while retaining a strict 1e-6 gate.
        reconstructed_update = decay_descent.clone()
        reconstructed_update.addcdiv_(
            exp_avg,
            denominator,
            value=(float(group["lr"]) / bias1),
        )
        delta = (reconstructed_update - realized_update).float()
        error_value = torch.sum(delta * delta).double()
        target = realized_update.float()
        target_value = torch.sum(target * target).double()

        materialization_delta = (realized_update - requested_update).float()
        materialization_error_value = torch.sum(
            materialization_delta * materialization_delta
        ).double()
        requested = requested_update.float()
        requested_value = torch.sum(requested * requested).double()

        # Retain a diagnostic replay through AdamW's parameter-scale in-place
        # arithmetic.  Its FP32 subtraction error is expected to be larger
        # than the stable equation error and is not mislabeled as state drift.
        replayed_after = before.detach().clone()
        replayed_after.mul_(
            1.0 - float(group["lr"]) * float(group["weight_decay"])
        )
        replayed_after.addcdiv_(
            exp_avg,
            denominator,
            value=-(float(group["lr"]) / bias1),
        )
        replayed_update = before.detach() - replayed_after
        replay_delta = (replayed_update - realized_update).float()
        replay_error_value = torch.sum(replay_delta * replay_delta).double()
        statistics.append(torch.stack((
            error_value,
            target_value,
            materialization_error_value,
            requested_value,
            replay_error_value,
            original_value,
            changed_value,
        )))
        reconciled += 1
    totals = (
        torch.stack(statistics).sum(dim=0).cpu().tolist()
        if statistics else [0.0] * 7
    )
    (
        error_total,
        target_total,
        materialization_error_total,
        requested_total,
        replay_error_total,
        original_moment_total,
        changed_moment_total,
    ) = map(float, totals)
    relative_error = float(
        np.sqrt(error_total / target_total)
        if target_total > 0 else 0.0
    )
    return {
        "reconciled_parameter_tensors": reconciled,
        "same_step_reconstruction_relative_error": relative_error,
        "materialized_update_relative_error": float(
            np.sqrt(materialization_error_total / requested_total)
            if requested_total > 0 else 0.0
        ),
        "same_step_fp32_parameter_replay_relative_error": float(
            np.sqrt(replay_error_total / target_total)
            if target_total > 0 else 0.0
        ),
        "reconstruction_target": "realized_materialized_parameter_displacement",
        "verification_arithmetic": "stable_decay_plus_adaptive_displacement",
        "first_moment_change_fraction": float(
            np.sqrt(changed_moment_total / original_moment_total)
            if original_moment_total > 0 else 0.0
        ),
        "second_moment_source": "actual_combined_gradient",
        "step_number_preserved": True,
        "gate_passed": bool(relative_error <= 1e-6),
    }
