"""Norm-neutral repair of AdamW update/action directional loss.

This is a pure candidate for a future direct-v4 canary.  It minimally rotates
an observed AdamW descent update toward the already risk-projected action
gradient.  The update norm is held fixed and a conflicting action component is
removed against the protective gradient before any rotation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DirectionalUpdateResult:
    updates: list[torch.Tensor | None]
    action_coefficient: float
    norm_rescale: float
    update_norm_before: float
    update_norm_after: float
    action_alignment_before: float
    action_alignment_after: float
    risk_alignment_before: float | None
    risk_alignment_after: float | None
    target_reached: bool
    coefficient_cap_hit: bool
    risk_constraint_active: bool


def _aligned(
    updates: list[torch.Tensor | None],
    action: list[torch.Tensor | None],
    risk: list[torch.Tensor | None] | None,
) -> tuple[
    list[torch.Tensor | None],
    list[torch.Tensor | None],
    list[torch.Tensor | None],
]:
    if len(updates) != len(action) or (risk is not None and len(risk) != len(updates)):
        raise ValueError("update/action/risk layouts must align")
    output_update, output_action, output_risk = [], [], []
    for index, (left, right) in enumerate(zip(updates, action)):
        protect = None if risk is None else risk[index]
        if left is None:
            if right is not None or protect is not None:
                raise ValueError("gradient exists where optimizer update is absent")
            output_update.append(None)
            output_action.append(None)
            output_risk.append(None)
            continue
        if right is None:
            right = torch.zeros_like(left)
        if protect is None:
            protect = torch.zeros_like(left)
        if right.shape != left.shape or protect.shape != left.shape:
            raise ValueError("update/action/risk tensors must have equal shapes")
        if not all(torch.isfinite(value).all() for value in (left, right, protect)):
            raise ValueError("update arbitration received a non-finite tensor")
        output_update.append(left.detach())
        output_action.append(right.detach())
        output_risk.append(protect.detach())
    if not any(value is not None for value in output_update):
        raise ValueError("at least one optimizer update is required")
    return output_update, output_action, output_risk


def _stats(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> tuple[float, float, float]:
    left_sq = right_sq = dot = 0.0
    for x, y in zip(left, right):
        if x is None or y is None:
            continue
        xf, yf = x.float(), y.float()
        left_sq += float(torch.sum(xf * xf))
        right_sq += float(torch.sum(yf * yf))
        dot += float(torch.sum(xf * yf))
    return left_sq, right_sq, dot


def _cosine(left_sq: float, right_sq: float, dot: float) -> float | None:
    if left_sq <= 0 or right_sq <= 0:
        return None
    return float(dot / np.sqrt(left_sq * right_sq))


def rotate_update_toward_action(
    actual_updates: list[torch.Tensor | None],
    action_gradients: list[torch.Tensor | None],
    *,
    minimum_action_alignment: float,
    maximum_action_coefficient: float = 0.50,
    risk_gradients: list[torch.Tensor | None] | None = None,
    materialize_updates: bool = True,
) -> DirectionalUpdateResult:
    """Minimally rotate an optimizer update while preserving its exact norm."""
    if not -1 < minimum_action_alignment < 1:
        raise ValueError("minimum action alignment must be in (-1, 1)")
    if maximum_action_coefficient < 0:
        raise ValueError("maximum action coefficient must be non-negative")
    update, action, risk = _aligned(actual_updates, action_gradients, risk_gradients)
    update_sq, action_sq, update_action_dot = _stats(update, action)
    if update_sq <= 0 or action_sq <= 0:
        raise ValueError("optimizer update and action gradient must be non-zero")
    risk_sq, _, _ = _stats(risk, risk)
    action_risk = _stats(action, risk)[2]
    projection = action_risk / risk_sq if risk_sq > 0 and action_risk < 0 else 0.0
    safe_action = [
        None if left is None else left - float(projection) * right
        for left, right in zip(action, risk)
    ]
    safe_action_sq = _stats(safe_action, safe_action)[0]
    if safe_action_sq <= 0:
        raise ValueError("action gradient vanished after risk projection")
    update_action_dot = _stats(update, safe_action)[2]
    alignment_before = _cosine(update_sq, safe_action_sq, update_action_dot)
    assert alignment_before is not None
    update_risk_dot = _stats(update, risk)[2]
    risk_before = _cosine(update_sq, risk_sq, update_risk_dot)
    if alignment_before >= minimum_action_alignment:
        return DirectionalUpdateResult(
            updates=(
                [None if value is None else value.clone() for value in update]
                if materialize_updates else []
            ),
            action_coefficient=0.0,
            norm_rescale=1.0,
            update_norm_before=float(np.sqrt(update_sq)),
            update_norm_after=float(np.sqrt(update_sq)),
            action_alignment_before=alignment_before,
            action_alignment_after=alignment_before,
            risk_alignment_before=risk_before,
            risk_alignment_after=risk_before,
            target_reached=True,
            coefficient_cap_hit=False,
            risk_constraint_active=False,
        )

    # Coefficient is expressed in units of the original update norm, so its
    # interpretation is stable across head/backbone learning rates.
    unit_scale = float(np.sqrt(update_sq / safe_action_sq))

    def alignment(coefficient: float) -> float:
        numerator = update_action_dot + coefficient * unit_scale * safe_action_sq
        candidate_sq = (
            update_sq
            + (coefficient * unit_scale) ** 2 * safe_action_sq
            + 2 * coefficient * unit_scale * update_action_dot
        )
        return float(numerator / np.sqrt(candidate_sq * safe_action_sq))

    high = float(maximum_action_coefficient)
    if alignment(high) < minimum_action_alignment:
        coefficient = high
        cap_hit = True
    else:
        low = 0.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if alignment(middle) >= minimum_action_alignment:
                high = middle
            else:
                low = middle
        coefficient = high
        cap_hit = bool(np.isclose(coefficient, maximum_action_coefficient))

    safe_action_risk_dot = _stats(safe_action, risk)[2]

    def protected_risk_component(value: float) -> float:
        candidate_sq = (
            update_sq
            + (value * unit_scale) ** 2 * safe_action_sq
            + 2 * value * unit_scale * update_action_dot
        )
        local_rescale = float(np.sqrt(update_sq / candidate_sq))
        return local_rescale * (
            update_risk_dot + value * unit_scale * safe_action_risk_dot
        )

    risk_limited = False
    if (
        risk_sq > 0
        and protected_risk_component(coefficient) < update_risk_dot - 1e-12
    ):
        # Zero rotation is exactly safe.  Find the largest coefficient on the
        # path to the requested action alignment that does not reduce the
        # original first-order protective descent component.
        low, high = 0.0, coefficient
        for _ in range(64):
            middle = 0.5 * (low + high)
            if protected_risk_component(middle) >= update_risk_dot - 1e-12:
                low = middle
            else:
                high = middle
        coefficient = low
        risk_limited = True
    candidate_sq = (
        update_sq
        + (coefficient * unit_scale) ** 2 * safe_action_sq
        + 2 * coefficient * unit_scale * update_action_dot
    )
    rescale = float(np.sqrt(update_sq / candidate_sq))
    output = (
        [
            None if left is None else (
                left + coefficient * unit_scale * right
            ).mul(rescale).to(dtype=original.dtype)
            for left, right, original in zip(update, safe_action, update)
        ]
        if materialize_updates else []
    )
    final_sq = update_sq
    final_action_dot = rescale * (
        update_action_dot + coefficient * unit_scale * safe_action_sq
    )
    alignment_after = _cosine(final_sq, safe_action_sq, final_action_dot)
    assert alignment_after is not None
    final_risk_dot = rescale * (
        update_risk_dot + coefficient * unit_scale * safe_action_risk_dot
    )
    risk_after = _cosine(final_sq, risk_sq, final_risk_dot)
    if risk_sq > 0 and final_risk_dot < update_risk_dot - 1e-6:
        raise RuntimeError("action rotation reduced the protective descent component")
    return DirectionalUpdateResult(
        updates=output,
        action_coefficient=float(coefficient),
        norm_rescale=rescale,
        update_norm_before=float(np.sqrt(update_sq)),
        update_norm_after=float(np.sqrt(final_sq)),
        action_alignment_before=alignment_before,
        action_alignment_after=alignment_after,
        risk_alignment_before=risk_before,
        risk_alignment_after=risk_after,
        target_reached=bool(alignment_after >= minimum_action_alignment - 1e-7),
        coefficient_cap_hit=cap_hit,
        risk_constraint_active=risk_limited,
    )
