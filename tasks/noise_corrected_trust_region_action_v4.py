"""Minimal-dose, gradient-defined direct actions for noise fine-tuning.

Unlike a fixed recipe that edits every selected peak by 50%, this action spends
only the intensity-change budget predicted to close the current clean-query
margin deficit.  It may attenuate non-identity adverse peaks and boost existing
identity/shared supported peaks.  No embedding vector is used as a teacher.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from noise_v3_core import IDENTITY_ONLY, SHARED


@dataclass(frozen=True)
class TrustRegionAction:
    attenuation_tokens: tuple[int, ...]
    attenuation_fractions: tuple[float, ...]
    boost_tokens: tuple[int, ...]
    boost_fractions: tuple[float, ...]
    predicted_gain: float
    total_fractional_dose: float
    target_gain: float
    target_reached: bool


def _fragment_mask(clean: torch.Tensor) -> np.ndarray:
    values = clean.detach().cpu().numpy()
    return (
        (np.arange(len(values)) > 0)
        & (values[:, 0] > 0)
        & (values[:, 1] > 0)
    )


def minimal_supported_margin_action(
    clean: torch.Tensor,
    intensity_gradient: np.ndarray,
    roles: np.ndarray,
    *,
    target_gain: float,
    maximum_peaks: int = 6,
    maximum_fraction_per_peak: float = 0.75,
    maximum_total_fraction: float = 2.0,
    directions: frozenset[str] = frozenset({"down", "up"}),
) -> TrustRegionAction:
    """Allocate the smallest first-order dose toward a positive margin gain."""
    if target_gain <= 0:
        raise ValueError("target gain must be positive")
    if maximum_peaks < 1:
        raise ValueError("maximum peaks must be positive")
    if not 0 < maximum_fraction_per_peak <= 1 or maximum_total_fraction <= 0:
        raise ValueError("trust-region dose bounds are invalid")
    if not directions or not directions <= {"down", "up"}:
        raise ValueError("trust-region directions must be down and/or up")
    values = clean.detach().cpu().numpy()
    gradient = np.asarray(intensity_gradient, dtype=float)
    role = np.asarray(roles, dtype=np.int8)
    if gradient.shape != (len(clean),) or role.shape != (len(clean),):
        raise ValueError("gradient/role vectors must align with spectrum tokens")
    valid = _fragment_mask(clean)
    intensity = values[:, 1].astype(float)
    candidates: list[tuple[float, str, int]] = []
    for token in np.flatnonzero(valid):
        token = int(token)
        down_gain = -intensity[token] * gradient[token]
        if (
            "down" in directions and role[token] != IDENTITY_ONLY
            and np.isfinite(down_gain) and down_gain > 0
        ):
            candidates.append((float(down_gain), "down", token))
        up_gain = intensity[token] * gradient[token]
        if (
            "up" in directions and role[token] in {IDENTITY_ONLY, SHARED}
            and np.isfinite(up_gain) and up_gain > 0
        ):
            candidates.append((float(up_gain), "up", token))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    down: list[tuple[int, float]] = []
    up: list[tuple[int, float]] = []
    predicted = 0.0
    total = 0.0
    used: set[int] = set()
    for unit_gain, direction, token in candidates:
        if len(used) >= maximum_peaks or total >= maximum_total_fraction - 1e-12:
            break
        if token in used:
            continue
        remaining_gain = max(float(target_gain) - predicted, 0.0)
        if remaining_gain <= 1e-12:
            break
        dose = min(
            float(maximum_fraction_per_peak),
            float(maximum_total_fraction) - total,
            remaining_gain / unit_gain,
        )
        if dose <= 1e-12:
            continue
        (down if direction == "down" else up).append((token, float(dose)))
        used.add(token)
        predicted += unit_gain * float(dose)
        total += float(dose)
    return TrustRegionAction(
        attenuation_tokens=tuple(token for token, _ in down),
        attenuation_fractions=tuple(dose for _, dose in down),
        boost_tokens=tuple(token for token, _ in up),
        boost_fractions=tuple(dose for _, dose in up),
        predicted_gain=float(predicted),
        total_fractional_dose=float(total),
        target_gain=float(target_gain),
        target_reached=bool(predicted >= float(target_gain) - 1e-8),
    )


def apply_trust_region_action(
    clean: torch.Tensor,
    action: TrustRegionAction,
) -> torch.Tensor:
    """Apply the frozen variable-dose recipe and normalize fragments once."""
    output = clean.clone()
    selected = action.attenuation_tokens + action.boost_tokens
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("trust-region action must contain unique edited peaks")
    for token, dose in zip(
        action.attenuation_tokens, action.attenuation_fractions,
    ):
        if token <= 0 or token >= len(clean) or not 0 < dose <= 1:
            raise ValueError("invalid attenuation edit")
        output[token, 1] *= 1.0 - float(dose)
    for token, dose in zip(action.boost_tokens, action.boost_fractions):
        if token <= 0 or token >= len(clean) or dose <= 0:
            raise ValueError("invalid boost edit")
        output[token, 1] *= 1.0 + float(dose)
    maximum = output[1:, 1].max()
    if maximum > 0:
        output[1:, 1] /= maximum
    output[0] = clean[0]
    return output
