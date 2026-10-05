"""Pure helpers for bounded multi-peak noise actions.

These actions extend the existing single-peak A4 and greedy sequential N
families without changing the deployed model.  They operate on real query
peaks, preserve the precursor token, and restore the same fragment maximum
normalization used by the existing action executors.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from noise_v3_core import IDENTITY_ONLY, SHARED


SUPPORTED_BOOST_ROLES = frozenset({IDENTITY_ONLY, SHARED})


def _real_fragment_mask(clean: torch.Tensor) -> np.ndarray:
    values = clean.detach().cpu().numpy()
    return (
        (np.arange(len(values)) > 0)
        & (values[:, 0] > 0)
        & (values[:, 1] > 0)
    )


def _tokens(values: Iterable[int], clean: torch.Tensor) -> tuple[int, ...]:
    output = tuple(map(int, values))
    if len(output) != len(set(output)):
        raise ValueError("multi-peak action tokens must be unique")
    valid = _real_fragment_mask(clean)
    if any(token <= 0 or token >= len(clean) or not valid[token] for token in output):
        raise ValueError("multi-peak action selected precursor, padding or an absent peak")
    return output


def _renormalize(clean: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    fragments = output[1:]
    maximum = fragments[:, 1].max()
    if maximum > 0:
        fragments[:, 1] /= maximum
    output[0] = clean[0]
    return output


def rank_attenuation_tokens(
    clean: torch.Tensor,
    intensity_gradient: np.ndarray,
    roles: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Rank non-identity peaks whose attenuation has positive first-order gain."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    values = clean.detach().cpu().numpy()
    gradient = np.asarray(intensity_gradient, dtype=float)
    roles = np.asarray(roles, dtype=np.int8)
    if gradient.shape != (len(clean),) or roles.shape != (len(clean),):
        raise ValueError("gradient/role vectors must align with spectrum tokens")
    gain = -values[:, 1].astype(float) * gradient
    valid = _real_fragment_mask(clean) & (roles != IDENTITY_ONLY)
    choices = np.flatnonzero(valid & np.isfinite(gain) & (gain > 0))
    order = np.lexsort((choices, -gain[choices]))[:maximum]
    return choices[order].astype(np.int64)


def rank_supported_boost_tokens(
    clean: torch.Tensor,
    intensity_gradient: np.ndarray,
    roles: np.ndarray,
    maximum: int,
) -> np.ndarray:
    """Rank positive-gradient peaks observed in the true or shared support."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    values = clean.detach().cpu().numpy()
    gradient = np.asarray(intensity_gradient, dtype=float)
    roles = np.asarray(roles, dtype=np.int8)
    if gradient.shape != (len(clean),) or roles.shape != (len(clean),):
        raise ValueError("gradient/role vectors must align with spectrum tokens")
    gain = values[:, 1].astype(float) * gradient
    supported = np.isin(roles, np.asarray(sorted(SUPPORTED_BOOST_ROLES)))
    valid = _real_fragment_mask(clean) & supported
    choices = np.flatnonzero(valid & np.isfinite(gain) & (gain > 0))
    order = np.lexsort((choices, -gain[choices]))[:maximum]
    return choices[order].astype(np.int64)


def attenuate_tokens_and_renormalize(
    clean: torch.Tensor,
    tokens: Iterable[int],
    attenuation: float,
) -> torch.Tensor:
    """Apply one joint attenuation and normalize once after all peak edits."""
    if not 0 < attenuation <= 1:
        raise ValueError("attenuation must be in (0, 1]")
    selected = _tokens(tokens, clean)
    if not selected:
        raise ValueError("at least one attenuation token is required")
    output = clean.clone()
    for token in selected:
        if attenuation == 1:
            output[token] = 0
        else:
            output[token, 1] *= 1.0 - attenuation
    return _renormalize(clean, output)


def boost_tokens_and_renormalize(
    clean: torch.Tensor,
    tokens: Iterable[int],
    boost: float,
) -> torch.Tensor:
    """Multiplicatively boost existing supported peaks and normalize once."""
    if boost <= 0:
        raise ValueError("boost must be positive")
    selected = _tokens(tokens, clean)
    if not selected:
        raise ValueError("at least one boost token is required")
    output = clean.clone()
    for token in selected:
        output[token, 1] *= 1.0 + boost
    return _renormalize(clean, output)


def signed_multiplicative_action(
    clean: torch.Tensor,
    attenuation_tokens: Iterable[int],
    boost_tokens: Iterable[int],
    *,
    attenuation: float,
    boost: float,
) -> torch.Tensor:
    """Jointly attenuate adverse peaks and boost true/shared supported peaks."""
    down = _tokens(attenuation_tokens, clean)
    up = _tokens(boost_tokens, clean)
    if not down or not up or set(down).intersection(up):
        raise ValueError("signed action requires disjoint non-empty down/up tokens")
    if not 0 < attenuation <= 1 or boost <= 0:
        raise ValueError("signed action doses are invalid")
    output = clean.clone()
    for token in down:
        output[token, 1] *= 1.0 - attenuation
    for token in up:
        output[token, 1] *= 1.0 + boost
    return _renormalize(clean, output)


def conservative_intensity_exchange(
    clean: torch.Tensor,
    attenuation_tokens: Iterable[int],
    boost_tokens: Iterable[int],
    *,
    attenuation: float,
) -> torch.Tensor:
    """Move, rather than create, intensity from adverse to supported peaks.

    Before max-normalization, the sum of fragment intensities is preserved.
    Removed intensity is distributed among supported peaks in proportion to
    their original intensities, with an equal fallback for numerical safety.
    """
    down = _tokens(attenuation_tokens, clean)
    up = _tokens(boost_tokens, clean)
    if not down or not up or set(down).intersection(up):
        raise ValueError("exchange requires disjoint non-empty down/up tokens")
    if not 0 < attenuation <= 1:
        raise ValueError("attenuation must be in (0, 1]")
    output = clean.clone()
    removed = sum(float(output[token, 1]) * attenuation for token in down)
    for token in down:
        output[token, 1] *= 1.0 - attenuation
    weights = torch.stack([output[token, 1] for token in up])
    if float(weights.sum()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()
    for token, weight in zip(up, weights):
        output[token, 1] += removed * weight
    return _renormalize(clean, output)


def smooth_monotone_transfer_delta(
    delta_uncapped: torch.Tensor,
    *,
    scale: float,
    asymptote: float,
) -> torch.Tensor:
    """Bound a positive transfer without making every strong action identical."""
    if scale <= 0 or asymptote <= 0:
        raise ValueError("smooth transfer scale/asymptote must be positive")
    positive = torch.relu(delta_uncapped.detach())
    return float(asymptote) * (1.0 - torch.exp(-positive / float(scale)))


def mass_matched_monotone_transfer_delta(
    delta_uncapped: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    hard_cap: float,
    maximum_cap_factor: float = 2.0,
) -> torch.Tensor:
    """Preserve old transfer mass while retaining within-query action order.

    The v3 hard cap maps every sufficiently strong action edge to the same
    target.  This helper uses log-compressed positive gains as allocation
    weights, preserves the exact sum of the old hard-capped targets, keeps
    inactive edges at zero and limits any one edge to ``factor * hard_cap``.
    It is intended as a separately tested v4 candidate, not a silent change to
    the running v3 experiment.
    """
    if hard_cap <= 0 or maximum_cap_factor < 1:
        raise ValueError("transfer cap must be positive and factor at least one")
    if delta_uncapped.shape != active_mask.shape:
        raise ValueError("transfer delta and active mask must align")
    positive = torch.relu(delta_uncapped.detach())
    active = active_mask.to(dtype=torch.bool) & (positive > 0)
    output = torch.zeros_like(positive)
    if not bool(active.any()):
        return output
    old = positive.clamp_max(float(hard_cap)) * active.to(positive.dtype)
    remaining_mass = old.sum()
    weights = torch.log1p(positive / float(hard_cap)) * active.to(positive.dtype)
    remaining = active.clone()
    ceiling = float(maximum_cap_factor) * float(hard_cap)
    # Water filling is deterministic and terminates after at most N saturated
    # edges.  All operands are detached training targets in the caller.
    for _ in range(int(active.sum().item()) + 1):
        if not bool(remaining.any()) or float(remaining_mass) <= 1e-12:
            break
        local_weights = weights * remaining.to(weights.dtype)
        denominator = local_weights.sum()
        if float(denominator) <= 0:
            local_weights = remaining.to(weights.dtype)
            denominator = local_weights.sum()
        proposal = remaining_mass * local_weights / denominator
        saturated = remaining & (proposal > ceiling)
        if not bool(saturated.any()):
            output = output + proposal * remaining.to(output.dtype)
            remaining_mass = remaining_mass * 0
            break
        output[saturated] = ceiling
        remaining_mass = remaining_mass - ceiling * saturated.sum().to(remaining_mass.dtype)
        remaining = remaining & ~saturated
    if abs(float(output.sum() - old.sum())) > 1e-5 * max(float(old.sum()), 1.0):
        raise RuntimeError("mass-matched transfer allocation failed")
    return output


def mass_matched_monotone_transfer_by_group(
    delta_uncapped: torch.Tensor,
    active_mask: torch.Tensor,
    groups: Iterable[str],
    *,
    hard_cap: float,
    maximum_cap_factor: float = 2.0,
) -> torch.Tensor:
    """Apply dose-neutral allocation inside each mechanism/source/family leaf."""
    labels = tuple(map(str, groups))
    if delta_uncapped.ndim != 2 or delta_uncapped.shape != active_mask.shape:
        raise ValueError("grouped transfer expects aligned action-by-edge matrices")
    if delta_uncapped.shape[0] != len(labels) or not labels:
        raise ValueError("group labels must align with action rows")

    def parse(label: str) -> tuple[str, str, str]:
        mechanism, leaf = label.split("::", 1) if "::" in label else (label, label)
        source, family = leaf.split("|", 1) if "|" in leaf else (leaf, leaf)
        return mechanism, source, family

    parsed = tuple(parse(label) for label in labels)
    output = torch.zeros_like(delta_uncapped)
    for leaf in sorted(set(parsed)):
        row_mask = torch.as_tensor(
            [value == leaf for value in parsed],
            device=delta_uncapped.device,
            dtype=torch.bool,
        )
        output[row_mask] = mass_matched_monotone_transfer_delta(
            delta_uncapped[row_mask], active_mask[row_mask],
            hard_cap=hard_cap, maximum_cap_factor=maximum_cap_factor,
        )
    return output
