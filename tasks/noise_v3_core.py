"""Pure helpers for candidate-conditioned directional noise v3."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


IDENTITY_ONLY = 0
CONFOUNDER_ONLY = 1
SHARED = 2
UNMATCHED = 3
ROLE_NAMES = np.asarray(
    ["identity_only", "confounder_only", "shared", "unmatched"], dtype=object
)


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _matched(query_mz: np.ndarray, reference_mz: np.ndarray, tolerance: float) -> np.ndarray:
    query_mz = np.asarray(query_mz, dtype=float)
    reference_mz = np.sort(np.asarray(reference_mz, dtype=float))
    reference_mz = reference_mz[reference_mz > 0]
    if not len(reference_mz):
        return np.zeros(len(query_mz), dtype=bool)
    positions = np.searchsorted(reference_mz, query_mz)
    left = reference_mz[np.clip(positions - 1, 0, len(reference_mz) - 1)]
    right = reference_mz[np.clip(positions, 0, len(reference_mz) - 1)]
    distance = np.minimum(np.abs(query_mz - left), np.abs(query_mz - right))
    return (query_mz > 0) & (distance <= tolerance)


def candidate_peak_roles(
    query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, tolerance: float = 0.02,
) -> np.ndarray:
    """Role for each query token. Precursor/padding are marked -1."""
    q = query.detach().cpu().numpy()
    p = positive.detach().cpu().numpy()
    n = negative.detach().cpu().numpy()
    valid = (q[:, 0] > 0) & (q[:, 1] > 0)
    valid[0] = False
    p_mz = p[1:, 0][(p[1:, 0] > 0) & (p[1:, 1] > 0)]
    n_mz = n[1:, 0][(n[1:, 0] > 0) & (n[1:, 1] > 0)]
    return candidate_peak_roles_from_mz(query, p_mz, n_mz, tolerance)


def candidate_peak_roles_from_mz(
    query: torch.Tensor,
    positive_mz: np.ndarray,
    negative_mz: np.ndarray,
    tolerance: float = 0.02,
) -> np.ndarray:
    """Roles against unions of multiple positive/negative reference spectra."""
    q = query.detach().cpu().numpy()
    valid = (q[:, 0] > 0) & (q[:, 1] > 0)
    valid[0] = False
    p_mz = np.asarray(positive_mz, dtype=float)
    n_mz = np.asarray(negative_mz, dtype=float)
    hit_p = _matched(q[:, 0], p_mz, tolerance) & valid
    hit_n = _matched(q[:, 0], n_mz, tolerance) & valid
    roles = np.full(len(q), -1, dtype=np.int8)
    roles[valid & hit_p & ~hit_n] = IDENTITY_ONLY
    roles[valid & ~hit_p & hit_n] = CONFOUNDER_ONLY
    roles[valid & hit_p & hit_n] = SHARED
    roles[valid & ~hit_p & ~hit_n] = UNMATCHED
    return roles


def predicted_gain(intensity: np.ndarray, gradient: np.ndarray, attenuation: float) -> np.ndarray:
    """First-order gain in margin when intensity is multiplied by 1-attenuation."""
    return -float(attenuation) * np.asarray(intensity, float) * np.asarray(gradient, float)


def select_gradient_target(
    clean: torch.Tensor,
    intensity_gradient: np.ndarray,
    roles: np.ndarray,
    attenuation: float,
    protect_identity: bool = True,
) -> int | None:
    targets = rank_gradient_targets(
        clean, intensity_gradient, roles, attenuation,
        max_targets=1, protect_identity=protect_identity,
    )
    return int(targets[0]) if len(targets) else None


def rank_gradient_targets(
    clean: torch.Tensor,
    intensity_gradient: np.ndarray,
    roles: np.ndarray,
    attenuation: float,
    max_targets: int,
    protect_identity: bool = True,
) -> np.ndarray:
    """Rank distinct positive-gain fragment actions with deterministic ties."""
    if max_targets < 1:
        raise ValueError("max_targets must be positive")
    values = clean.detach().cpu().numpy()
    gain = predicted_gain(values[:, 1], intensity_gradient, attenuation)
    valid = (values[:, 0] > 0) & (values[:, 1] > 0) & (roles >= 0)
    valid[0] = False
    if protect_identity:
        valid &= roles != IDENTITY_ONLY
    choices = np.flatnonzero(valid & np.isfinite(gain) & (gain > 0))
    if not len(choices):
        return np.empty(0, dtype=np.int64)
    # Primary: descending gain. Secondary: stable ascending token index.
    order = np.lexsort((choices, -gain[choices]))[:max_targets]
    return choices[order].astype(np.int64)


def select_role_only_target(clean: torch.Tensor, roles: np.ndarray) -> int | None:
    """Original hypothesis control: strongest confounder-only query peak."""
    return select_role_target(clean, roles, CONFOUNDER_ONLY)


def select_role_target(
    clean: torch.Tensor, roles: np.ndarray, role: int,
) -> int | None:
    """Select the strongest real fragment with one preregistered candidate role.

    This is deliberately independent of the outcome label and of gradients.  It
    supports both the confounder-only intervention and the identity-only
    direction control while using exactly the same tie-breaking rule.
    """
    targets = rank_role_targets(clean, roles, role, max_targets=1)
    return int(targets[0]) if len(targets) else None


def rank_role_targets(
    clean: torch.Tensor, roles: np.ndarray, role: int, max_targets: int,
) -> np.ndarray:
    """Rank real fragments of one role by intensity with stable token ties."""
    if role not in {IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED, UNMATCHED}:
        raise ValueError(f"unknown peak role: {role}")
    if max_targets < 1:
        raise ValueError("max_targets must be positive")
    values = clean.detach().cpu().numpy()
    choices = np.flatnonzero(
        (roles == role) & (values[:, 0] > 0) & (values[:, 1] > 0)
        & (np.arange(len(values)) > 0)
    )
    if not len(choices):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((choices, -values[choices, 1]))[:max_targets]
    return choices[order].astype(np.int64)


def attenuate_and_renormalize(
    clean: torch.Tensor, token: int, attenuation: float,
) -> torch.Tensor:
    """Attenuate exactly one fragment and restore fragment max-normalization."""
    if not 0 < attenuation <= 1:
        raise ValueError("attenuation must be in (0, 1]")
    if token <= 0 or token >= len(clean):
        raise ValueError("precursor/padding cannot be selected")
    output = clean.clone()
    if output[token, 0] <= 0 or output[token, 1] <= 0:
        raise ValueError("selected token is not a real fragment")
    if attenuation == 1:
        output[token] = 0
    else:
        output[token, 1] *= 1.0 - attenuation
    fragments = output[1:]
    maximum = fragments[:, 1].max()
    if maximum > 0:
        fragments[:, 1] /= maximum
    output[0, 1] = clean[0, 1]
    return output


def matched_control_tokens(
    clean: torch.Tensor,
    target: int,
    roles: np.ndarray,
    repeats: int,
    seed: int,
    same_role: bool = True,
) -> np.ndarray:
    """Deterministic intensity/mz-matched controls, without replacement."""
    values = clean.detach().cpu().numpy()
    valid = np.flatnonzero(
        (values[:, 0] > 0) & (values[:, 1] > 0) & (np.arange(len(values)) > 0)
    )
    valid = valid[valid != target]
    if same_role:
        role_pool = valid[roles[valid] == roles[target]]
        if len(role_pool) >= repeats:
            valid = role_pool
    if not len(valid):
        return np.empty(0, dtype=np.int64)
    intensity = np.log(np.clip(values[:, 1].astype(float), 1e-8, None))
    mz_scale = max(float(np.std(values[valid, 0])), 25.0)
    cost = (
        4.0 * np.abs(intensity[valid] - intensity[target])
        + 0.15 * np.abs(values[valid, 0] - values[target, 0]) / mz_scale
    )
    rng = np.random.default_rng(seed)
    cost = cost + rng.gumbel(0.0, 1e-8, len(cost))
    order = np.argsort(cost, kind="mergesort")[: min(repeats, len(valid))]
    return valid[order].astype(np.int64)


def matched_control_tokens_strict(
    clean: torch.Tensor,
    target: int,
    roles: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Strict role-, intensity- and m/z-matched controls.

    Unlike :func:`matched_control_tokens`, this helper never silently falls
    back to a different role.  Returning fewer than ``repeats`` is an explicit
    ineligibility signal for the strict causal comparison.
    """
    values = clean.detach().cpu().numpy()
    if target <= 0 or target >= len(values):
        raise ValueError("target must be a real fragment token")
    if roles[target] < 0 or values[target, 0] <= 0 or values[target, 1] <= 0:
        raise ValueError("target must have a valid candidate role")
    valid = np.flatnonzero(
        (values[:, 0] > 0)
        & (values[:, 1] > 0)
        & (np.arange(len(values)) > 0)
        & (roles == roles[target])
    )
    valid = valid[valid != target]
    if len(valid) < repeats:
        return np.empty(0, dtype=np.int64)
    intensity = np.log(np.clip(values[:, 1].astype(float), 1e-8, None))
    mz_scale = max(float(np.std(values[valid, 0])), 25.0)
    cost = (
        4.0 * np.abs(intensity[valid] - intensity[target])
        + 0.15 * np.abs(values[valid, 0] - values[target, 0]) / mz_scale
    )
    rng = np.random.default_rng(seed)
    cost = cost + rng.gumbel(0.0, 1e-8, len(cost))
    order = np.argsort(cost, kind="mergesort")[:repeats]
    return valid[order].astype(np.int64)


def matched_control_tokens_strict_excluding(
    clean: torch.Tensor,
    target: int,
    roles: np.ndarray,
    repeats: int,
    seed: int,
    excluded: set[int] | np.ndarray | tuple[int, ...] = (),
) -> np.ndarray:
    """Strict matched controls after excluding an intervention path.

    Sequential controls must never reuse a target token or a control selected
    at an earlier step. Returning fewer than ``repeats`` explicitly marks the
    sequence as ineligible for a complete paired comparison at that step.
    """
    values = clean.detach().cpu().numpy()
    if target <= 0 or target >= len(values):
        raise ValueError("target must be a real fragment token")
    if roles[target] < 0 or values[target, 0] <= 0 or values[target, 1] <= 0:
        raise ValueError("target must have a valid candidate role")
    blocked = {int(value) for value in excluded}
    blocked.add(int(target))
    valid = np.flatnonzero(
        (values[:, 0] > 0)
        & (values[:, 1] > 0)
        & (np.arange(len(values)) > 0)
        & (roles == roles[target])
    )
    valid = np.asarray(
        [value for value in valid if int(value) not in blocked], dtype=np.int64,
    )
    if len(valid) < repeats:
        return np.empty(0, dtype=np.int64)
    intensity = np.log(np.clip(values[:, 1].astype(float), 1e-8, None))
    mz_scale = max(float(np.std(values[valid, 0])), 25.0)
    cost = (
        4.0 * np.abs(intensity[valid] - intensity[target])
        + 0.15 * np.abs(values[valid, 0] - values[target, 0]) / mz_scale
    )
    rng = np.random.default_rng(seed)
    cost = cost + rng.gumbel(0.0, 1e-8, len(cost))
    order = np.argsort(cost, kind="mergesort")[:repeats]
    return valid[order].astype(np.int64)


def attenuate_sequence(
    clean: torch.Tensor,
    tokens: list[int] | tuple[int, ...] | np.ndarray,
    attenuation: float,
) -> torch.Tensor:
    """Apply a unique fragment-token sequence, normalizing after each step."""
    token_list = [int(token) for token in tokens]
    if len(set(token_list)) != len(token_list):
        raise ValueError("a sequential intervention cannot reuse a token")
    output = clean.clone()
    for token in token_list:
        output = attenuate_and_renormalize(output, token, attenuation)
    return output


@dataclass(frozen=True)
class CandidateRepresentatives:
    positive_row: int
    negative_rows: tuple[int, ...]
    positive_score: float
    hardest_negative_score: float


def candidate_representatives(
    pair_scores: np.ndarray,
    pair_rows: np.ndarray,
    molecule_ptr: np.ndarray,
    top_k_negatives: int,
) -> CandidateRepresentatives:
    """Positive molecule is first. Pick its best spectrum and top negative molecules."""
    scores = np.asarray(pair_scores, dtype=float)
    rows = np.asarray(pair_rows, dtype=np.int64)
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if ptr[0] != 0 or ptr[-1] != len(scores) or len(ptr) < 3:
        raise ValueError("invalid local molecule pointer")
    representatives: list[tuple[float, int]] = []
    for left, right in zip(ptr[:-1], ptr[1:]):
        local = int(np.argmax(scores[left:right])) + int(left)
        representatives.append((float(scores[local]), int(rows[local])))
    negative_order = sorted(
        range(1, len(representatives)),
        key=lambda index: (-representatives[index][0], representatives[index][1]),
    )[:top_k_negatives]
    if not negative_order:
        raise ValueError("query has no negative molecule")
    return CandidateRepresentatives(
        positive_row=representatives[0][1],
        negative_rows=tuple(representatives[index][1] for index in negative_order),
        positive_score=representatives[0][0],
        hardest_negative_score=representatives[negative_order[0]][0],
    )
