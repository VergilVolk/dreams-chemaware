"""Pure helpers for chemically specific, capacity-matched peak actions.

The module fixes three confounders in the first ICEBERG action screen:

* the candidate boundary is selected by frozen official DreaMS retrieval;
* evidence must agree against several high-scoring same-formula negatives;
* target and controls modify the same number of observed peak slots with the
  same multiplicative doses.

Candidate structures and predicted spectra are training/audit-only inputs.
Every returned action changes intensity at already observed peaks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EvidenceProfile:
    """Peak-aligned candidate contrast and its support diagnostics."""

    signed: np.ndarray
    agreement: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class PairEvidenceProfile:
    """Antisymmetric evidence for candidate-discriminative peak ratios."""

    signed: np.ndarray
    agreement: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class PeakActionPlan:
    """Sparse multiplicative edit over non-precursor spectrum slots."""

    positions: np.ndarray
    factors: np.ndarray
    evidence: np.ndarray
    roles: np.ndarray

    @property
    def abstained(self) -> bool:
        return len(self.positions) == 0

    @property
    def attenuated(self) -> int:
        return int(np.sum(self.roles == -1))

    @property
    def boosted(self) -> int:
        return int(np.sum(self.roles == 1))


def _empty_plan() -> PeakActionPlan:
    return PeakActionPlan(
        positions=np.empty(0, dtype=np.int64),
        factors=np.empty(0, dtype=np.float32),
        evidence=np.empty(0, dtype=np.float32),
        roles=np.empty(0, dtype=np.int8),
    )


def peak_bins(mz: np.ndarray, n_bins: int, upper_mz: float = 1500.0) -> np.ndarray:
    """Map observed m/z values to the persisted ICEBERG spectrum grid."""
    values = np.asarray(mz, dtype=np.float64)
    if values.ndim != 1 or n_bins < 2 or upper_mz <= 0:
        raise ValueError("invalid peak-bin inputs")
    grid = np.linspace(0.0, float(upper_mz), int(n_bins))
    return np.clip(np.digitize(values, bins=grid), 0, n_bins - 1)


def top_official_negative_positions(
    official_scores: np.ndarray,
    candidate_formula: np.ndarray,
    query_formula: str,
    top_k: int,
    minimum_negatives: int = 2,
) -> np.ndarray:
    """Select the official-DreaMS boundary and peers within one formula.

    Candidate zero is required to be the unique positive.  Returning an empty
    array is a deliberate abstention when the same-formula boundary is not
    observable with enough independent negative identities.
    """
    score = np.asarray(official_scores, dtype=np.float64)
    formula = np.asarray(candidate_formula).astype(str)
    if score.ndim != 1 or len(score) != len(formula) or len(score) < 2:
        raise ValueError("candidate score/formula arrays are not aligned")
    if not np.all(np.isfinite(score)):
        raise ValueError("official candidate scores must be finite")
    if formula[0] != str(query_formula):
        raise ValueError("positive candidate formula differs from query formula")
    if top_k < 1 or minimum_negatives < 1:
        raise ValueError("negative-count requirements must be positive")
    eligible = np.flatnonzero(formula[1:] == str(query_formula)) + 1
    if len(eligible) < minimum_negatives:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-score[eligible], kind="stable")
    return eligible[order[: min(int(top_k), len(order))]].astype(np.int64)


def consensus_evidence(
    predictions: np.ndarray,
    true_position: int,
    boundary_position: int,
    negative_positions: np.ndarray,
    observed_mz: np.ndarray,
    agreement_quantile: float = 0.75,
    minimum_prediction: float = 0.10,
) -> EvidenceProfile:
    """Compute signed evidence that is stable against a negative ensemble.

    Let ``d_j = sqrt(p_true) - sqrt(p_negative_j)`` at an observed peak.
    Support requires both the official boundary contrast and the lower
    ``1-agreement_quantile`` quantile of ``d_j`` to be positive.  Conflict
    requires the boundary contrast and the upper ``agreement_quantile``
    quantile to be negative.  Mixed-sign peaks receive exactly zero evidence.
    """
    values = np.asarray(predictions, dtype=np.float32)
    negatives = np.asarray(negative_positions, dtype=np.int64)
    mz = np.asarray(observed_mz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("predictions must be candidate-by-bin")
    if mz.ndim != 1:
        raise ValueError("observed_mz must be one-dimensional")
    if not 0.5 <= agreement_quantile < 1.0:
        raise ValueError("agreement_quantile must lie in [0.5, 1)")
    if minimum_prediction < 0:
        raise ValueError("minimum_prediction must be non-negative")
    if len(negatives) == 0 or boundary_position not in set(map(int, negatives)):
        raise ValueError("negative ensemble must contain the frozen boundary")
    all_positions = np.r_[true_position, boundary_position, negatives]
    if np.any(all_positions < 0) or np.any(all_positions >= len(values)):
        raise IndexError("candidate position lies outside predictions")
    if true_position in set(map(int, negatives)):
        raise ValueError("true candidate cannot appear in the negative ensemble")

    maximum = np.maximum(np.max(values, axis=1, keepdims=True), 1e-12)
    normalized = np.sqrt(np.maximum(values / maximum, 0.0))
    index = peak_bins(mz, values.shape[1])
    true = normalized[int(true_position), index]
    negative = normalized[negatives][:, index]
    boundary = normalized[int(boundary_position), index]
    delta = true[None, :] - negative
    boundary_delta = true - boundary
    lower = np.quantile(delta, 1.0 - agreement_quantile, axis=0)
    upper = np.quantile(delta, agreement_quantile, axis=0)

    signed = np.zeros(len(mz), dtype=np.float32)
    support = (boundary_delta > 0) & (lower > 0)
    conflict = (boundary_delta < 0) & (upper < 0)
    signed[support] = np.minimum(boundary_delta[support], lower[support])
    signed[conflict] = np.maximum(boundary_delta[conflict], upper[conflict])
    amplitude = np.maximum(true, np.max(negative, axis=0)).astype(np.float32)
    signed[amplitude < minimum_prediction] = 0.0
    agreement = np.zeros(len(mz), dtype=np.float32)
    agreement[signed > 0] = np.mean(delta[:, signed > 0] > 0, axis=0)
    agreement[signed < 0] = np.mean(delta[:, signed < 0] < 0, axis=0)
    return EvidenceProfile(signed=signed, agreement=agreement, amplitude=amplitude)


def pairwise_consensus_evidence(
    predictions: np.ndarray,
    true_position: int,
    boundary_position: int,
    negative_positions: np.ndarray,
    observed_mz: np.ndarray,
    agreement_quantile: float = 0.75,
    minimum_prediction: float = 0.10,
    pseudocount: float = 0.02,
) -> PairEvidenceProfile:
    """Compute robust true-vs-negative evidence for every peak log ratio.

    For observed peaks ``i`` and ``j``, the signed statistic is the robust part
    of ``log(p_true_i / p_true_j) - log(p_negative_i / p_negative_j)`` whose
    direction agrees with both the frozen official boundary and the requested
    quantile of same-formula negatives.  Peak ratios remove every candidate's
    global predicted-intensity scale before an action is proposed.
    """
    values = np.asarray(predictions, dtype=np.float32)
    negatives = np.asarray(negative_positions, dtype=np.int64)
    mz = np.asarray(observed_mz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("predictions must be candidate-by-bin")
    if mz.ndim != 1:
        raise ValueError("observed_mz must be one-dimensional")
    if not 0.5 <= agreement_quantile < 1.0:
        raise ValueError("agreement_quantile must lie in [0.5, 1)")
    if minimum_prediction < 0 or pseudocount <= 0:
        raise ValueError("prediction threshold and pseudocount are invalid")
    if len(negatives) == 0 or boundary_position not in set(map(int, negatives)):
        raise ValueError("negative ensemble must contain the frozen boundary")
    all_positions = np.r_[true_position, boundary_position, negatives]
    if np.any(all_positions < 0) or np.any(all_positions >= len(values)):
        raise IndexError("candidate position lies outside predictions")
    if true_position in set(map(int, negatives)):
        raise ValueError("true candidate cannot appear in the negative ensemble")

    maximum = np.maximum(np.max(values, axis=1, keepdims=True), 1e-12)
    normalized = np.sqrt(np.maximum(values / maximum, 0.0))
    index = peak_bins(mz, values.shape[1])
    observed = normalized[:, index]
    logged = np.log(observed + float(pseudocount))
    true_ratio = (
        logged[int(true_position), :, None] - logged[int(true_position), None, :]
    )
    negative_ratio = logged[negatives, :, None] - logged[negatives, None, :]
    delta = true_ratio[None, :, :] - negative_ratio
    boundary_index = int(np.flatnonzero(negatives == boundary_position)[0])
    boundary_delta = delta[boundary_index]
    lower = np.quantile(delta, 1.0 - agreement_quantile, axis=0)
    upper = np.quantile(delta, agreement_quantile, axis=0)

    signed = np.zeros_like(boundary_delta, dtype=np.float32)
    support = (boundary_delta > 0) & (lower > 0)
    conflict = (boundary_delta < 0) & (upper < 0)
    signed[support] = np.minimum(boundary_delta[support], lower[support])
    signed[conflict] = np.maximum(boundary_delta[conflict], upper[conflict])
    peak_amplitude = np.max(observed[np.r_[true_position, negatives]], axis=0)
    amplitude = np.minimum(peak_amplitude[:, None], peak_amplitude[None, :]).astype(
        np.float32
    )
    signed[amplitude < minimum_prediction] = 0.0
    np.fill_diagonal(signed, 0.0)
    agreement = np.zeros_like(signed, dtype=np.float32)
    agreement[signed > 0] = np.mean(delta[:, signed > 0] > 0, axis=0)
    agreement[signed < 0] = np.mean(delta[:, signed < 0] < 0, axis=0)
    return PairEvidenceProfile(
        signed=signed,
        agreement=agreement,
        amplitude=amplitude,
    )


def _observable_positions(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    minimum_observed_intensity: float,
    precursor_exclusion_da: float,
) -> np.ndarray:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.ndim != 1 or intensity.shape != mz.shape:
        raise ValueError("m/z and intensity arrays are not aligned")
    finite_positive = np.isfinite(intensity) & (intensity > 0)
    maximum = (
        float(np.max(intensity[finite_positive])) if np.any(finite_positive) else 0.0
    )
    valid = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0)
        & (intensity >= minimum_observed_intensity)
        & (intensity < maximum)
        & (np.abs(mz - float(precursor_mz)) > precursor_exclusion_da)
    )
    return np.flatnonzero(valid).astype(np.int64)


def build_peak_action_plan(
    profile: EvidenceProfile,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    mode: str,
    strength: float,
    top_k: int,
    minimum_abs_evidence: float,
    minimum_agreement: float = 0.75,
    minimum_observed_intensity: float = 0.01,
    precursor_exclusion_da: float = 1.1,
) -> PeakActionPlan:
    """Create a thresholded action; return an empty plan when unsupported."""
    signed = np.asarray(profile.signed, dtype=np.float32)
    agreement = np.asarray(profile.agreement, dtype=np.float32)
    mz = np.asarray(observed_mz, dtype=np.float64)
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    if signed.shape != mz.shape or agreement.shape != mz.shape:
        raise ValueError("evidence profile does not align to observed peaks")
    if not 0 < strength < 1 or top_k < 1 or minimum_abs_evidence <= 0:
        raise ValueError("invalid action strength, top_k, or evidence threshold")
    valid = _observable_positions(
        mz,
        intensity,
        precursor_mz,
        minimum_observed_intensity,
        precursor_exclusion_da,
    )
    support = valid[
        (signed[valid] >= minimum_abs_evidence)
        & (agreement[valid] >= minimum_agreement)
    ]
    conflict = valid[
        (signed[valid] <= -minimum_abs_evidence)
        & (agreement[valid] >= minimum_agreement)
    ]
    maximum = float(np.max(intensity))
    support_multiplier = max(1.0 + strength, 1.0 / (1.0 + strength))
    conflict_multiplier = max(1.0 - strength, 1.0 / (1.0 - strength))
    support = support[intensity[support] * support_multiplier < maximum]
    conflict = conflict[intensity[conflict] * conflict_multiplier < maximum]
    support = support[np.argsort(-signed[support], kind="stable")[:top_k]]
    conflict = conflict[np.argsort(signed[conflict], kind="stable")[:top_k]]

    if mode == "support_boost":
        positions, roles = support, np.ones(len(support), dtype=np.int8)
    elif mode == "conflict_attenuate":
        positions, roles = conflict, -np.ones(len(conflict), dtype=np.int8)
    elif mode == "bidirectional_sharpen":
        if not len(support) or not len(conflict):
            return _empty_plan()
        positions = np.r_[conflict, support]
        roles = np.r_[
            -np.ones(len(conflict), dtype=np.int8), np.ones(len(support), dtype=np.int8)
        ]
    else:
        raise ValueError(f"unknown action mode: {mode}")
    if not len(positions):
        return _empty_plan()
    factors = np.where(roles < 0, 1.0 - strength, 1.0 + strength).astype(np.float32)
    return PeakActionPlan(
        positions=np.asarray(positions, dtype=np.int64),
        factors=factors,
        evidence=signed[positions].astype(np.float32),
        roles=roles,
    )


def build_pair_logratio_action_plan(
    profile: PairEvidenceProfile,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    log_ratio_dose: float,
    top_pairs: int,
    minimum_abs_evidence: float,
    minimum_agreement: float = 0.75,
    minimum_observed_intensity: float = 0.01,
    precursor_exclusion_da: float = 1.1,
) -> PeakActionPlan:
    """Sharpen disjoint peak ratios with symmetric log-intensity doses."""
    signed = np.asarray(profile.signed, dtype=np.float32)
    agreement = np.asarray(profile.agreement, dtype=np.float32)
    mz = np.asarray(observed_mz, dtype=np.float64)
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    if signed.shape != (len(mz), len(mz)) or agreement.shape != signed.shape:
        raise ValueError("pair evidence does not align to observed peaks")
    if log_ratio_dose <= 0 or top_pairs < 1 or minimum_abs_evidence < 0:
        raise ValueError("invalid pair-action dose, count, or evidence threshold")
    valid = _observable_positions(
        mz,
        intensity,
        precursor_mz,
        minimum_observed_intensity,
        precursor_exclusion_da,
    )
    if len(valid) < 2:
        return _empty_plan()
    left, right = np.triu_indices(len(mz), k=1)
    eligible = np.isin(left, valid) & np.isin(right, valid)
    eligible &= np.abs(signed[left, right]) >= minimum_abs_evidence
    eligible &= agreement[left, right] >= minimum_agreement
    maximum = float(np.max(intensity))
    upper_factor = float(np.exp(0.5 * log_ratio_dose))
    eligible &= intensity[left] * upper_factor < maximum
    eligible &= intensity[right] * upper_factor < maximum
    pair_positions = np.flatnonzero(eligible)
    if not len(pair_positions):
        return _empty_plan()
    order = pair_positions[
        np.argsort(
            -np.abs(signed[left[pair_positions], right[pair_positions]]), kind="stable"
        )
    ]
    used: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for pair_position in order:
        first = int(left[pair_position])
        second = int(right[pair_position])
        if first in used or second in used:
            continue
        selected.append((first, second, float(signed[first, second])))
        used.update((first, second))
        if len(selected) == top_pairs:
            break
    if not selected:
        return _empty_plan()

    positions: list[int] = []
    factors: list[float] = []
    evidence: list[float] = []
    roles: list[int] = []
    lower_factor = float(np.exp(-0.5 * log_ratio_dose))
    for first, second, value in selected:
        boosted, attenuated = (first, second) if value > 0 else (second, first)
        positions.extend((attenuated, boosted))
        factors.extend((lower_factor, upper_factor))
        evidence.extend((-abs(value), abs(value)))
        roles.extend((-1, 1))
    return PeakActionPlan(
        positions=np.asarray(positions, dtype=np.int64),
        factors=np.asarray(factors, dtype=np.float32),
        evidence=np.asarray(evidence, dtype=np.float32),
        roles=np.asarray(roles, dtype=np.int8),
    )


def capacity_matched_pair_logratio_plan(
    profile: PairEvidenceProfile,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    target: PeakActionPlan,
    minimum_observed_intensity: float = 0.01,
    precursor_exclusion_da: float = 1.1,
) -> PeakActionPlan:
    """Build a pair control with the target's exact count and log-dose."""
    if target.abstained:
        return _empty_plan()
    if target.attenuated != target.boosted or len(target.positions) % 2:
        raise ValueError("pair target must contain balanced peak pairs")
    positive_factors = target.factors[target.roles == 1]
    negative_factors = target.factors[target.roles == -1]
    if not len(positive_factors) or not np.allclose(
        positive_factors, positive_factors[0]
    ):
        raise ValueError("pair target boost doses are inconsistent")
    if not np.allclose(negative_factors, negative_factors[0]):
        raise ValueError("pair target attenuation doses are inconsistent")
    log_ratio_dose = float(np.log(positive_factors[0] / negative_factors[0]))
    return build_pair_logratio_action_plan(
        profile,
        observed_mz,
        observed_intensity,
        precursor_mz,
        log_ratio_dose,
        target.boosted,
        0.0,
        0.0,
        minimum_observed_intensity,
        precursor_exclusion_da,
    )


def invert_action_plan(plan: PeakActionPlan) -> PeakActionPlan:
    """Reverse an action at the same peak slots with equal absolute log-dose."""
    if plan.abstained:
        return _empty_plan()
    if np.any(plan.factors <= 0):
        raise ValueError("action factors must be positive")
    return PeakActionPlan(
        positions=np.asarray(plan.positions, dtype=np.int64).copy(),
        factors=(1.0 / np.asarray(plan.factors, dtype=np.float32)).astype(np.float32),
        evidence=(-np.asarray(plan.evidence, dtype=np.float32)).copy(),
        roles=(-np.asarray(plan.roles, dtype=np.int8)).copy(),
    )


def capacity_matched_plan(
    profile: EvidenceProfile,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    target: PeakActionPlan,
    minimum_observed_intensity: float = 0.01,
    precursor_exclusion_da: float = 1.1,
) -> PeakActionPlan:
    """Match target boost/attenuation counts and factors without thresholding."""
    if target.abstained:
        return _empty_plan()
    signed = np.asarray(profile.signed, dtype=np.float32)
    valid = _observable_positions(
        observed_mz,
        observed_intensity,
        precursor_mz,
        minimum_observed_intensity,
        precursor_exclusion_da,
    )
    if len(valid) < len(target.positions):
        return _empty_plan()
    chosen: list[int] = []
    roles: list[int] = []
    factors: list[float] = []
    evidence: list[float] = []
    for role in (-1, 1):
        target_factors = target.factors[target.roles == role]
        if not len(target_factors):
            continue
        remaining = np.asarray(
            [value for value in valid if int(value) not in chosen], dtype=np.int64
        )
        maximum = float(np.max(np.asarray(observed_intensity, dtype=np.float64)))
        maximum_multiplier = max(
            float(np.max(target_factors)),
            float(np.max(1.0 / target_factors)),
        )
        remaining = remaining[
            np.asarray(observed_intensity, dtype=np.float64)[remaining]
            * maximum_multiplier
            < maximum
        ]
        order = np.argsort(signed[remaining], kind="stable")
        if role == 1:
            order = order[::-1]
        positions = remaining[order[: len(target_factors)]]
        if len(positions) != len(target_factors):
            return _empty_plan()
        chosen.extend(map(int, positions))
        roles.extend([role] * len(positions))
        factors.extend(map(float, target_factors))
        evidence.extend(map(float, signed[positions]))
    return PeakActionPlan(
        positions=np.asarray(chosen, dtype=np.int64),
        factors=np.asarray(factors, dtype=np.float32),
        evidence=np.asarray(evidence, dtype=np.float32),
        roles=np.asarray(roles, dtype=np.int8),
    )


def intensity_rank_permuted_profile(
    profile: EvidenceProfile,
    observed_intensity: np.ndarray,
    seed: int,
    groups: int = 5,
) -> EvidenceProfile:
    """Break peak identity while retaining local intensity-rank capacity."""
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    if intensity.ndim != 1 or intensity.shape != np.asarray(profile.signed).shape:
        raise ValueError("intensity does not align to evidence")
    valid = np.flatnonzero(np.isfinite(intensity) & (intensity > 0))
    if len(valid) < 2:
        return EvidenceProfile(
            signed=np.asarray(profile.signed).copy(),
            agreement=np.asarray(profile.agreement).copy(),
            amplitude=np.asarray(profile.amplitude).copy(),
        )
    ordered = valid[np.argsort(intensity[valid], kind="stable")]
    n_groups = max(1, min(int(groups), len(ordered) // 2))
    rng = np.random.default_rng(int(seed))
    permutation = np.arange(len(intensity), dtype=np.int64)
    for block in np.array_split(ordered, n_groups):
        if len(block) < 2:
            continue
        shift = int(rng.integers(1, len(block)))
        permutation[block] = np.roll(block, shift)
    return EvidenceProfile(
        signed=np.asarray(profile.signed)[permutation].astype(np.float32),
        agreement=np.asarray(profile.agreement)[permutation].astype(np.float32),
        amplitude=np.asarray(profile.amplitude)[permutation].astype(np.float32),
    )


def intensity_rank_permuted_pair_profile(
    profile: PairEvidenceProfile,
    observed_intensity: np.ndarray,
    seed: int,
    groups: int = 5,
) -> PairEvidenceProfile:
    """Break peak-pair identity while retaining intensity-rank strata."""
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    signed = np.asarray(profile.signed)
    if intensity.ndim != 1 or signed.shape != (len(intensity), len(intensity)):
        raise ValueError("intensity does not align to pair evidence")
    valid = np.flatnonzero(np.isfinite(intensity) & (intensity > 0))
    permutation = np.arange(len(intensity), dtype=np.int64)
    if len(valid) >= 2:
        ordered = valid[np.argsort(intensity[valid], kind="stable")]
        n_groups = max(1, min(int(groups), len(ordered) // 2))
        rng = np.random.default_rng(int(seed))
        for block in np.array_split(ordered, n_groups):
            if len(block) < 2:
                continue
            shift = int(rng.integers(1, len(block)))
            permutation[block] = np.roll(block, shift)
    index = np.ix_(permutation, permutation)
    return PairEvidenceProfile(
        signed=np.asarray(profile.signed)[index].astype(np.float32),
        agreement=np.asarray(profile.agreement)[index].astype(np.float32),
        amplitude=np.asarray(profile.amplitude)[index].astype(np.float32),
    )


def apply_action_to_arrays(
    mz: np.ndarray,
    intensity: np.ndarray,
    plan: PeakActionPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy action executor used by contracts and the torch wrapper."""
    mz_values = np.asarray(mz).copy()
    intensity_values = np.asarray(intensity, dtype=np.float32).copy()
    if mz_values.ndim != 1 or intensity_values.shape != mz_values.shape:
        raise ValueError("m/z and intensity arrays are not aligned")
    if plan.abstained:
        return mz_values, intensity_values
    if np.any(plan.positions < 0) or np.any(plan.positions >= len(mz_values)):
        raise IndexError("action peak position lies outside the spectrum")
    if len(np.unique(plan.positions)) != len(plan.positions):
        raise ValueError("an action plan cannot edit one peak twice")
    if len(plan.positions) != len(plan.factors) or np.any(plan.factors <= 0):
        raise ValueError("action factors are invalid")
    intensity_values[plan.positions] *= plan.factors
    maximum = float(np.max(intensity_values)) if len(intensity_values) else 0.0
    if maximum > 0:
        intensity_values /= maximum
    return mz_values, intensity_values


def apply_action_plan(clean: torch.Tensor, plan: PeakActionPlan) -> torch.Tensor:
    """Apply a sparse plan and preserve all peak m/z values exactly."""
    import torch

    if clean.ndim != 2 or clean.shape[1] != 2 or len(clean) < 2:
        raise ValueError("expected a preprocessed DreaMS spectrum tensor")
    if plan.abstained:
        return clean.clone()
    if np.any(plan.positions < 0) or np.any(plan.positions >= len(clean) - 1):
        raise IndexError("action peak position lies outside the spectrum")
    if len(np.unique(plan.positions)) != len(plan.positions):
        raise ValueError("an action plan cannot edit one peak twice")
    if len(plan.positions) != len(plan.factors) or np.any(plan.factors <= 0):
        raise ValueError("action factors are invalid")
    output = clean.clone()
    index = torch.as_tensor(plan.positions + 1, dtype=torch.long, device=output.device)
    factor = torch.as_tensor(plan.factors, dtype=output.dtype, device=output.device)
    output[index, 1] *= factor
    maximum = output[1:, 1].max()
    if float(maximum) > 0:
        output[1:, 1] /= maximum
    return output
