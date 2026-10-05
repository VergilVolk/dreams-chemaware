"""Pure helpers for chemistry-by-input-Jacobian intersection actions.

Candidate-differential chemistry says which observed peak should increase or
decrease.  The frozen DreaMS input Jacobian says whether that edit can improve
the exact positive-versus-boundary margin.  An executable action requires both
signals to agree.  Candidate-role and peak-permuted controls are additionally
matched on intensity and absolute Jacobian magnitude, not only edit count.
"""

from __future__ import annotations

import numpy as np

from chemaware_boundary_consensus_action_core import (
    EvidenceProfile,
    PeakActionPlan,
)


def _empty_plan() -> PeakActionPlan:
    return PeakActionPlan(
        positions=np.empty(0, dtype=np.int64),
        factors=np.empty(0, dtype=np.float32),
        evidence=np.empty(0, dtype=np.float32),
        roles=np.empty(0, dtype=np.int8),
    )


def _valid_positions(
    mz: np.ndarray,
    intensity: np.ndarray,
    jacobian: np.ndarray,
    precursor_mz: float,
    minimum_observed_intensity: float,
    precursor_exclusion_da: float,
) -> np.ndarray:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if mz.ndim != 1 or intensity.shape != mz.shape or jacobian.shape != mz.shape:
        raise ValueError("peak values and input Jacobian are not aligned")
    finite_positive = np.isfinite(intensity) & (intensity > 0)
    maximum = (
        float(np.max(intensity[finite_positive])) if np.any(finite_positive) else 0.0
    )
    valid = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & np.isfinite(jacobian)
        & (mz > 0)
        & (intensity >= minimum_observed_intensity)
        & (intensity < maximum)
        & (np.abs(mz - float(precursor_mz)) > precursor_exclusion_da)
    )
    return np.flatnonzero(valid).astype(np.int64)


def first_order_log_intensity_gain(
    plan: PeakActionPlan,
    log_intensity_jacobian: np.ndarray,
) -> float:
    """Return ``J dot delta_log_intensity`` for one sparse action."""
    if plan.abstained:
        return 0.0
    jacobian = np.asarray(log_intensity_jacobian, dtype=np.float64)
    if np.any(plan.positions < 0) or np.any(plan.positions >= len(jacobian)):
        raise IndexError("action position lies outside the Jacobian")
    if np.any(plan.factors <= 0):
        raise ValueError("action factors must be positive")
    return float(np.sum(jacobian[plan.positions] * np.log(plan.factors)))


def build_jacobian_intersection_plan(
    profile: EvidenceProfile,
    log_intensity_jacobian: np.ndarray,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    mode: str,
    log_dose: float,
    top_k: int,
    minimum_abs_evidence: float,
    minimum_agreement: float = 0.75,
    minimum_observed_intensity: float = 0.01,
    precursor_exclusion_da: float = 1.1,
    minimum_first_order_gain: float = 0.0,
) -> PeakActionPlan:
    """Keep chemical directions with strictly positive boundary derivative."""
    signed = np.asarray(profile.signed, dtype=np.float64)
    agreement = np.asarray(profile.agreement, dtype=np.float64)
    jacobian = np.asarray(log_intensity_jacobian, dtype=np.float64)
    mz = np.asarray(observed_mz, dtype=np.float64)
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    if signed.shape != mz.shape or agreement.shape != mz.shape:
        raise ValueError("chemical evidence does not align to observed peaks")
    if log_dose <= 0 or top_k < 1 or minimum_abs_evidence <= 0:
        raise ValueError("invalid Jacobian-intersection action setting")
    valid = _valid_positions(
        mz,
        intensity,
        jacobian,
        precursor_mz,
        minimum_observed_intensity,
        precursor_exclusion_da,
    )
    maximum = float(np.max(intensity))
    valid = valid[intensity[valid] * np.exp(log_dose) < maximum]

    def select(role: int) -> np.ndarray:
        chemical = role * signed[valid]
        gain = role * log_dose * jacobian[valid]
        eligible = (
            (chemical >= minimum_abs_evidence)
            & (agreement[valid] >= minimum_agreement)
            & (gain > minimum_first_order_gain)
        )
        positions = valid[eligible]
        if not len(positions):
            return positions
        # Geometric combination prevents either chemistry or model sensitivity
        # from dominating selection on its own.
        utility = np.sqrt(
            np.maximum(role * signed[positions], 0.0)
            * np.maximum(role * log_dose * jacobian[positions], 0.0)
        )
        return positions[np.argsort(-utility, kind="stable")[:top_k]]

    if mode == "support_boost":
        chosen = [(1, select(1))]
    elif mode == "conflict_attenuate":
        chosen = [(-1, select(-1))]
    elif mode == "bidirectional_sharpen":
        negative, positive = select(-1), select(1)
        if not len(negative) or not len(positive):
            return _empty_plan()
        chosen = [(-1, negative), (1, positive)]
    else:
        raise ValueError(f"unknown Jacobian-intersection mode: {mode}")
    positions = np.concatenate([value for _, value in chosen])
    if not len(positions):
        return _empty_plan()
    roles = np.concatenate(
        [np.full(len(value), role, dtype=np.int8) for role, value in chosen]
    )
    factors = np.exp(log_dose * roles.astype(np.float64)).astype(np.float32)
    plan = PeakActionPlan(
        positions=positions.astype(np.int64),
        factors=factors,
        evidence=signed[positions].astype(np.float32),
        roles=roles,
    )
    if first_order_log_intensity_gain(plan, jacobian) <= 0:
        raise RuntimeError("Jacobian-intersection action has nonpositive local gain")
    return plan


def matched_jacobian_control_plan(
    profile: EvidenceProfile,
    log_intensity_jacobian: np.ndarray,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    precursor_mz: float,
    target: PeakActionPlan,
    minimum_agreement: float = 0.0,
    minimum_abs_evidence: float = 0.0,
    minimum_observed_intensity: float = 0.01,
    precursor_exclusion_da: float = 1.1,
) -> PeakActionPlan:
    """Match target count, dose, intensity and ``abs(J)`` with fake chemistry.

    Matching is greedy and deterministic.  A query abstains across all arms if
    a control lacks enough Jacobian-aligned peaks in either requested role.
    """
    if target.abstained:
        return _empty_plan()
    signed = np.asarray(profile.signed, dtype=np.float64)
    agreement = np.asarray(profile.agreement, dtype=np.float64)
    jacobian = np.asarray(log_intensity_jacobian, dtype=np.float64)
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    if signed.shape != jacobian.shape or agreement.shape != jacobian.shape:
        raise ValueError("control evidence does not align to the Jacobian")
    if minimum_abs_evidence < 0:
        raise ValueError("control evidence threshold must be non-negative")
    valid = _valid_positions(
        observed_mz,
        intensity,
        jacobian,
        precursor_mz,
        minimum_observed_intensity,
        precursor_exclusion_da,
    )
    maximum = float(np.max(intensity))
    maximum_log_dose = float(np.max(np.abs(np.log(target.factors))))
    valid = valid[intensity[valid] * np.exp(maximum_log_dose) < maximum]
    chosen: list[int] = []
    factors: list[float] = []
    evidence: list[float] = []
    roles: list[int] = []
    for role in (-1, 1):
        target_index = np.flatnonzero(target.roles == role)
        if not len(target_index):
            continue
        available = valid[
            (role * signed[valid] >= max(minimum_abs_evidence, np.finfo(float).eps))
            & (agreement[valid] >= minimum_agreement)
            & (role * jacobian[valid] > 0)
        ]
        available = np.asarray(
            [value for value in available if int(value) not in chosen],
            dtype=np.int64,
        )
        if len(available) < len(target_index):
            return _empty_plan()
        ordered_target = target_index[
            np.argsort(
                -np.abs(jacobian[target.positions[target_index]]),
                kind="stable",
            )
        ]
        for target_plan_index in ordered_target:
            target_position = int(target.positions[target_plan_index])
            log_j_distance = np.abs(
                np.log(np.abs(jacobian[available]) + 1e-12)
                - np.log(abs(jacobian[target_position]) + 1e-12)
            )
            log_i_distance = np.abs(
                np.log(intensity[available] + 1e-12)
                - np.log(intensity[target_position] + 1e-12)
            )
            chemical_tie_break = -role * signed[available]
            order = np.lexsort(
                (available, chemical_tie_break, log_i_distance, log_j_distance)
            )
            position = int(available[order[0]])
            chosen.append(position)
            roles.append(role)
            factors.append(float(target.factors[target_plan_index]))
            evidence.append(float(signed[position]))
            available = available[available != position]
    plan = PeakActionPlan(
        positions=np.asarray(chosen, dtype=np.int64),
        factors=np.asarray(factors, dtype=np.float32),
        evidence=np.asarray(evidence, dtype=np.float32),
        roles=np.asarray(roles, dtype=np.int8),
    )
    if first_order_log_intensity_gain(plan, jacobian) <= 0:
        return _empty_plan()
    return plan


def jacobian_match_error(
    target: PeakActionPlan,
    control: PeakActionPlan,
    log_intensity_jacobian: np.ndarray,
    observed_intensity: np.ndarray,
) -> dict[str, float]:
    """Summarize role-wise sensitivity/intensity mismatch after control match."""
    if target.abstained or control.abstained:
        raise ValueError("match error requires active plans")
    jacobian = np.asarray(log_intensity_jacobian, dtype=np.float64)
    intensity = np.asarray(observed_intensity, dtype=np.float64)
    j_error: list[float] = []
    i_error: list[float] = []
    for role in (-1, 1):
        target_value = np.sort(np.abs(jacobian[target.positions[target.roles == role]]))
        control_value = np.sort(
            np.abs(jacobian[control.positions[control.roles == role]])
        )
        target_intensity = np.sort(intensity[target.positions[target.roles == role]])
        control_intensity = np.sort(intensity[control.positions[control.roles == role]])
        if len(target_value) != len(control_value):
            raise ValueError("target and control roles have different capacity")
        j_error.extend(
            np.abs(np.log(target_value + 1e-12) - np.log(control_value + 1e-12))
        )
        i_error.extend(
            np.abs(np.log(target_intensity + 1e-12) - np.log(control_intensity + 1e-12))
        )
    return {
        "mean_abs_log_jacobian_error": float(np.mean(j_error) if j_error else 0.0),
        "max_abs_log_jacobian_error": float(np.max(j_error) if j_error else 0.0),
        "mean_abs_log_intensity_error": float(np.mean(i_error) if i_error else 0.0),
        "max_abs_log_intensity_error": float(np.max(i_error) if i_error else 0.0),
    }
