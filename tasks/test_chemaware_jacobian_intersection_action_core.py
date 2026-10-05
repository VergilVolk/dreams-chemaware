"""CPU contracts for chemistry-by-input-Jacobian actions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_boundary_consensus_action_core import EvidenceProfile, invert_action_plan
from chemaware_jacobian_intersection_action_core import (
    build_jacobian_intersection_plan,
    first_order_log_intensity_gain,
    jacobian_match_error,
    matched_jacobian_control_plan,
)


def fixture() -> tuple[EvidenceProfile, np.ndarray, np.ndarray, np.ndarray]:
    profile = EvidenceProfile(
        signed=np.asarray([-0.8, -0.5, 0.7, 0.4, 0.3, -0.2], dtype=np.float32),
        agreement=np.ones(6, dtype=np.float32),
        amplitude=np.ones(6, dtype=np.float32),
    )
    jacobian = np.asarray([-0.9, 0.4, 0.8, -0.5, 0.2, -0.1], dtype=np.float32)
    mz = np.asarray([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
    intensity = np.asarray([0.7, 0.6, 0.65, 1.0, 0.5, 0.4])
    return profile, jacobian, mz, intensity


def test_intersection_rejects_chemistry_jacobian_conflicts() -> None:
    profile, jacobian, mz, intensity = fixture()
    plan = build_jacobian_intersection_plan(
        profile,
        jacobian,
        mz,
        intensity,
        precursor_mz=500.0,
        mode="bidirectional_sharpen",
        log_dose=0.25,
        top_k=2,
        minimum_abs_evidence=0.1,
    )
    assert set(plan.positions.tolist()) == {0, 2, 4, 5}
    assert np.all(jacobian[plan.positions] * plan.roles > 0)
    assert first_order_log_intensity_gain(plan, jacobian) > 0


def test_precursor_and_weak_evidence_are_excluded() -> None:
    profile, jacobian, mz, intensity = fixture()
    mz[2] = 499.5
    plan = build_jacobian_intersection_plan(
        profile,
        jacobian,
        mz,
        intensity,
        precursor_mz=500.0,
        mode="support_boost",
        log_dose=0.25,
        top_k=3,
        minimum_abs_evidence=0.25,
    )
    assert plan.positions.tolist() == [4]


def test_direction_inverse_has_opposite_first_order_gain() -> None:
    profile, jacobian, mz, intensity = fixture()
    plan = build_jacobian_intersection_plan(
        profile, jacobian, mz, intensity, 500.0, "support_boost", 0.5, 2, 0.1
    )
    inverse = invert_action_plan(plan)
    assert np.isclose(
        first_order_log_intensity_gain(inverse, jacobian),
        -first_order_log_intensity_gain(plan, jacobian),
    )


def test_control_matches_capacity_and_jacobian_strata() -> None:
    profile, jacobian, mz, intensity = fixture()
    target = build_jacobian_intersection_plan(
        profile, jacobian, mz, intensity, 500.0, "bidirectional_sharpen", 0.25, 1, 0.1
    )
    control_profile = EvidenceProfile(
        signed=np.asarray([-0.2, -0.6, 0.4, -0.3, 0.5, -0.7], dtype=np.float32),
        agreement=np.ones(6, dtype=np.float32),
        amplitude=np.ones(6, dtype=np.float32),
    )
    control = matched_jacobian_control_plan(
        control_profile, jacobian, mz, intensity, 500.0, target
    )
    assert not control.abstained
    assert control.attenuated == target.attenuated
    assert control.boosted == target.boosted
    assert np.allclose(np.sort(control.factors), np.sort(target.factors))
    assert first_order_log_intensity_gain(control, jacobian) > 0
    error = jacobian_match_error(target, control, jacobian, intensity)
    assert set(error) == {
        "mean_abs_log_jacobian_error",
        "max_abs_log_jacobian_error",
        "mean_abs_log_intensity_error",
        "max_abs_log_intensity_error",
    }


def test_control_abstains_if_one_role_is_unavailable() -> None:
    profile, jacobian, mz, intensity = fixture()
    target = build_jacobian_intersection_plan(
        profile, jacobian, mz, intensity, 500.0, "bidirectional_sharpen", 0.25, 1, 0.1
    )
    no_negative = EvidenceProfile(
        signed=np.ones(6, dtype=np.float32),
        agreement=np.ones(6, dtype=np.float32),
        amplitude=np.ones(6, dtype=np.float32),
    )
    control = matched_jacobian_control_plan(
        no_negative, jacobian, mz, intensity, 500.0, target
    )
    assert control.abstained


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware Jacobian-intersection action contracts")


if __name__ == "__main__":
    main()
