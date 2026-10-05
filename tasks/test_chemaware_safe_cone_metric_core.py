"""CPU contracts for the ChemAware safe-cone metric solver."""

from __future__ import annotations

import numpy as np

from chemaware_safe_cone_metric_core import (
    control_orthogonal_residual,
    fit_safe_cone_ridge,
)


def test_control_residual_removes_matched_nuisance() -> None:
    nuisance = np.asarray([[1.0], [2.0], [-1.0], [-2.0]])
    unique = np.asarray([1.0, -1.0, 1.0, -1.0])
    correct = 3.0 * nuisance[:, 0] + unique
    residual, coefficient, report = control_orthogonal_residual(
        correct, nuisance, np.ones(4), ridge=0.0,
    )
    assert np.allclose(coefficient, [3.0])
    assert np.allclose(residual, unique)
    assert report["retained_energy_fraction"] > 0


def test_safe_cone_changes_direction_instead_of_crossing_boundary() -> None:
    design = np.eye(2)
    target = np.asarray([1.0, 1.0])
    safety = np.asarray([[-1.0, 0.0]])
    fit = fit_safe_cone_ridge(
        design, target, np.ones(2), safety, np.asarray([0.0]), ridge=0.01,
    )
    assert fit.unconstrained_violations == 1
    assert safety @ fit.coefficient >= -1e-7
    assert fit.coefficient[1] > 0.9
    assert abs(fit.coefficient[0]) < 1e-6


def test_zero_update_is_returned_when_every_useful_direction_is_unsafe() -> None:
    fit = fit_safe_cone_ridge(
        np.asarray([[1.0]]), np.asarray([1.0]), np.ones(1),
        np.asarray([[-1.0]]), np.asarray([0.0]), ridge=0.1,
    )
    assert abs(float(fit.coefficient[0])) < 1e-6
    assert fit.maximum_constraint_violation <= 1e-7


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware safe-cone metric contracts")


if __name__ == "__main__":
    main()
