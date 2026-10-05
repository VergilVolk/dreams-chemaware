"""Convex utilities for safety-constrained ChemAware metric transfer.

The old observable tangent audit treated already-correct retrieval margins as
zero-valued regression examples.  A small mean-squared error on those examples
does not prevent a few near-boundary decisions from changing sign.  This
module instead solves a ridge problem inside an explicit polyhedral safety
cone.  The zero update is always feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize


@dataclass(frozen=True)
class SafeConeFit:
    coefficient: np.ndarray
    column_scale: np.ndarray
    converged: bool
    iterations: int
    objective: float
    maximum_constraint_violation: float
    active_constraints: int
    unconstrained_violations: int


def control_orthogonal_residual(
    correct: np.ndarray,
    controls: np.ndarray,
    weight: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Remove the component of a correct target explained by matched controls."""

    y = np.asarray(correct, dtype=np.float64)
    nuisance = np.asarray(controls, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if y.ndim != 1 or nuisance.ndim != 2 or nuisance.shape[0] != len(y):
        raise ValueError("correct and control targets are not aligned")
    if w.shape != y.shape or np.any(w <= 0) or ridge < 0:
        raise ValueError("invalid residualization weights or ridge")
    root = np.sqrt(w)
    xw = nuisance * root[:, None]
    yw = y * root
    gram = xw.T @ xw + float(ridge) * np.eye(nuisance.shape[1])
    coefficient = np.linalg.solve(gram, xw.T @ yw)
    residual = y - nuisance @ coefficient
    denominator = float(np.sum(w * y * y))
    report = {
        "correct_weighted_energy": denominator,
        "orthogonal_weighted_energy": float(np.sum(w * residual * residual)),
        "retained_energy_fraction": (
            float(np.sum(w * residual * residual)) / denominator if denominator else 0.0
        ),
        "weighted_control_correlation_after": float(
            np.linalg.norm(nuisance.T @ (w * residual))
        ),
    }
    return residual, coefficient, report


def fit_safe_cone_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    safety_design: np.ndarray,
    safety_lower_bound: np.ndarray,
    *,
    ridge: float,
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> SafeConeFit:
    """Fit weighted ridge subject to ``safety_design @ w >= lower_bound``.

    Columns are standardized from the action design only.  SLSQP is applied to
    the resulting strictly convex quadratic.  The returned solution is rejected
    if it is not feasible to numerical tolerance; callers must never silently
    use a failed optimization result.
    """

    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    a = np.asarray(safety_design, dtype=np.float64)
    b = np.asarray(safety_lower_bound, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),):
        raise ValueError("action arrays are not aligned")
    if a.ndim != 2 or a.shape[1] != x.shape[1] or b.shape != (len(a),):
        raise ValueError("safety arrays are not aligned")
    if not len(x) or not len(a) or np.any(w <= 0) or ridge <= 0:
        raise ValueError("nonempty data, positive weights, and positive ridge are required")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(a)):
        raise ValueError("design contains non-finite values")
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(b)) or np.any(b > tolerance):
        raise ValueError("safety cone must contain the zero update")

    column_scale = np.sqrt(np.average(x * x, axis=0, weights=w))
    active = column_scale > 1e-10
    if not np.any(active):
        raise RuntimeError("action design has no effective columns")
    safe_scale = np.where(active, column_scale, 1.0)
    xs = x[:, active] / safe_scale[active]
    aas = a[:, active] / safe_scale[active]
    hessian = (xs.T * w) @ xs + float(ridge) * np.eye(xs.shape[1])
    linear = xs.T @ (w * y)

    def objective(value: np.ndarray) -> float:
        return float(0.5 * value @ hessian @ value - linear @ value)

    def gradient(value: np.ndarray) -> np.ndarray:
        return hessian @ value - linear

    unconstrained = np.linalg.solve(hessian, linear)
    unconstrained_violation = b - aas @ unconstrained
    unconstrained_violations = int(np.sum(unconstrained_violation > tolerance))
    if unconstrained_violations == 0:
        solution = unconstrained
        success = True
        iterations = 0
        final_objective = objective(solution)
    else:
        constraint = LinearConstraint(aas, b, np.full(len(b), np.inf))
        result = minimize(
            objective,
            np.zeros(xs.shape[1], dtype=np.float64),
            jac=gradient,
            method="SLSQP",
            constraints=(constraint,),
            options={"ftol": tolerance, "maxiter": int(max_iterations), "disp": False},
        )
        solution = np.asarray(result.x, dtype=np.float64)
        success = bool(result.success)
        iterations = int(result.nit)
        final_objective = float(result.fun)

    violation = b - aas @ solution
    maximum_violation = float(max(0.0, np.max(violation)))
    if not success or maximum_violation > max(1e-7, 10 * tolerance):
        raise RuntimeError(
            "safe-cone quadratic optimization failed: "
            f"converged={success} max_violation={maximum_violation:.3e}"
        )
    coefficient = np.zeros(x.shape[1], dtype=np.float64)
    coefficient[active] = solution / safe_scale[active]
    slack = a @ coefficient - b
    return SafeConeFit(
        coefficient=coefficient,
        column_scale=safe_scale,
        converged=success,
        iterations=iterations,
        objective=final_objective,
        maximum_constraint_violation=maximum_violation,
        active_constraints=int(np.sum(slack <= max(1e-6, 100 * tolerance))),
        unconstrained_violations=unconstrained_violations,
    )

