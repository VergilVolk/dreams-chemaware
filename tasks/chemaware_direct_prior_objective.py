"""Mathematical core of candidate-centred direct chemical supervision.

For query q with M_q candidates, let s_q be current shared-embedding scores,
b_q frozen official scores, H_q = I - 11^T/M_q, and c_q an independently
compiled, already centred chemical residual.  The treatment objective is

    L_q = mean_m huber_delta([H_q(s_q-b_q)]_m - alpha*c_qm).

Only active queries contribute and queries are equally weighted.  Alpha zero
is implemented as a literal absent treatment, not as a preservation target.
Consequently the chemical loss cannot create query-wise common score drift and
cannot spend capacity on a ranking-irrelevant offset.
"""

from __future__ import annotations

import numpy as np

from chemaware_direct_chemical_prior_core import validate_query_ptr


def direct_prior_loss_and_gradient(
    student_score: np.ndarray,
    official_score: np.ndarray,
    centered_prior: np.ndarray,
    query_ptr: np.ndarray,
    active_query: np.ndarray,
    *,
    alpha: float,
    huber_delta: float,
) -> tuple[float, np.ndarray]:
    """Return query-equal Huber loss and its exact score-space gradient."""
    student = np.asarray(student_score, dtype=np.float64)
    official = np.asarray(official_score, dtype=np.float64)
    prior = np.asarray(centered_prior, dtype=np.float64)
    if student.shape != official.shape or student.shape != prior.shape or student.ndim != 1:
        raise ValueError("student, official, and prior scores must be aligned vectors")
    if not all(np.all(np.isfinite(value)) for value in (student, official, prior)):
        raise ValueError("direct-prior objective inputs must be finite")
    ptr = validate_query_ptr(query_ptr, len(student))
    active = np.asarray(active_query, dtype=bool)
    if active.shape != (len(ptr) - 1,):
        raise ValueError("active-query mask does not align with query_ptr")
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("alpha must be nonnegative and Huber delta positive")

    gradient = np.zeros_like(student)
    # Alpha zero is the clean/selected-clean control: the chemical objective is
    # absent, rather than secretly pulling a concurrently trained model back to
    # the official score geometry.
    if alpha == 0 or not np.any(active):
        return 0.0, gradient

    losses = []
    active_count = int(np.sum(active))
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        if not active[query]:
            continue
        left, right = int(left), int(right)
        block_prior = prior[left:right]
        if abs(float(np.sum(block_prior))) > 1e-5:
            raise ValueError("chemical prior must be candidate-centred per query")
        residual = student[left:right] - official[left:right]
        residual = residual - np.mean(residual)
        error = residual - float(alpha) * block_prior
        absolute = np.abs(error)
        huber = np.where(
            absolute <= huber_delta,
            0.5 * error**2 / huber_delta,
            absolute - 0.5 * huber_delta,
        )
        losses.append(float(np.mean(huber)))
        derivative = np.where(
            absolute <= huber_delta,
            error / huber_delta,
            np.sign(error),
        )
        # H_q is symmetric and idempotent.  Projecting the derivative through
        # H_q makes its sum exactly zero, so no common score offset is induced.
        derivative = derivative - np.mean(derivative)
        gradient[left:right] = derivative / ((right - left) * active_count)
    return float(np.mean(losses)), gradient

