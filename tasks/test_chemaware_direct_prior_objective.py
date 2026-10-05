"""Finite-difference and invariance contracts for direct-prior injection."""

from __future__ import annotations

import numpy as np

from chemaware_direct_prior_objective import direct_prior_loss_and_gradient


def main() -> None:
    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    official = np.asarray([0.4, 0.2, -0.1, 0.3, -0.2], dtype=np.float64)
    student = official + np.asarray([0.01, -0.02, 0.015, 0.03, -0.01])
    prior = np.asarray([0.6, -0.2, -0.4, 0.5, -0.5], dtype=np.float64)
    active = np.asarray([True, True])
    loss, gradient = direct_prior_loss_and_gradient(
        student,
        official,
        prior,
        ptr,
        active,
        alpha=0.04,
        huber_delta=0.02,
    )
    assert loss > 0
    assert abs(float(np.sum(gradient[:3]))) < 1e-12
    assert abs(float(np.sum(gradient[3:]))) < 1e-12

    # Query-wise score offsets cannot alter either loss or gradient.
    shifted = student + np.asarray([7.0, 7.0, 7.0, -4.0, -4.0])
    shifted_loss, shifted_gradient = direct_prior_loss_and_gradient(
        shifted,
        official,
        prior,
        ptr,
        active,
        alpha=0.04,
        huber_delta=0.02,
    )
    assert np.isclose(loss, shifted_loss)
    assert np.allclose(gradient, shifted_gradient)

    # Exact finite differences verify the derivative, including centring.
    epsilon = 1e-6
    numerical = np.empty_like(student)
    for index in range(len(student)):
        plus, minus = student.copy(), student.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss, _ = direct_prior_loss_and_gradient(
            plus, official, prior, ptr, active, alpha=0.04, huber_delta=0.02
        )
        minus_loss, _ = direct_prior_loss_and_gradient(
            minus, official, prior, ptr, active, alpha=0.04, huber_delta=0.02
        )
        numerical[index] = (plus_loss - minus_loss) / (2 * epsilon)
    assert np.allclose(gradient, numerical, atol=2e-9)

    # At official initialization, gradient descent moves along the chemical
    # target rather than generating ranking-irrelevant score drift.
    _, initial_gradient = direct_prior_loss_and_gradient(
        official,
        official,
        prior,
        ptr,
        active,
        alpha=0.01,
        huber_delta=0.02,
    )
    assert float(np.dot(-initial_gradient, prior)) > 0

    # The matched alpha-zero arm is a literal no-op even after clean training
    # has moved the student away from the official initialization.
    zero_loss, zero_gradient = direct_prior_loss_and_gradient(
        student,
        official,
        prior,
        ptr,
        active,
        alpha=0.0,
        huber_delta=0.02,
    )
    assert zero_loss == 0.0
    assert np.array_equal(zero_gradient, np.zeros_like(student))

    print("PASS: direct candidate-centred prior objective contracts")


if __name__ == "__main__":
    main()
