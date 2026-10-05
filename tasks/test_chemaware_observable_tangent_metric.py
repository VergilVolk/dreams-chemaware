"""CPU contracts for the ChemAware observable tangent metric field."""

from __future__ import annotations

import numpy as np

from chemaware_observable_tangent_metric_core import (
    apply_tangent_field,
    edge_design,
    fit_tangent_field,
    margin_design,
    tangent_basis_from_edges,
)


def test_edge_design_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(8, 12)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    left = np.asarray([0, 1, 2, 3, 4]); right = np.asarray([5, 6, 7, 4, 0])
    q = tangent_basis_from_edges(z, left, right, rank=4, seed=3)
    gates = np.c_[np.ones(len(z)), np.tanh(z[:, :2])]
    design = edge_design(z, q, gates, left, right)
    coefficient = rng.normal(scale=0.02, size=design.shape[1])
    packed = 4 * 5 // 2
    matrices = []
    a, b = np.triu_indices(4)
    for gate in range(gates.shape[1]):
        matrix = np.zeros((4, 4)); values = coefficient[gate * packed : (gate + 1) * packed]
        matrix[a, b] = values; matrix[b, a] = values; matrices.append(matrix)
    from chemaware_observable_tangent_metric_core import TangentFieldFit
    fit = TangentFieldFit(q, np.stack(matrices), np.ones_like(coefficient), 0.0, 0.0, len(coefficient))
    epsilon = 1e-5
    tiny = TangentFieldFit(q, fit.matrices * epsilon, fit.column_scale, 0.0, 0.0, len(coefficient))
    updated, _ = apply_tangent_field(z, gates, tiny)
    finite = (np.einsum("ij,ij->i", updated[left], updated[right])
              - np.einsum("ij,ij->i", z[left], z[right])) / epsilon
    assert np.max(np.abs(finite - design @ coefficient)) < 2e-5


def test_margin_design_and_ridge_recover_signal() -> None:
    rng = np.random.default_rng(11)
    z = rng.normal(size=(20, 16)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    left = np.repeat(np.arange(5), 3); right = np.arange(5, 20)
    q = tangent_basis_from_edges(z, left, right, rank=5, seed=9)
    gates = np.ones((len(z), 1))
    edge = edge_design(z, q, gates, left, right)
    positive = np.arange(0, 15, 3); negative = np.asarray([i for i in range(15) if i % 3 != 0])
    positive = np.repeat(positive, 2)
    design = margin_design(edge, positive, negative)
    truth = rng.normal(scale=0.1, size=design.shape[1]); target = design @ truth
    fit = fit_tangent_field(design, target, np.ones(len(target)), basis=q, gates=1, ridge=1e-9)
    predicted = edge @ fit.matrices.reshape(-1)[np.ravel_multi_index(np.triu_indices(5), (5, 5))]
    observed_margin = predicted[positive] - predicted[negative]
    assert np.sqrt(np.mean((observed_margin - target) ** 2)) < 1e-5


def test_application_is_shared_normalized_and_tangent() -> None:
    rng = np.random.default_rng(19)
    z = rng.normal(size=(10, 18)); z /= np.linalg.norm(z, axis=1, keepdims=True)
    left = np.arange(5); right = np.arange(5, 10)
    q = tangent_basis_from_edges(z, left, right, rank=4, seed=5)
    gates = np.c_[np.ones(10), np.linspace(-1, 1, 10)]
    design = edge_design(z, q, gates, left, right)
    fit = fit_tangent_field(design, np.linspace(-0.1, 0.1, 5), np.ones(5), basis=q, gates=2, ridge=0.1)
    first, audit = apply_tangent_field(z, gates, fit)
    second, _ = apply_tangent_field(z, gates, fit)
    assert np.allclose(first, second)
    assert np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0)) < 1e-10
    assert audit["maximum_tangent_error"] < 1e-10


if __name__ == "__main__":
    test_edge_design_matches_finite_difference()
    test_margin_design_and_ridge_recover_signal()
    test_application_is_shared_normalized_and_tangent()
    print("PASS: 3 observable tangent metric contracts")
