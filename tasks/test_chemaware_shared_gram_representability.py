from __future__ import annotations

import numpy as np

from chemaware_shared_gram_representability_core import (
    apply_normalized_updates,
    build_edge_operator,
    solve_tangent_projection,
)


def test_single_edge_is_representable() -> None:
    embeddings = {
        1: np.asarray([1.0, 0.0, 0.0]),
        2: np.asarray([0.0, 1.0, 0.0]),
    }
    operator = build_edge_operator(np.asarray([1]), np.asarray([2]), embeddings)
    result = solve_tangent_projection(operator, np.asarray([0.2]), ridge=1e-10)
    assert result["explained_energy_fraction"] > 1 - 1e-12
    assert abs(result["projected"][0] - 0.2) < 1e-9
    finite = apply_normalized_updates(operator, result["updates"])
    assert np.isclose(np.linalg.norm(finite[1]), 1.0)
    assert np.isclose(np.linalg.norm(finite[2]), 1.0)


def test_contradictory_duplicate_shared_edge_is_not_fully_representable() -> None:
    embeddings = {
        1: np.asarray([1.0, 0.0, 0.0]),
        2: np.asarray([0.0, 1.0, 0.0]),
    }
    operator = build_edge_operator(
        np.asarray([1, 2]), np.asarray([2, 1]), embeddings,
    )
    result = solve_tangent_projection(operator, np.asarray([0.2, -0.2]), ridge=1e-10)
    assert result["explained_energy_fraction"] < 1e-10
    assert np.max(np.abs(result["projected"])) < 1e-9


def test_shared_node_couples_multiple_edges() -> None:
    root = 1.0 / np.sqrt(2.0)
    embeddings = {
        1: np.asarray([1.0, 0.0, 0.0]),
        2: np.asarray([0.0, 1.0, 0.0]),
        3: np.asarray([0.0, root, root]),
    }
    operator = build_edge_operator(
        np.asarray([1, 1]), np.asarray([2, 3]), embeddings,
    )
    result = solve_tangent_projection(operator, np.asarray([0.1, -0.05]), ridge=1e-8)
    assert result["explained_energy_fraction"] > 0.999
    assert result["maximum_tangent_error"] < 1e-9

