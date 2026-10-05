"""Linearized representability of pair-score corrections by a shared embedding.

For unit embeddings ``z_i`` and tangent updates ``u_i`` (``z_i.T u_i = 0``),
the first-order change of a shared dot-product score is

    delta K_ij = z_i.T u_j + u_i.T z_j.

This module builds the Gram matrix of that linear operator on a sparse edge
set.  It deliberately gives every observed spectrum its own free update.  The
result is therefore a geometric ceiling, not a learnable spectrum encoder.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


@dataclass(frozen=True)
class EdgeOperator:
    """Sparse shared-embedding tangent operator represented by ``A A^T``."""

    edge_left: np.ndarray
    edge_right: np.ndarray
    node_rows: np.ndarray
    node_embeddings: np.ndarray
    edge_gram: sparse.csr_matrix
    edge_node_gradients: tuple[tuple[tuple[int, np.ndarray], ...], ...]


def _unit_rows(values: np.ndarray, *, tolerance: float = 2e-4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("node embeddings must be a nonempty matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or float(np.max(np.abs(norms - 1.0))) > tolerance:
        raise ValueError("node embeddings must be finite unit vectors")
    return values / norms[:, None]


def build_edge_operator(
    edge_left_rows: np.ndarray,
    edge_right_rows: np.ndarray,
    embedding_by_row: dict[int, np.ndarray],
) -> EdgeOperator:
    """Build ``A A^T`` for score constraints on unordered shared-score edges."""

    left = np.asarray(edge_left_rows, dtype=np.int64)
    right = np.asarray(edge_right_rows, dtype=np.int64)
    if left.ndim != 1 or right.shape != left.shape or len(left) == 0:
        raise ValueError("edge endpoints must be aligned nonempty vectors")
    if np.any(left == right):
        raise ValueError("self-edges do not carry a normalized-cosine tangent signal")

    rows = np.asarray(sorted(set(map(int, left)) | set(map(int, right))), dtype=np.int64)
    missing = [int(row) for row in rows if int(row) not in embedding_by_row]
    if missing:
        raise KeyError(f"missing embeddings for {len(missing)} edge nodes")
    embeddings = _unit_rows(np.stack([embedding_by_row[int(row)] for row in rows]))
    position = {int(row): index for index, row in enumerate(rows)}

    contributions: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    per_edge: list[tuple[tuple[int, np.ndarray], ...]] = []
    for edge, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        i, j = position[int(left_row)], position[int(right_row)]
        zi, zj = embeddings[i], embeddings[j]
        cosine = float(zi @ zj)
        gradient_i = zj - cosine * zi
        gradient_j = zi - cosine * zj
        contributions[i].append((edge, gradient_i))
        contributions[j].append((edge, gradient_j))
        per_edge.append(((i, gradient_i), (j, gradient_j)))

    gram_row: list[np.ndarray] = []
    gram_col: list[np.ndarray] = []
    gram_value: list[np.ndarray] = []
    for items in contributions.values():
        edges = np.asarray([edge for edge, _ in items], dtype=np.int64)
        gradients = np.stack([gradient for _, gradient in items])
        block = gradients @ gradients.T
        gram_row.append(np.repeat(edges, len(edges)))
        gram_col.append(np.tile(edges, len(edges)))
        gram_value.append(block.reshape(-1))

    n_edges = len(left)
    gram = sparse.coo_matrix(
        (
            np.concatenate(gram_value),
            (np.concatenate(gram_row), np.concatenate(gram_col)),
        ),
        shape=(n_edges, n_edges),
        dtype=np.float64,
    ).tocsr()
    gram = ((gram + gram.T) * 0.5).tocsr()
    return EdgeOperator(
        edge_left=left,
        edge_right=right,
        node_rows=rows,
        node_embeddings=embeddings,
        edge_gram=gram,
        edge_node_gradients=tuple(per_edge),
    )


def solve_tangent_projection(
    operator: EdgeOperator,
    target: np.ndarray,
    *,
    ridge: float,
    rtol: float = 1e-9,
    maxiter: int = 20_000,
) -> dict[str, object]:
    """Project an edge-score target onto the regularized shared tangent space."""

    target = np.asarray(target, dtype=np.float64)
    if target.shape != (operator.edge_gram.shape[0],):
        raise ValueError("target must have one finite value per edge")
    if np.any(~np.isfinite(target)):
        raise ValueError("target contains non-finite values")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    system = operator.edge_gram + sparse.eye(len(target), format="csr") * float(ridge)
    dual, info = cg(system, target, rtol=rtol, atol=0.0, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"conjugate-gradient projection did not converge: info={info}")
    projected = np.asarray(operator.edge_gram @ dual).reshape(-1)

    updates = np.zeros_like(operator.node_embeddings)
    for edge, terms in enumerate(operator.edge_node_gradients):
        for node, gradient in terms:
            updates[node] += dual[edge] * gradient
    tangent_error = np.abs(np.einsum("ij,ij->i", operator.node_embeddings, updates))
    if float(np.max(tangent_error, initial=0.0)) > 2e-7:
        raise RuntimeError("reconstructed update left the unit-sphere tangent space")

    residual = target - projected
    target_energy = float(target @ target)
    projected_energy = float(projected @ projected)
    residual_energy = float(residual @ residual)
    denominator = float(np.linalg.norm(target) * np.linalg.norm(projected))
    return {
        "dual": dual,
        "projected": projected,
        "updates": updates,
        "target_energy": target_energy,
        "projected_energy": projected_energy,
        "residual_energy": residual_energy,
        "explained_energy_fraction": (
            1.0 - residual_energy / target_energy if target_energy > 0 else 1.0
        ),
        "target_projected_cosine": (
            float(target @ projected) / denominator if denominator > 0 else 0.0
        ),
        "update_norms": np.linalg.norm(updates, axis=1),
        "maximum_tangent_error": float(np.max(tangent_error, initial=0.0)),
    }


def apply_normalized_updates(operator: EdgeOperator, updates: np.ndarray) -> dict[int, np.ndarray]:
    """Return finite normalized node embeddings after applying tangent updates."""

    updates = np.asarray(updates, dtype=np.float64)
    if updates.shape != operator.node_embeddings.shape:
        raise ValueError("updates are not aligned with operator nodes")
    values = operator.node_embeddings + updates
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("finite update produced invalid embeddings")
    values = values / norms[:, None]
    return {int(row): values[index] for index, row in enumerate(operator.node_rows)}

