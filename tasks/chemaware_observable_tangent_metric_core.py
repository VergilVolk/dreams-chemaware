"""Observable, shared-embedding tangent metric fields for ChemAware.

The module implements a deliberately small and falsifiable adapter.  Given a
unit DreaMS embedding ``z`` it produces

    T(z, x) = normalize(z + P_z Q B(x) Q.T z),
    B(x) = sum_r psi_r(x) B_r,

where ``P_z = I - z z.T``, ``Q`` is a target-free low-rank basis, the gates
``psi`` depend on one clean spectrum only, and every ``B_r`` is symmetric.
For fixed ``Q`` and gates, the first-order change of a candidate margin is
linear in the packed matrices ``B_r``.  Fitting the field is therefore a
convex ridge problem rather than a 116M-parameter fine-tuning run.

Candidate structures may define training targets, but never enter ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.extmath import randomized_svd


@dataclass(frozen=True)
class TangentFieldFit:
    """A fitted conditional low-rank field."""

    basis: np.ndarray
    matrices: np.ndarray
    column_scale: np.ndarray
    training_rms: float
    training_weighted_rms: float
    effective_columns: int


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("expected a nonempty embedding matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("embedding matrix contains invalid rows")
    return values / norms[:, None]


def tangent_basis_from_edges(
    embeddings: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    *,
    rank: int,
    seed: int,
) -> np.ndarray:
    """Fit a target-free basis to score gradients on training edges only."""

    z = unit_rows(embeddings)
    left = np.asarray(edge_left, dtype=np.int64)
    right = np.asarray(edge_right, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("edge endpoints must be aligned nonempty vectors")
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= len(z)) or np.any(right >= len(z)):
        raise IndexError("edge endpoint is outside the embedding matrix")
    maximum_rank = min(z.shape[1], 2 * len(left))
    if not 1 <= rank <= maximum_rank:
        raise ValueError(f"rank must lie in [1, {maximum_rank}]")
    cosine = np.einsum("ij,ij->i", z[left], z[right])
    gradients = np.concatenate(
        (
            z[right] - cosine[:, None] * z[left],
            z[left] - cosine[:, None] * z[right],
        ),
        axis=0,
    ).astype(np.float32)
    _u, _s, vt = randomized_svd(
        gradients, n_components=rank, n_iter=5, random_state=seed,
    )
    basis = np.asarray(vt.T, dtype=np.float64)
    error = np.max(np.abs(basis.T @ basis - np.eye(rank)))
    if not np.isfinite(error) or error > 1e-5:
        raise RuntimeError("tangent basis is not orthonormal")
    return basis


def fit_clean_gates(
    train_features: np.ndarray,
    all_features: np.ndarray,
    *,
    components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return bounded target-free gates fitted on training spectrum rows."""

    train = np.asarray(train_features, dtype=np.float64)
    all_values = np.asarray(all_features, dtype=np.float64)
    if train.ndim != 2 or all_values.ndim != 2 or train.shape[1] != all_values.shape[1]:
        raise ValueError("clean feature matrices are not aligned")
    if not 1 <= components <= min(train.shape):
        raise ValueError("invalid clean-gate component count")
    center = np.mean(train, axis=0)
    train_centered = train - center
    _u, singular, vt = randomized_svd(
        train_centered, n_components=components, n_iter=5, random_state=seed,
    )
    train_coordinates = train_centered @ vt.T
    scale = np.std(train_coordinates, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    coordinates = (all_values - center) @ vt.T / scale
    gates = np.tanh(coordinates).astype(np.float64)
    return gates, {
        "components": int(components),
        "singular_values": singular.astype(float).tolist(),
        "training_coordinate_scale": scale.astype(float).tolist(),
        "maximum_absolute_gate": float(np.max(np.abs(gates))),
    }


def _packed_symmetric_bilinear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Features whose dot product with upper-triangle(B) is a.T B b."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("bilinear operands must be aligned matrices")
    rank = a.shape[1]
    upper_a, upper_b = np.triu_indices(rank)
    output = a[:, upper_a] * b[:, upper_b]
    off = upper_a != upper_b
    output[:, off] += a[:, upper_b[off]] * b[:, upper_a[off]]
    return output


def edge_design(
    embeddings: np.ndarray,
    basis: np.ndarray,
    gates: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
) -> np.ndarray:
    """Linear design for first-order shared-score changes on directed edges."""

    z = unit_rows(embeddings)
    q = z @ np.asarray(basis, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.float64)
    left = np.asarray(edge_left, dtype=np.int64)
    right = np.asarray(edge_right, dtype=np.int64)
    if gates.ndim != 2 or gates.shape[0] != len(z):
        raise ValueError("gates must have one row per embedding")
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("edge endpoints are not aligned")
    cosine = np.einsum("ij,ij->i", z[left], z[right])
    left_gradient = q[right] - cosine[:, None] * q[left]
    right_gradient = q[left] - cosine[:, None] * q[right]
    left_packed = _packed_symmetric_bilinear(left_gradient, q[left])
    right_packed = _packed_symmetric_bilinear(right_gradient, q[right])
    blocks = [
        gates[left, gate, None] * left_packed + gates[right, gate, None] * right_packed
        for gate in range(gates.shape[1])
    ]
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if np.any(~np.isfinite(output)):
        raise RuntimeError("edge design contains non-finite values")
    return output


def margin_design(
    edge_features: np.ndarray,
    positive_edge: np.ndarray,
    negative_edge: np.ndarray,
) -> np.ndarray:
    """Convert candidate-edge features into positive-minus-negative margins."""

    edge_features = np.asarray(edge_features)
    positive = np.asarray(positive_edge, dtype=np.int64)
    negative = np.asarray(negative_edge, dtype=np.int64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("margin edge indices are not aligned")
    return np.asarray(edge_features[positive] - edge_features[negative], dtype=np.float64)


def _unpack_symmetric(values: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    expected = rank * (rank + 1) // 2
    if values.shape[-1] != expected:
        raise ValueError("packed symmetric matrix has the wrong width")
    output = np.zeros(values.shape[:-1] + (rank, rank), dtype=np.float64)
    a, b = np.triu_indices(rank)
    output[..., a, b] = values
    output[..., b, a] = values
    return output


def fit_tangent_field(
    design: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    basis: np.ndarray,
    gates: int,
    ridge: float,
) -> TangentFieldFit:
    """Fit conditional symmetric metric matrices by weighted ridge."""

    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),):
        raise ValueError("training arrays are not aligned")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(w)):
        raise ValueError("training arrays contain non-finite values")
    if np.any(w <= 0) or ridge <= 0:
        raise ValueError("weights and ridge must be positive")
    rank = int(np.asarray(basis).shape[1])
    packed = rank * (rank + 1) // 2
    if x.shape[1] != gates * packed:
        raise ValueError("design width does not match rank and gate count")

    column_scale = np.sqrt(np.average(x * x, axis=0, weights=w))
    active_columns = column_scale > 1e-10
    safe_scale = np.where(active_columns, column_scale, 1.0)
    scaled = x[:, active_columns] / safe_scale[active_columns]
    model = Ridge(
        alpha=float(ridge), fit_intercept=False, solver="lsqr", tol=1e-9, max_iter=20_000,
    )
    model.fit(scaled, y, sample_weight=w)
    packed_coefficient = np.zeros(x.shape[1], dtype=np.float64)
    packed_coefficient[active_columns] = np.asarray(model.coef_) / safe_scale[active_columns]
    prediction = x @ packed_coefficient
    matrices = _unpack_symmetric(packed_coefficient.reshape(gates, packed), rank)
    return TangentFieldFit(
        basis=np.asarray(basis, dtype=np.float64),
        matrices=matrices,
        column_scale=safe_scale,
        training_rms=float(np.sqrt(np.mean((y - prediction) ** 2))),
        training_weighted_rms=float(np.sqrt(np.average((y - prediction) ** 2, weights=w))),
        effective_columns=int(np.sum(active_columns)),
    )


def fit_tangent_fields(
    design: np.ndarray,
    targets: np.ndarray,
    weight: np.ndarray,
    *,
    basis: np.ndarray,
    gates: int,
    ridge: float,
) -> list[TangentFieldFit]:
    """Multi-target version of :func:`fit_tangent_field` with shared scaling."""

    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != len(x):
        raise ValueError("targets must have shape (observations, arms)")
    if x.ndim != 2 or w.shape != (len(x),):
        raise ValueError("training arrays are not aligned")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(w)):
        raise ValueError("training arrays contain non-finite values")
    if np.any(w <= 0) or ridge <= 0:
        raise ValueError("weights and ridge must be positive")
    rank = int(np.asarray(basis).shape[1])
    packed = rank * (rank + 1) // 2
    if x.shape[1] != gates * packed:
        raise ValueError("design width does not match rank and gate count")

    column_scale = np.sqrt(np.average(x * x, axis=0, weights=w))
    active_columns = column_scale > 1e-10
    safe_scale = np.where(active_columns, column_scale, 1.0)
    scaled = x[:, active_columns] / safe_scale[active_columns]
    model = Ridge(
        alpha=float(ridge), fit_intercept=False, solver="lsqr", tol=1e-9, max_iter=20_000,
    )
    model.fit(scaled, y, sample_weight=w)
    coefficients = np.zeros((y.shape[1], x.shape[1]), dtype=np.float64)
    coefficients[:, active_columns] = np.asarray(model.coef_) / safe_scale[active_columns]
    prediction = x @ coefficients.T
    output = []
    for arm in range(y.shape[1]):
        output.append(TangentFieldFit(
            basis=np.asarray(basis, dtype=np.float64),
            matrices=_unpack_symmetric(coefficients[arm].reshape(gates, packed), rank),
            column_scale=safe_scale,
            training_rms=float(np.sqrt(np.mean((y[:, arm] - prediction[:, arm]) ** 2))),
            training_weighted_rms=float(np.sqrt(np.average(
                (y[:, arm] - prediction[:, arm]) ** 2, weights=w,
            ))),
            effective_columns=int(np.sum(active_columns)),
        ))
    return output


def apply_tangent_field(
    embeddings: np.ndarray,
    gates: np.ndarray,
    fit: TangentFieldFit,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the field once and return finite normalized shared embeddings."""

    z = unit_rows(embeddings)
    gates = np.asarray(gates, dtype=np.float64)
    if gates.shape != (len(z), len(fit.matrices)):
        raise ValueError("application gates are not aligned with the fitted field")
    coordinates = z @ fit.basis
    local = np.zeros_like(coordinates)
    for gate, matrix in enumerate(fit.matrices):
        local += gates[:, gate, None] * (coordinates @ matrix.T)
    ambient = local @ fit.basis.T
    tangent = ambient - z * np.einsum("ij,ij->i", z, ambient)[:, None]
    values = unit_rows(z + tangent)
    preservation = np.einsum("ij,ij->i", z, values)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    return values, {
        "mean_update_l2": float(np.mean(tangent_norm)),
        "q95_update_l2": float(np.quantile(tangent_norm, 0.95)),
        "maximum_update_l2": float(np.max(tangent_norm)),
        "mean_preservation_cosine": float(np.mean(preservation)),
        "q01_preservation_cosine": float(np.quantile(preservation, 0.01)),
        "minimum_preservation_cosine": float(np.min(preservation)),
        "maximum_tangent_error": float(np.max(np.abs(np.einsum("ij,ij->i", z, tangent)))),
    }
