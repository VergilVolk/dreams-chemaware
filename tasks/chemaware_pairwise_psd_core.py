"""Cross-fit pairwise PSD metric updates for shared ChemAware embeddings."""
from __future__ import annotations

import numpy as np


def unit_rows(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(output, norm, out=np.zeros_like(output), where=norm > 1e-12)


def formula_balanced_pair_operator(
    query: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    formula: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Average symmetric query/(positive-negative) operators by formula."""
    query = np.asarray(query, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    formula = np.asarray(formula).astype(str)
    if (
        query.ndim != 2
        or positive.shape != query.shape
        or negative.shape != query.shape
        or formula.shape != (len(query),)
        or not np.isfinite(query).all()
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise ValueError("pair-operator inputs are not aligned finite matrices")
    if weight is None:
        weight_value = np.ones(len(query), dtype=np.float64)
    else:
        weight_value = np.asarray(weight, dtype=np.float64)
        if weight_value.shape != (len(query),) or np.any(weight_value < 0) or not np.isfinite(weight_value).all():
            raise ValueError("pair-operator weights must be finite and nonnegative")
    formula_operators = []
    formula_weights = []
    per_query_margin = np.sum(query * (positive - negative), axis=1)
    for name in np.unique(formula):
        index = np.flatnonzero(formula == name)
        local_weight = weight_value[index]
        denominator = float(local_weight.sum())
        if denominator <= 0:
            continue
        local_query = query[index]
        local_difference = positive[index] - negative[index]
        cross = (local_query * local_weight[:, None]).T @ local_difference / denominator
        formula_operators.append((cross + cross.T) * 0.5)
        formula_weights.append(float(np.mean(local_weight)))
    if len(formula_operators) < 2:
        raise ValueError("pair operator needs at least two formulas")
    operator = np.mean(np.stack(formula_operators), axis=0)
    return operator, {
        "queries": int(len(query)),
        "formulas": int(len(formula_operators)),
        "weight_mean": float(weight_value.mean()),
        "weight_nonzero_fraction": float(np.mean(weight_value > 0)),
        "chemical_margin_quantiles": np.quantile(
            per_query_margin, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        ).astype(float).tolist(),
        "operator_frobenius_norm": float(np.linalg.norm(operator)),
        "formula_weight_mean": float(np.mean(formula_weights)),
    }


def cross_split_spectral_consensus(
    first: np.ndarray,
    second: np.ndarray,
    *,
    eigenvalue_floor: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep only spectral update signs reproduced by two independent fits."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("consensus operators must be aligned finite square matrices")
    first = (first + first.T) * 0.5
    second = (second + second.T) * 0.5
    average = (first + second) * 0.5
    _, basis = np.linalg.eigh(average)
    first_projection = np.diag(basis.T @ first @ basis)
    second_projection = np.diag(basis.T @ second @ basis)
    agrees = (
        np.sign(first_projection) == np.sign(second_projection)
    ) & (np.abs(first_projection) > eigenvalue_floor) & (np.abs(second_projection) > eigenvalue_floor)
    consensus_value = np.zeros_like(first_projection)
    consensus_value[agrees] = (
        np.sign(first_projection[agrees])
        * np.sqrt(np.abs(first_projection[agrees] * second_projection[agrees]))
    )
    raw = (basis * consensus_value[None, :]) @ basis.T
    eigenvalue, eigenvector = np.linalg.eigh((raw + raw.T) * 0.5)
    spectral_radius = float(np.max(np.abs(eigenvalue)))
    if spectral_radius <= eigenvalue_floor:
        normalized = np.zeros_like(raw)
        normalized_eigenvalue = np.zeros_like(eigenvalue)
    else:
        normalized = raw / spectral_radius
        normalized_eigenvalue = eigenvalue / spectral_radius
    cosine = float(
        np.sum(first * second)
        / max(float(np.linalg.norm(first) * np.linalg.norm(second)), eigenvalue_floor)
    )
    return normalized.astype(np.float32), {
        "dimension": int(first.shape[0]),
        "reproduced_directions": int(np.sum(agrees)),
        "reproduced_positive_directions": int(np.sum(agrees & (first_projection > 0))),
        "reproduced_negative_directions": int(np.sum(agrees & (first_projection < 0))),
        "split_operator_cosine": cosine,
        "raw_consensus_spectral_radius": spectral_radius,
        "normalized_min_eigenvalue": float(normalized_eigenvalue[0]),
        "normalized_max_eigenvalue": float(normalized_eigenvalue[-1]),
        "first_order_alignment_first": float(np.sum(first * normalized)),
        "first_order_alignment_second": float(np.sum(second * normalized)),
    }


def cross_split_coordinate_consensus(
    first: np.ndarray,
    second: np.ndarray,
    *,
    entry_floor: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    """Cross-fit consensus in the fixed, chemically named rule coordinates."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("coordinate consensus needs aligned finite square matrices")
    first = (first + first.T) * 0.5
    second = (second + second.T) * 0.5
    agrees = (
        (np.sign(first) == np.sign(second))
        & (np.abs(first) > entry_floor)
        & (np.abs(second) > entry_floor)
    )
    raw = np.zeros_like(first)
    raw[agrees] = np.sign(first[agrees]) * np.sqrt(np.abs(first[agrees] * second[agrees]))
    raw = (raw + raw.T) * 0.5
    eigenvalue = np.linalg.eigvalsh(raw)
    spectral_radius = float(np.max(np.abs(eigenvalue)))
    normalized = raw / spectral_radius if spectral_radius > entry_floor else np.zeros_like(raw)
    normalized_eigenvalue = eigenvalue / spectral_radius if spectral_radius > entry_floor else np.zeros_like(eigenvalue)
    upper = np.triu_indices(first.shape[0])
    cosine = float(
        np.sum(first * second)
        / max(float(np.linalg.norm(first) * np.linalg.norm(second)), entry_floor)
    )
    return normalized.astype(np.float32), {
        "dimension": int(first.shape[0]),
        "fixed_rule_coordinate_basis": True,
        "reproduced_upper_triangle_entries": int(np.sum(agrees[upper])),
        "upper_triangle_entries": int(len(upper[0])),
        "reproduced_upper_triangle_fraction": float(np.mean(agrees[upper])),
        "split_operator_cosine": cosine,
        "raw_consensus_spectral_radius": spectral_radius,
        "normalized_min_eigenvalue": float(normalized_eigenvalue[0]),
        "normalized_max_eigenvalue": float(normalized_eigenvalue[-1]),
        "first_order_alignment_first": float(np.sum(first * normalized)),
        "first_order_alignment_second": float(np.sum(second * normalized)),
    }


def compose_psd_update(
    base_transform: np.ndarray,
    normalized_operator: np.ndarray,
    eta: float,
    *,
    eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compose W with sqrt(I + eta*A), retaining one shared PSD map."""
    base = np.asarray(base_transform, dtype=np.float64)
    operator = np.asarray(normalized_operator, dtype=np.float64)
    if (
        base.ndim != 2
        or base.shape[1] != operator.shape[0]
        or operator.shape[0] != operator.shape[1]
        or not 0.0 <= eta < 1.0
    ):
        raise ValueError("invalid PSD update inputs")
    if eta == 0.0:
        return base.astype(np.float32), {
            "eta": 0.0, "metric_min_eigenvalue": 1.0, "metric_max_eigenvalue": 1.0,
        }
    eigenvalue, eigenvector = np.linalg.eigh((operator + operator.T) * 0.5)
    metric_value = np.maximum(1.0 + float(eta) * eigenvalue, eigenvalue_floor)
    square_root = (eigenvector * np.sqrt(metric_value)[None, :]) @ eigenvector.T
    return (base @ square_root).astype(np.float32), {
        "eta": float(eta),
        "metric_min_eigenvalue": float(metric_value[0]),
        "metric_max_eigenvalue": float(metric_value[-1]),
    }
