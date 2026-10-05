"""Conditional Fisher maps for shared ChemAware rule-response embeddings."""
from __future__ import annotations

import numpy as np


def _group_registry(
    feature: np.ndarray,
    identity: np.ndarray,
    formula: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(feature, dtype=np.float64)
    identity = np.asarray(identity).astype(str)
    formula = np.asarray(formula).astype(str)
    if (
        value.ndim != 2
        or identity.shape != (len(value),)
        or formula.shape != (len(value),)
        or not np.isfinite(value).all()
    ):
        raise ValueError("conditional Fisher inputs are not aligned finite arrays")
    unique_identity, inverse = np.unique(identity, return_inverse=True)
    counts = np.bincount(inverse)
    centroids = np.zeros((len(unique_identity), value.shape[1]), dtype=np.float64)
    identity_formula = np.empty(len(unique_identity), dtype=object)
    for group in range(len(unique_identity)):
        index = np.flatnonzero(inverse == group)
        formulas = np.unique(formula[index])
        if len(formulas) != 1:
            raise ValueError("one molecular identity maps to multiple formulas")
        centroids[group] = value[index].mean(axis=0)
        identity_formula[group] = str(formulas[0])
    return value, inverse, counts, centroids, identity_formula.astype(str)


def permute_identity_formulas(
    identity_formula: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Destroy formula membership while preserving its exact size multiset."""
    labels = np.asarray(identity_formula).astype(str)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("formula permutation needs at least two identities")
    rng = np.random.default_rng(seed)
    output = labels[rng.permutation(len(labels))]
    before_name, before_count = np.unique(labels, return_counts=True)
    after_name, after_count = np.unique(output, return_counts=True)
    if not (
        np.array_equal(before_name, after_name)
        and np.array_equal(before_count, after_count)
    ):
        raise RuntimeError("formula-size multiset changed during permutation")
    return output, {
        "seed": int(seed),
        "changed_identity_fraction": float(np.mean(output != labels)),
        "formula_counts_preserved_exactly": True,
    }


def _within_covariance(
    value: np.ndarray,
    inverse: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, int]:
    eligible = np.flatnonzero(counts >= 2)
    if len(eligible) < 2:
        raise ValueError("conditional Fisher fit needs replicated identities")
    covariance = np.zeros((value.shape[1], value.shape[1]), dtype=np.float64)
    for group in eligible:
        local = value[inverse == group]
        residual = local - local.mean(axis=0)
        covariance += residual.T @ residual / float(len(local) - 1)
    return covariance / float(len(eligible)), int(len(eligible))


def _conditional_between_covariance(
    centroids: np.ndarray,
    identity_formula: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    covariance = np.zeros((centroids.shape[1], centroids.shape[1]), dtype=np.float64)
    eligible = 0
    identities = 0
    for formula in np.unique(identity_formula):
        local = centroids[identity_formula == formula]
        if len(local) < 2:
            continue
        residual = local - local.mean(axis=0)
        covariance += residual.T @ residual / float(len(local) - 1)
        eligible += 1
        identities += len(local)
    if eligible < 2:
        raise ValueError("conditional Fisher fit needs multi-identity formulas")
    return covariance / float(eligible), int(eligible), int(identities)


def fit_conditional_fisher_map(
    feature: np.ndarray,
    identity: np.ndarray,
    formula: np.ndarray,
    within_shrinkage: float,
    rank: int,
    *,
    between_mode: str = "conditional",
    eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit top generalized eigenvectors of between signal versus within noise.

    The returned transform is a spectrum-local shared map.  Its columns are
    orthonormal under the regularized within-identity covariance.
    """
    if not 0.0 <= within_shrinkage <= 1.0 or eigenvalue_floor <= 0.0:
        raise ValueError("invalid conditional Fisher regularization")
    value, inverse, counts, centroids, identity_formula = _group_registry(
        feature, identity, formula,
    )
    dimension = value.shape[1]
    if rank < 1 or rank > dimension:
        raise ValueError("conditional Fisher rank is out of range")
    within, replicated = _within_covariance(value, inverse, counts)
    if between_mode == "conditional":
        between, eligible_formulas, between_identities = _conditional_between_covariance(
            centroids, identity_formula,
        )
    elif between_mode == "global":
        residual = centroids - centroids.mean(axis=0)
        between = residual.T @ residual / float(len(centroids) - 1)
        eligible_formulas = 0
        between_identities = len(centroids)
    else:
        raise ValueError(f"unsupported between mode: {between_mode}")

    within_value, within_vector = np.linalg.eigh(within)
    isotropic = float(np.trace(within) / dimension)
    regularized = (
        (1.0 - float(within_shrinkage)) * within_value
        + float(within_shrinkage) * isotropic
    )
    floor = max(float(eigenvalue_floor) * max(isotropic, 1e-12), 1e-12)
    regularized = np.maximum(regularized, floor)
    inverse_sqrt = (
        within_vector * (1.0 / np.sqrt(regularized))[None, :]
    ) @ within_vector.T
    whitened_between = inverse_sqrt @ between @ inverse_sqrt
    whitened_between = (whitened_between + whitened_between.T) * 0.5
    fisher_value, fisher_vector = np.linalg.eigh(whitened_between)
    order = np.argsort(fisher_value)[::-1]
    fisher_value = np.maximum(fisher_value[order], 0.0)
    fisher_vector = fisher_vector[:, order]
    transform = inverse_sqrt @ fisher_vector[:, :rank]
    mean = centroids.mean(axis=0)
    retained = float(np.sum(fisher_value[:rank]) / max(np.sum(fisher_value), 1e-12))
    report: dict[str, object] = {
        "within_shrinkage": float(within_shrinkage),
        "rank": int(rank),
        "rows": int(len(value)),
        "dimension": int(dimension),
        "identities": int(len(centroids)),
        "replicated_identities": replicated,
        "eligible_multi_identity_formulas": eligible_formulas,
        "identities_in_between_covariance": between_identities,
        "between_mode": between_mode,
        "within_condition_number": float(regularized[-1] / regularized[0]),
        "largest_generalized_eigenvalue": float(fisher_value[0]),
        "smallest_retained_generalized_eigenvalue": float(fisher_value[rank - 1]),
        "retained_generalized_eigenvalue_fraction": retained,
        "generalized_eigenvalues": fisher_value.astype(float).tolist(),
    }
    return mean.astype(np.float32), transform.astype(np.float32), report
