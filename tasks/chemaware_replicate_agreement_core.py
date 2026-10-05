"""Identity-balanced replicate-agreement maps for ChemAware features."""
from __future__ import annotations

import numpy as np


def fit_replicate_agreement_basis(
    feature: np.ndarray,
    identity: np.ndarray,
    *,
    min_replicates: int = 3,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Estimate chemically repeatable directions from independent spectra.

    For each identity, the ordered-pair cross moment is computed without
    materializing O(n^2) pairs.  Identities, rather than spectra, are averaged
    equally so highly replicated compounds cannot dominate the basis.
    """
    value = np.asarray(feature, dtype=np.float64)
    labels = np.asarray(identity).astype(str)
    if (
        value.ndim != 2 or labels.shape != (len(value),)
        or not np.isfinite(value).all() or min_replicates < 2
    ):
        raise ValueError("replicate-agreement inputs are invalid")
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    groups = np.flatnonzero(counts >= int(min_replicates))
    if len(groups) < 2:
        raise ValueError("replicate-agreement fit needs at least two eligible identities")
    centroids = np.stack([value[inverse == group].mean(axis=0) for group in groups])
    grand_mean = centroids.mean(axis=0)
    cross = np.zeros((value.shape[1], value.shape[1]), dtype=np.float64)
    marginal = np.zeros_like(cross)
    for group in groups:
        local = value[inverse == group] - grand_mean
        total = local.sum(axis=0)
        cross += (
            np.outer(total, total) - local.T @ local
        ) / float(len(local) * (len(local) - 1))
        marginal += local.T @ local / float(len(local))
    cross /= float(len(groups))
    marginal /= float(len(groups))
    cross = 0.5 * (cross + cross.T)
    eigenvalue, eigenvector = np.linalg.eigh(cross)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = eigenvalue[order]
    eigenvector = eigenvector[:, order]
    positive = np.maximum(eigenvalue, 0.0)
    maximum = float(positive.max())
    normalized = positive / maximum if maximum > 0 else positive
    positive_sum = float(positive.sum())
    return {
        "eigenvector": eigenvector.astype(np.float32),
        "normalized_agreement": normalized.astype(np.float32),
    }, {
        "rows": int(len(value)), "dimension": int(value.shape[1]),
        "identities": int(len(unique)), "minimum_replicates": int(min_replicates),
        "eligible_identities": int(len(groups)),
        "eligible_rows": int(np.sum(counts[groups])),
        "median_eligible_multiplicity": float(np.median(counts[groups])),
        "maximum_eligible_multiplicity": int(np.max(counts[groups])),
        "positive_directions": int(np.sum(eigenvalue > 0)),
        "negative_directions": int(np.sum(eigenvalue < 0)),
        "largest_cross_eigenvalue": float(eigenvalue[0]),
        "smallest_cross_eigenvalue": float(eigenvalue[-1]),
        "positive_agreement_trace": positive_sum,
        "top16_positive_trace_fraction": float(positive[:16].sum() / max(positive_sum, 1e-12)),
        "top64_positive_trace_fraction": float(positive[:64].sum() / max(positive_sum, 1e-12)),
        "marginal_trace": float(np.trace(marginal)),
        "identity_equal_weighted": True,
        "ordered_self_pairs_excluded": True,
    }


def agreement_transform(
    basis: dict[str, np.ndarray],
    *,
    floor: float,
    power: float,
) -> np.ndarray:
    """Return a PSD feature transform; floor=1 exactly preserves cosine."""
    vector = np.asarray(basis["eigenvector"], dtype=np.float64)
    agreement = np.asarray(basis["normalized_agreement"], dtype=np.float64)
    if (
        vector.ndim != 2 or vector.shape[0] != vector.shape[1]
        or agreement.shape != (vector.shape[1],) or not np.isfinite(vector).all()
        or not np.isfinite(agreement).all() or not 0 <= floor <= 1 or power <= 0
    ):
        raise ValueError("invalid replicate-agreement transform")
    strength = float(floor) + (1.0 - float(floor)) * agreement ** float(power)
    return (vector * np.sqrt(strength)[None, :]).astype(np.float32)


def fit_formula_conditional_contrast_basis(
    feature: np.ndarray,
    identity: np.ndarray,
    formula: np.ndarray,
    *,
    min_replicates: int = 3,
    reverse: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit directions repeatable within identity but not across formula isomers.

    The contrast is formula-balanced: within each eligible molecular formula,
    it subtracts the mean cross moment between different identity centroids
    from the mean distinct-spectrum cross moment within each identity.
    """
    value = np.asarray(feature, dtype=np.float64)
    labels = np.asarray(identity).astype(str)
    formulas = np.asarray(formula).astype(str)
    if (
        value.ndim != 2 or labels.shape != (len(value),)
        or formulas.shape != (len(value),) or not np.isfinite(value).all()
        or min_replicates < 2
    ):
        raise ValueError("formula-conditional contrast inputs are invalid")
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    identity_formula = np.empty(len(unique), dtype=object)
    centroid = np.empty((len(unique), value.shape[1]), dtype=np.float64)
    positive = np.zeros((len(unique), value.shape[1], value.shape[1]), dtype=np.float64)
    for group in range(len(unique)):
        index = np.flatnonzero(inverse == group)
        local_formula = np.unique(formulas[index])
        if len(local_formula) != 1:
            raise ValueError("one identity maps to multiple formulas")
        identity_formula[group] = str(local_formula[0])
        local = value[index]
        centroid[group] = local.mean(axis=0)
        if len(local) >= int(min_replicates):
            total = local.sum(axis=0)
            positive[group] = (
                np.outer(total, total) - local.T @ local
            ) / float(len(local) * (len(local) - 1))
    identity_formula = identity_formula.astype(str)
    contrasts = []
    identities_used = 0
    for name in np.unique(identity_formula):
        group = np.flatnonzero(
            (identity_formula == name) & (counts >= int(min_replicates))
        )
        if len(group) < 2:
            continue
        pos = positive[group].mean(axis=0)
        local_centroid = centroid[group]
        total = local_centroid.sum(axis=0)
        neg = (
            np.outer(total, total) - local_centroid.T @ local_centroid
        ) / float(len(group) * (len(group) - 1))
        contrasts.append(pos - neg)
        identities_used += len(group)
    if len(contrasts) < 2:
        raise ValueError("formula-conditional contrast needs two eligible formulas")
    contrast = np.mean(np.stack(contrasts), axis=0)
    contrast = 0.5 * (contrast + contrast.T)
    if reverse:
        contrast = -contrast
    eigenvalue, eigenvector = np.linalg.eigh(contrast)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = eigenvalue[order]
    eigenvector = eigenvector[:, order]
    positive_eigenvalue = np.maximum(eigenvalue, 0.0)
    maximum = float(positive_eigenvalue.max())
    normalized = positive_eigenvalue / maximum if maximum > 0 else positive_eigenvalue
    positive_sum = float(positive_eigenvalue.sum())
    return {
        "eigenvector": eigenvector.astype(np.float32),
        "normalized_agreement": normalized.astype(np.float32),
    }, {
        "rows": int(len(value)), "dimension": int(value.shape[1]),
        "identities": int(len(unique)), "minimum_replicates": int(min_replicates),
        "eligible_multi_identity_formulas": int(len(contrasts)),
        "eligible_identities": int(identities_used),
        "positive_directions": int(np.sum(eigenvalue > 0)),
        "negative_directions": int(np.sum(eigenvalue < 0)),
        "largest_contrast_eigenvalue": float(eigenvalue[0]),
        "smallest_contrast_eigenvalue": float(eigenvalue[-1]),
        "positive_contrast_trace": positive_sum,
        "top16_positive_trace_fraction": float(
            positive_eigenvalue[:16].sum() / max(positive_sum, 1e-12)
        ),
        "top64_positive_trace_fraction": float(
            positive_eigenvalue[:64].sum() / max(positive_sum, 1e-12)
        ),
        "formula_equal_weighted": True,
        "identity_equal_weighted_within_formula": True,
        "ordered_self_pairs_excluded": True,
        "contrast_reversed": bool(reverse),
    }


def apply_agreement_transform(feature: np.ndarray, transform: np.ndarray) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    if value.ndim not in (1, 2) or transform.shape != (value.shape[-1], value.shape[-1]):
        raise ValueError("agreement feature and transform disagree")
    output = value @ transform
    norm = np.linalg.norm(output, axis=-1, keepdims=True)
    output = np.divide(output, norm, out=np.zeros_like(output), where=norm > 1e-12)
    return output.astype(np.float32)
