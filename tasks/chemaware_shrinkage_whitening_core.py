"""Shrinkage whitening for shared ChemAware rule-response features."""
from __future__ import annotations

import numpy as np


def fit_shrinkage_whitener(
    feature: np.ndarray,
    shrinkage: float,
    *,
    eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a symmetric inverse-square-root covariance transform."""
    value = np.asarray(feature, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("whitener needs a finite matrix with at least two rows")
    if not 0.0 <= shrinkage <= 1.0 or eigenvalue_floor <= 0.0:
        raise ValueError("invalid shrinkage or eigenvalue floor")
    mean = value.mean(axis=0)
    centered = value - mean
    covariance = centered.T @ centered / float(len(centered) - 1)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    isotropic = float(np.trace(covariance) / covariance.shape[0])
    shrunk = (1.0 - float(shrinkage)) * eigenvalue + float(shrinkage) * isotropic
    floor = max(float(eigenvalue_floor) * max(isotropic, 1e-12), 1e-12)
    shrunk = np.maximum(shrunk, floor)
    transform = (eigenvector * (1.0 / np.sqrt(shrunk))[None, :]) @ eigenvector.T
    report = {
        "shrinkage": float(shrinkage),
        "rows": int(len(value)),
        "dimension": int(value.shape[1]),
        "raw_min_eigenvalue": float(eigenvalue[0]),
        "raw_max_eigenvalue": float(eigenvalue[-1]),
        "shrunk_min_eigenvalue": float(shrunk[0]),
        "shrunk_max_eigenvalue": float(shrunk[-1]),
        "shrunk_condition_number": float(shrunk[-1] / shrunk[0]),
    }
    return mean.astype(np.float32), transform.astype(np.float32), report


def apply_whitener(
    feature: np.ndarray,
    mean: np.ndarray,
    transform: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Apply one frozen spectrum-local linear feature map."""
    value = np.asarray(feature, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    if value.ndim not in (1, 2) or mean.shape != (value.shape[-1],):
        raise ValueError("feature and mean disagree")
    if transform.ndim != 2 or transform.shape[0] != value.shape[-1]:
        raise ValueError("feature and transform disagree")
    output = (value - mean) @ transform
    if normalize:
        norm = np.linalg.norm(output, axis=-1, keepdims=True)
        output = np.divide(output, norm, out=np.zeros_like(output), where=norm > 1e-12)
    return output.astype(np.float32)


def fit_balanced_within_whitener(
    feature: np.ndarray,
    identity: np.ndarray,
    shrinkage: float,
    *,
    eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Whiten identity-balanced within-molecule acquisition variation."""
    value = np.asarray(feature, dtype=np.float64)
    labels = np.asarray(identity).astype(str)
    if value.ndim != 2 or labels.shape != (len(value),) or not np.isfinite(value).all():
        raise ValueError("within whitener inputs are not aligned finite arrays")
    if not 0.0 <= shrinkage <= 1.0 or eigenvalue_floor <= 0.0:
        raise ValueError("invalid shrinkage or eigenvalue floor")
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    eligible = np.flatnonzero(counts >= 2)
    if len(eligible) < 2:
        raise ValueError("within whitener needs at least two replicated identities")
    covariance = np.zeros((value.shape[1], value.shape[1]), dtype=np.float64)
    centroids = np.zeros((len(unique), value.shape[1]), dtype=np.float64)
    for group in range(len(unique)):
        local = value[inverse == group]
        centroids[group] = local.mean(axis=0)
        if len(local) >= 2:
            residual = local - centroids[group]
            covariance += residual.T @ residual / float(len(local) - 1)
    covariance /= float(len(eligible))
    mean = centroids.mean(axis=0)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    isotropic = float(np.trace(covariance) / covariance.shape[0])
    shrunk = (1.0 - float(shrinkage)) * eigenvalue + float(shrinkage) * isotropic
    floor = max(float(eigenvalue_floor) * max(isotropic, 1e-12), 1e-12)
    shrunk = np.maximum(shrunk, floor)
    transform = (eigenvector * (1.0 / np.sqrt(shrunk))[None, :]) @ eigenvector.T
    report = {
        "shrinkage": float(shrinkage),
        "rows": int(len(value)),
        "identities": int(len(unique)),
        "replicated_identities": int(len(eligible)),
        "rows_in_replicated_identities": int(np.sum(counts[eligible])),
        "median_spectra_per_identity": float(np.median(counts)),
        "max_spectra_per_identity": int(np.max(counts)),
        "dimension": int(value.shape[1]),
        "raw_min_eigenvalue": float(eigenvalue[0]),
        "raw_max_eigenvalue": float(eigenvalue[-1]),
        "shrunk_min_eigenvalue": float(shrunk[0]),
        "shrunk_max_eigenvalue": float(shrunk[-1]),
        "shrunk_condition_number": float(shrunk[-1] / shrunk[0]),
    }
    return mean.astype(np.float32), transform.astype(np.float32), report
