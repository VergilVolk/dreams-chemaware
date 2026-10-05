"""Replicate-calibrated heteroscedastic reliability for ChemAware kernels."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def leave_one_out_identity_consistency(
    feature: np.ndarray,
    identity: np.ndarray,
    *,
    min_replicates: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Cosine to the leave-one-out centroid for every replicated identity row."""
    feature = np.asarray(feature, dtype=np.float64)
    identity = np.asarray(identity).astype(str)
    if (
        feature.ndim != 2 or identity.shape != (len(feature),)
        or not np.isfinite(feature).all() or min_replicates < 2
    ):
        raise ValueError("replicate consistency inputs are invalid")
    unique, inverse = np.unique(identity, return_inverse=True)
    counts = np.bincount(inverse)
    eligible = counts[inverse] >= int(min_replicates)
    index = np.flatnonzero(eligible)
    target_all = np.full(len(feature), np.nan, dtype=np.float64)
    for group in np.flatnonzero(counts >= int(min_replicates)):
        local_index = np.flatnonzero(inverse == group)
        local = feature[local_index]
        total = local.sum(axis=0)
        other = (total[None, :] - local) / float(len(local) - 1)
        local_norm = np.linalg.norm(local, axis=1)
        other_norm = np.linalg.norm(other, axis=1)
        cosine = np.sum(local * other, axis=1) / np.maximum(local_norm * other_norm, 1e-12)
        target_all[local_index] = cosine
    target = target_all[index]
    if not np.isfinite(target).all():
        raise RuntimeError("replicate consistency failed to align targets to rows")
    return index.astype(np.int64), target.astype(np.float32), {
        "rows": int(len(feature)), "identities": int(len(unique)),
        "minimum_replicates": int(min_replicates),
        "replicated_identities": int(np.sum(counts >= int(min_replicates))),
        "replicated_rows": int(len(index)),
        "target_quantiles": np.quantile(target, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)).astype(float).tolist(),
    }


def identity_balanced_sample_weight(identity: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Give every identity equal total regression weight, regardless of multiplicity."""
    identity = np.asarray(identity).astype(str)
    if identity.ndim != 1 or len(identity) == 0:
        raise ValueError("identity weights require a non-empty vector")
    unique, inverse = np.unique(identity, return_inverse=True)
    counts = np.bincount(inverse)
    weight = 1.0 / counts[inverse]
    weight /= weight.mean()
    identity_total = np.bincount(inverse, weights=weight)
    return weight.astype(np.float64), {
        "rows": int(len(identity)), "identities": int(len(unique)),
        "minimum_multiplicity": int(counts.min()),
        "median_multiplicity": float(np.median(counts)),
        "maximum_multiplicity": int(counts.max()),
        "identity_total_weight_range": [float(identity_total.min()), float(identity_total.max())],
        "effective_sample_size": float(weight.sum() ** 2 / np.sum(weight ** 2)),
    }


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    weight = weight / weight.sum()
    left_centered = left - np.sum(weight * left)
    right_centered = right - np.sum(weight * right)
    numerator = np.sum(weight * left_centered * right_centered)
    denominator = np.sqrt(
        np.sum(weight * left_centered ** 2) * np.sum(weight * right_centered ** 2)
    )
    return float(numerator / max(float(denominator), 1e-12))


def fit_reliability_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 10.0,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray | float], dict[str, object]]:
    feature = np.asarray(feature, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if sample_weight is None:
        weight = np.ones(len(feature), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
    if (
        feature.ndim != 2 or target.shape != (len(feature),)
        or weight.shape != (len(feature),) or not np.isfinite(weight).all()
        or np.any(weight <= 0) or not np.isfinite(feature).all()
        or not np.isfinite(target).all() or alpha <= 0
    ):
        raise ValueError("reliability ridge inputs are invalid")
    normalized_weight = weight / weight.sum()
    mean = np.sum(normalized_weight[:, None] * feature, axis=0)
    variance = np.sum(normalized_weight[:, None] * (feature - mean) ** 2, axis=0)
    scale = np.where(np.sqrt(variance) > 1e-8, np.sqrt(variance), 1.0)
    standardized = (feature - mean) / scale
    model = Ridge(alpha=float(alpha))
    model.fit(standardized, target, sample_weight=weight)
    prediction = model.predict(standardized)
    correlation = float(np.corrcoef(prediction, target)[0, 1])
    weighted_correlation = _weighted_correlation(prediction, target, weight)
    payload: dict[str, np.ndarray | float] = {
        "mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
        "coefficient": np.asarray(model.coef_, dtype=np.float32),
        "intercept": float(model.intercept_),
    }
    return payload, {
        "rows": int(len(feature)), "features": int(feature.shape[1]),
        "alpha": float(alpha), "training_correlation": correlation,
        "weighted_training_correlation": weighted_correlation,
        "weighted_standardization": True,
        "effective_sample_size": float(weight.sum() ** 2 / np.sum(weight ** 2)),
        "coefficient": np.asarray(model.coef_, dtype=float).tolist(),
        "target_mean": float(target.mean()), "target_std": float(target.std()),
        "prediction_std": float(prediction.std()),
    }


def predict_reliability(
    feature: np.ndarray,
    model: dict[str, np.ndarray | float],
) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float64)
    standardized = (
        feature - np.asarray(model["mean"], dtype=np.float64)
    ) / np.asarray(model["scale"], dtype=np.float64)
    return (
        standardized @ np.asarray(model["coefficient"], dtype=np.float64)
        + float(model["intercept"])
    ).astype(np.float32)


def empirical_gate_amplitude(
    prediction: np.ndarray,
    calibration: np.ndarray,
    floor: float,
    power: float,
    *,
    reversed_order: bool = False,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    calibration = np.sort(np.asarray(calibration, dtype=np.float64))
    if (
        prediction.ndim != 1 or calibration.ndim != 1 or len(calibration) == 0
        or not 0 <= floor <= 1 or power <= 0
    ):
        raise ValueError("invalid empirical reliability gate")
    percentile = np.searchsorted(calibration, prediction, side="right") / float(len(calibration))
    if reversed_order:
        percentile = 1.0 - percentile
    strength = floor + (1.0 - floor) * percentile ** float(power)
    return np.sqrt(strength).astype(np.float32)
