"""Low-dimensional spectrum-only gates for shared ChemAware kernels."""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression


SCALAR_NAMES = (
    "precursor_mz", "peak_count", "intensity_entropy", "base_peak_fraction",
    "intensity_weighted_mz_mean", "intensity_weighted_mz_std",
    "first_five_intensity_fraction", "minimum_fragment_mz", "maximum_fragment_mz",
)


def spectrum_scalar_vector(cache: object, row: int) -> np.ndarray:
    position = cache.row_position[int(row)]
    valid = np.asarray(cache.valid[position], dtype=bool)
    mz = np.asarray(cache.mz[position][valid], dtype=np.float64)
    intensity = np.maximum(np.asarray(cache.intensity[position][valid], dtype=np.float64), 0.0)
    total = max(float(intensity.sum()), 1e-12)
    probability = intensity / total
    mean_mz = float(np.sum(probability * mz)) if len(mz) else 0.0
    return np.asarray((
        float(cache.precursor[position]), float(len(mz)),
        float(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))),
        float(np.max(probability)) if len(probability) else 0.0,
        mean_mz,
        float(np.sqrt(np.sum(probability * (mz - mean_mz) ** 2))) if len(mz) else 0.0,
        float(np.sum(probability[: min(5, len(probability))])),
        float(np.min(mz)) if len(mz) else 0.0,
        float(np.max(mz)) if len(mz) else 0.0,
    ), dtype=np.float32)


def fit_logistic_gate(
    feature: np.ndarray,
    label: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray | float], dict[str, object]]:
    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.int8)
    if feature.ndim != 2 or label.shape != (len(feature),) or len(np.unique(label)) != 2:
        raise ValueError("logistic gate needs a matrix and two outcome classes")
    mean = feature.mean(axis=0)
    scale = feature.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (feature - mean) / scale
    model = LogisticRegression(
        C=0.1, class_weight="balanced", solver="liblinear",
        max_iter=5000, random_state=seed,
    )
    model.fit(standardized, label)
    payload: dict[str, np.ndarray | float] = {
        "mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
        "coefficient": np.asarray(model.coef_[0], dtype=np.float32),
        "intercept": float(model.intercept_[0]),
    }
    return payload, {
        "rows": int(len(feature)), "positive": int(np.sum(label == 1)),
        "negative": int(np.sum(label == 0)), "features": int(feature.shape[1]),
        "coefficient": np.asarray(model.coef_[0], dtype=float).tolist(),
        "intercept": float(model.intercept_[0]),
    }


def stable_group_binary_split(group: np.ndarray, seed: int) -> np.ndarray:
    group = np.asarray(group).astype(str)
    return np.asarray([
        int.from_bytes(
            hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little",
        ) % 2
        for value in group
    ], dtype=np.int8)


def fit_cross_split_consensus_gate(
    feature: np.ndarray,
    label: np.ndarray,
    group: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray | float], dict[str, object]]:
    """Retain logistic coefficients whose signs reproduce across formula halves."""
    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.int8)
    group = np.asarray(group).astype(str)
    if (
        feature.ndim != 2
        or label.shape != (len(feature),)
        or group.shape != (len(feature),)
        or len(np.unique(label)) != 2
    ):
        raise ValueError("cross-split gate inputs are invalid")
    split = stable_group_binary_split(group, seed)
    mean = feature.mean(axis=0)
    scale = np.where(feature.std(axis=0) > 1e-8, feature.std(axis=0), 1.0)
    standardized = (feature - mean) / scale
    coefficients = []
    intercepts = []
    split_reports = []
    for value in (0, 1):
        index = np.flatnonzero(split == value)
        if len(np.unique(label[index])) != 2:
            raise ValueError("one formula half lacks a correction or harm class")
        model = LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=seed + value,
        )
        model.fit(standardized[index], label[index])
        coefficients.append(np.asarray(model.coef_[0], dtype=np.float64))
        intercepts.append(float(model.intercept_[0]))
        split_reports.append({
            "split": value, "rows": int(len(index)),
            "positive": int(np.sum(label[index] == 1)),
            "negative": int(np.sum(label[index] == 0)),
            "coefficient": np.asarray(model.coef_[0], dtype=float).tolist(),
            "intercept": float(model.intercept_[0]),
        })
    first, second = coefficients
    agrees = (np.sign(first) == np.sign(second)) & (np.abs(first) > 1e-12) & (np.abs(second) > 1e-12)
    consensus = np.zeros_like(first)
    consensus[agrees] = np.sign(first[agrees]) * np.sqrt(np.abs(first[agrees] * second[agrees]))
    payload: dict[str, np.ndarray | float] = {
        "mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
        "coefficient": consensus.astype(np.float32),
        "intercept": float(np.mean(intercepts)),
    }
    return payload, {
        "rows": int(len(feature)), "positive": int(np.sum(label == 1)),
        "negative": int(np.sum(label == 0)), "features": int(feature.shape[1]),
        "formula_disjoint_halves": True,
        "split_reports": split_reports,
        "reproduced_features": int(np.sum(agrees)),
        "consensus_coefficient": consensus.astype(float).tolist(),
        "intercept": float(np.mean(intercepts)),
    }


def gate_probability(
    feature: np.ndarray,
    model: dict[str, np.ndarray | float],
) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float64)
    standardized = (
        feature - np.asarray(model["mean"], dtype=np.float64)
    ) / np.asarray(model["scale"], dtype=np.float64)
    logit = standardized @ np.asarray(model["coefficient"], dtype=np.float64) + float(model["intercept"])
    return (1.0 / (1.0 + np.exp(np.clip(-logit, -40.0, 40.0)))).astype(np.float32)


def gate_amplitude(probability: np.ndarray, floor: float, power: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if np.any((probability < 0) | (probability > 1)) or not 0 <= floor <= 1 or power <= 0:
        raise ValueError("invalid gate probability, floor, or power")
    strength = floor + (1.0 - floor) * probability ** float(power)
    return np.sqrt(strength).astype(np.float32)
