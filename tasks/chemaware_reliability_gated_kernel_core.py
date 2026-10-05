"""Core algebra for a monotone reliability-gated shared chemical kernel."""
from __future__ import annotations

import numpy as np


def empirical_percentile(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Map scalar reliability to a frozen empirical CDF in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    if values.ndim != 1 or calibration.ndim != 1 or len(calibration) == 0:
        raise ValueError("values and nonempty calibration must be one-dimensional")
    if not np.isfinite(values).all() or not np.isfinite(calibration).all():
        raise ValueError("reliability contains non-finite values")
    ordered = np.sort(calibration)
    return (
        np.searchsorted(ordered, values, side="right").astype(np.float64)
        / float(len(ordered))
    ).astype(np.float32)


def monotone_gate(
    reliability: np.ndarray,
    calibration: np.ndarray,
    *,
    floor: float,
    power: float,
) -> np.ndarray:
    """A spectrum-only nonnegative gate, calibrated without evaluation rows."""
    if not 0.0 <= floor <= 1.0 or power <= 0.0:
        raise ValueError("gate floor/power out of range")
    percentile = empirical_percentile(reliability, calibration)
    return (floor + (1.0 - floor) * percentile ** power).astype(np.float32)


def centered_reliability(
    center: np.ndarray,
    backgrounds: tuple[np.ndarray, ...] | list[np.ndarray],
) -> np.ndarray:
    """Norm of a proposed center after subtraction of matched local centers."""
    center = np.asarray(center, dtype=np.float32)
    background = [np.asarray(value, dtype=np.float32) for value in backgrounds]
    if center.ndim != 2 or not background or any(value.shape != center.shape for value in background):
        raise ValueError("center/background feature shapes disagree")
    residual = center - np.mean(np.stack(background, axis=0), axis=0)
    return np.linalg.norm(residual, axis=1).astype(np.float32)


def gated_pair_score(
    official_pair: np.ndarray,
    mass_pair: np.ndarray,
    chemical_pair: np.ndarray,
    query_gate: np.ndarray,
    reference_gate: np.ndarray,
    *,
    mass_beta: float,
    chemical_beta: float,
) -> np.ndarray:
    """Pair score induced by one ordinary shared feature map."""
    arrays = [official_pair, mass_pair, chemical_pair, query_gate, reference_gate]
    arrays = [np.asarray(value, dtype=np.float32) for value in arrays]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("pair arrays disagree")
    if mass_beta < 0.0 or chemical_beta < 0.0:
        raise ValueError("kernel weights must be nonnegative")
    return (
        arrays[0]
        + float(mass_beta) * arrays[1]
        + float(chemical_beta) * arrays[3] * arrays[4] * arrays[2]
    )


def reliability_gated_embedding(
    official: np.ndarray,
    mass: np.ndarray,
    chemical: np.ndarray,
    gate: np.ndarray,
    *,
    mass_beta: float,
    chemical_beta: float,
) -> np.ndarray:
    """Materialize the PSD primal map used by :func:`gated_pair_score`."""
    official = np.asarray(official, dtype=np.float32)
    mass = np.asarray(mass, dtype=np.float32)
    chemical = np.asarray(chemical, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    if official.ndim != 2 or mass.ndim != 2 or chemical.ndim != 2:
        raise ValueError("embedding blocks must be matrices")
    if not (len(official) == len(mass) == len(chemical) == len(gate)):
        raise ValueError("embedding blocks disagree on rows")
    if gate.ndim != 1 or np.any(gate < 0.0):
        raise ValueError("gate must be a nonnegative vector")
    if mass_beta < 0.0 or chemical_beta < 0.0:
        raise ValueError("kernel weights must be nonnegative")
    return np.concatenate((
        official,
        np.sqrt(float(mass_beta)) * mass,
        np.sqrt(float(chemical_beta)) * gate[:, None] * chemical,
    ), axis=1)
