"""Spectrum-pair scores used by the Noise/P2b GNPS article benchmark.

The module deliberately separates two contracts:

* literature-style classical scores (greedy cosine, precursor-shift modified
  cosine, weighted spectral entropy), and
* the historical P2b features, which must remain bit-for-bit tied to the
  frozen P2b implementation rather than being silently replaced by a newer
  definition.
"""
from __future__ import annotations

import math

import numpy as np

from audit_large_observability_residual import symmetric_features
from noise_corrected_fullgraph_evaluation import GraphScores


P2B_WEIGHTS = np.asarray([0.10, 0.00, 0.10, 0.80], dtype=np.float64)

try:  # The formal sbatch installs the pinned official wheel node-locally.
    import ms_entropy as _ms_entropy
except ImportError:  # A transparent fallback keeps dependency-free unit tests possible.
    _ms_entropy = None


def prepare_spectrum(spectrum: np.ndarray, n_peaks: int = 100) -> np.ndarray:
    """Return finite positive peaks, intensity-truncated and m/z-sorted."""
    values = np.asarray(spectrum, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("spectrum must have shape (2, peaks)")
    keep = (
        np.isfinite(values[0]) & np.isfinite(values[1])
        & (values[0] > 0) & (values[1] > 0)
    )
    values = values[:, keep]
    if values.shape[1] > n_peaks:
        selected = np.argpartition(values[1], -n_peaks)[-n_peaks:]
        values = values[:, selected]
    return values[:, np.argsort(values[0], kind="stable")].astype(np.float32)


def _normalised_intensity(spectrum: np.ndarray, power: float) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=np.float64)
    intensity = np.asarray(spectrum[1], dtype=np.float64) ** float(power)
    norm = float(np.linalg.norm(intensity))
    if norm <= 0:
        return mz, np.zeros_like(intensity)
    return mz, intensity / norm


def _greedy_score(
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
    tolerance: float,
    precursor_shift: float | None,
    intensity_power: float,
) -> float:
    mz_a, intensity_a = _normalised_intensity(spectrum_a, intensity_power)
    mz_b, intensity_b = _normalised_intensity(spectrum_b, intensity_power)
    candidates: dict[tuple[int, int], float] = {}
    shifts = (0.0,) if precursor_shift is None else (0.0, float(precursor_shift))
    for i, value in enumerate(mz_a):
        for shift in shifts:
            target = value - shift
            left = int(np.searchsorted(mz_b, target - tolerance, side="left"))
            right = int(np.searchsorted(mz_b, target + tolerance, side="right"))
            for j in range(left, right):
                key = (i, j)
                candidates[key] = max(
                    candidates.get(key, 0.0),
                    float(intensity_a[i] * intensity_b[j]),
                )
    used_a: set[int] = set()
    used_b: set[int] = set()
    score = 0.0
    for (i, j), contribution in sorted(
        candidates.items(), key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        score += contribution
    return float(np.clip(score, 0.0, 1.0))


def cosine_greedy(
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
    tolerance: float = 0.02,
    intensity_power: float = 1.0,
) -> float:
    return _greedy_score(
        spectrum_a, spectrum_b, tolerance,
        precursor_shift=None, intensity_power=intensity_power,
    )


def modified_cosine(
    spectrum_a: np.ndarray,
    precursor_a: float,
    spectrum_b: np.ndarray,
    precursor_b: float,
    tolerance: float = 0.02,
    intensity_power: float = 1.0,
) -> float:
    """Greedy cosine allowing the matchms-style precursor mass shift."""
    return _greedy_score(
        spectrum_a, spectrum_b, tolerance,
        precursor_shift=float(precursor_a) - float(precursor_b),
        intensity_power=intensity_power,
    )


def _entropy_weight(intensity: np.ndarray) -> np.ndarray:
    values = np.asarray(intensity, dtype=np.float64)
    values = values / max(float(values.sum()), 1e-15)
    entropy = -float(np.sum(values[values > 0] * np.log(values[values > 0])))
    if entropy < 3.0:
        values = values ** (0.25 + 0.25 * entropy)
        values /= max(float(values.sum()), 1e-15)
    return values


def weighted_entropy_similarity(
    spectrum_a: np.ndarray,
    spectrum_b: np.ndarray,
    tolerance: float = 0.02,
) -> float:
    """Weighted spectral-entropy similarity (Jensen-Shannon form)."""
    if _ms_entropy is not None:
        left = np.asarray(spectrum_a.T, dtype=np.float32, order="C")
        right = np.asarray(spectrum_b.T, dtype=np.float32, order="C")
        return float(_ms_entropy.calculate_entropy_similarity(
            left, right,
            ms2_tolerance_in_da=float(tolerance),
            clean_spectra=True,
            noise_threshold=0.01,
            max_peak_num=100,
        ))
    mz_a = np.asarray(spectrum_a[0], dtype=np.float64)
    mz_b = np.asarray(spectrum_b[0], dtype=np.float64)
    intensity_a = _entropy_weight(spectrum_a[1])
    intensity_b = _entropy_weight(spectrum_b[1])
    candidates: list[tuple[float, int, int]] = []
    for i, value in enumerate(mz_a):
        left = int(np.searchsorted(mz_b, value - tolerance, side="left"))
        right = int(np.searchsorted(mz_b, value + tolerance, side="right"))
        candidates.extend((abs(float(value - mz_b[j])), i, j) for j in range(left, right))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((i, j))
    pa: list[float] = []
    pb: list[float] = []
    for i, j in matches:
        pa.append(float(intensity_a[i]))
        pb.append(float(intensity_b[j]))
    for i in set(range(len(intensity_a))) - used_a:
        pa.append(float(intensity_a[i])); pb.append(0.0)
    for j in set(range(len(intensity_b))) - used_b:
        pa.append(0.0); pb.append(float(intensity_b[j]))
    left = np.asarray(pa, dtype=np.float64)
    right = np.asarray(pb, dtype=np.float64)
    mean = 0.5 * (left + right)
    nz_left, nz_right = left > 0, right > 0
    js = 0.5 * np.sum(left[nz_left] * np.log(left[nz_left] / mean[nz_left]))
    js += 0.5 * np.sum(right[nz_right] * np.log(right[nz_right] / mean[nz_right]))
    return float(np.clip(1.0 - js / math.log(2.0), 0.0, 1.0))


def weighted_entropy_backend() -> str:
    if _ms_entropy is None:
        return "dependency_free_formula_fallback"
    return f"ms_entropy_{_ms_entropy.__version__}"


def frozen_p2b_pair_features(
    spectrum_a: np.ndarray,
    precursor_a: float,
    spectrum_b: np.ndarray,
    precursor_b: float,
    tolerance: float = 0.02,
) -> tuple[float, float, float]:
    """Return the exact three non-DreaMS features consumed by frozen P2b."""
    values = symmetric_features(
        spectrum_a, precursor_a, spectrum_b, precursor_b, tolerance,
    )
    return (
        float(values["sqrt_cosine"]),
        float(values["entropy_similarity"]),
        float(values["neutral_loss_sqrt_cosine"]),
    )


def _unique_top(values: np.ndarray, eps: float = 1e-12) -> int | None:
    maximum = float(np.max(values))
    winners = np.flatnonzero(values >= maximum - eps)
    return int(winners[0]) if len(winners) == 1 else None


def apply_frozen_p2b(
    graph,
    dreams_pair: np.ndarray,
    sqrt_pair: np.ndarray,
    entropy_pair: np.ndarray,
    neutral_loss_pair: np.ndarray,
) -> GraphScores:
    """Apply the registered P2b weights, normalization *and* its support gate.

    The frozen ``g8r_p2_rank_fusion_core.normalize_pair_features("absolute")``
    contract maps the DreaMS channel through ``(x + 1) / 2`` onto [0, 1] and
    clips the three classical channels before weighting, and tie detection
    uses a 1e-12 epsilon window.  Both are reproduced exactly; without the
    channel transform the fused dynamic range drifts and this stops being the
    frozen P2b.
    """
    dreams_channel = np.clip(
        (np.asarray(dreams_pair, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0,
    )
    channels = np.stack([
        dreams_channel,
        np.clip(np.asarray(sqrt_pair, dtype=np.float64), 0.0, 1.0),
        np.clip(np.asarray(entropy_pair, dtype=np.float64), 0.0, 1.0),
        np.clip(np.asarray(neutral_loss_pair, dtype=np.float64), 0.0, 1.0),
    ], axis=1)
    fused_pair = channels @ P2B_WEIGHTS
    dreams_molecule = np.maximum.reduceat(np.asarray(dreams_pair), graph.molecule_ptr[:-1])
    raw_molecule = np.stack([
        np.maximum.reduceat(channels[:, column], graph.molecule_ptr[:-1])
        for column in (1, 2, 3)
    ], axis=1)
    fused_molecule = np.maximum.reduceat(fused_pair, graph.molecule_ptr[:-1])
    output = np.asarray(dreams_pair, dtype=np.float32).copy()
    for query, (left, right) in enumerate(zip(graph.query_ptr[:-1], graph.query_ptr[1:])):
        base_block = dreams_molecule[left:right]
        fused_block = fused_molecule[left:right]
        fused_winner = _unique_top(fused_block)
        if fused_winner is None:
            continue
        support = sum(
            _unique_top(raw_molecule[left:right, column]) == fused_winner
            for column in range(3)
        )
        baseline_winner = _unique_top(base_block)
        if baseline_winner is None:
            advantage = math.inf
        elif fused_winner == baseline_winner:
            advantage = 0.0
        else:
            advantage = float(fused_block[fused_winner] - fused_block[baseline_winner])
        if support >= 1 and advantage >= -1e-12:
            pair_left = int(graph.molecule_ptr[left])
            pair_right = int(graph.molecule_ptr[right])
            output[pair_left:pair_right] = fused_pair[pair_left:pair_right]
    molecule = np.maximum.reduceat(output, graph.molecule_ptr[:-1])
    return GraphScores(pair=output, molecule=molecule)
