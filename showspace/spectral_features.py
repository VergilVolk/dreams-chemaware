"""Auditable raw MS/MS pair features used by the frozen P2b scorer.

This is a deployment-only extraction of the feature definitions used during
P2b development.  It deliberately contains no learned weights or labels.
"""

from __future__ import annotations

import numpy as np


def clean_peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted positive finite m/z and intensity vectors.

    The accepted layouts are ``(n, 2)`` for uploaded spectra and ``(2, n)``
    for the MassSpecGym HDF5 reference library.
    """
    values = np.asarray(spectrum, dtype=float)
    if values.ndim != 2:
        raise ValueError("spectrum must be a two-dimensional peak array")
    if values.shape[1] == 2:
        mz, intensity = values[:, 0], values[:, 1]
    elif values.shape[0] == 2:
        mz, intensity = values[0], values[1]
    else:
        raise ValueError("spectrum must have shape (n, 2) or (2, n)")
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind="stable")
    return mz[order], intensity[order]


def greedy_matches(a: np.ndarray, b: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for i, value in enumerate(a):
        left = int(np.searchsorted(b, value - tolerance, side="left"))
        right = int(np.searchsorted(b, value + tolerance, side="right"))
        candidates.extend((abs(value - b[j]), i, j) for j in range(left, right))
    used_a: set[int] = set()
    used_b: set[int] = set()
    output: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            output.append((i, j))
    return output


def matched_metrics(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    if len(mz_a) == 0 or len(mz_b) == 0:
        return {"sqrt_cosine": 0.0, "entropy_similarity": 0.0}
    matches = greedy_matches(mz_a, mz_b, tolerance)
    ia = int_a / max(float(int_a.sum()), 1e-12)
    ib = int_b / max(float(int_b.sum()), 1e-12)
    sqrt_cosine = float(sum(np.sqrt(ia[i]) * np.sqrt(ib[j]) for i, j in matches))

    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    pa: list[float] = []
    pb: list[float] = []
    for i, j in matches:
        pa.append(float(ia[i])); pb.append(float(ib[j]))
    for i in set(range(len(ia))) - matched_a:
        pa.append(float(ia[i])); pb.append(0.0)
    for j in set(range(len(ib))) - matched_b:
        pa.append(0.0); pb.append(float(ib[j]))
    pa_arr, pb_arr = np.asarray(pa), np.asarray(pb)
    mean = 0.5 * (pa_arr + pb_arr)
    nonzero_a, nonzero_b = pa_arr > 0, pb_arr > 0
    js = 0.5 * np.sum(pa_arr[nonzero_a] * np.log(pa_arr[nonzero_a] / mean[nonzero_a]))
    js += 0.5 * np.sum(pb_arr[nonzero_b] * np.log(pb_arr[nonzero_b] / mean[nonzero_b]))
    entropy = float(np.clip(1.0 - js / np.log(2.0), 0.0, 1.0))
    return {"sqrt_cosine": sqrt_cosine, "entropy_similarity": entropy}


def p2b_pair_features(
    query: np.ndarray,
    query_precursor: float,
    reference: np.ndarray,
    reference_precursor: float,
    dreams_similarity: float,
    tolerance: float = 0.02,
) -> np.ndarray:
    """Return the four frozen P2b features in their registered order."""
    mz_a, int_a = clean_peaks(query)
    mz_b, int_b = clean_peaks(reference)
    fragment = matched_metrics(mz_a, int_a, mz_b, int_b, tolerance)
    loss_a = query_precursor - mz_a
    loss_b = reference_precursor - mz_b
    keep_a, keep_b = loss_a > 0, loss_b > 0
    order_a, order_b = np.argsort(loss_a[keep_a]), np.argsort(loss_b[keep_b])
    neutral = matched_metrics(
        loss_a[keep_a][order_a], int_a[keep_a][order_a],
        loss_b[keep_b][order_b], int_b[keep_b][order_b],
        tolerance,
    )
    return np.asarray([
        float(dreams_similarity),
        fragment["sqrt_cosine"],
        fragment["entropy_similarity"],
        neutral["sqrt_cosine"],
    ], dtype=np.float64)
