"""MetDNA/MetDNA3-compatible dot-product spectral similarity.

The implementation follows the public MetDNA ``IdentifyFeature.R`` defaults:
intensity exponent 1, m/z exponent 0, and the 400 m/z low-mass tolerance
floor. MetDNA3 v3.1.1 uses 25 ppm and the reverse score with the smaller-
precursor spectrum as reference.
"""
from __future__ import annotations

import numpy as np


def spectrum_from_dreams_tensor(tensor: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return precursor, fragment m/z and intensity from a DreaMS tensor."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 2 or tensor.shape[1] != 2 or tensor.shape[0] < 2:
        raise ValueError("expected a DreaMS [tokens, 2] tensor")
    precursor = float(tensor[0, 0])
    fragments = tensor[1:]
    valid = (
        np.isfinite(fragments[:, 0])
        & np.isfinite(fragments[:, 1])
        & (fragments[:, 0] > 0)
        & (fragments[:, 1] > 0)
    )
    fragments = fragments[valid]
    order = np.argsort(fragments[:, 0], kind="stable")
    return precursor, fragments[order, 0], fragments[order, 1]


def _difference_ppm(mz: np.ndarray, mz_floor: float) -> np.ndarray:
    upper = mz[1:]
    difference = np.diff(mz) / upper * 1e6
    low = upper <= mz_floor
    difference[low] *= upper[low] / mz_floor
    return difference


def _align_to_union(
    source_mz: np.ndarray,
    source_intensity: np.ndarray,
    union_mz: np.ndarray,
) -> np.ndarray:
    """Mirror R ``match`` semantics: the first equal union position is used."""
    aligned = np.zeros(len(union_mz), dtype=float)
    positions: dict[float, int] = {}
    for position, value in enumerate(union_mz):
        positions.setdefault(float(value), position)
    for mz, intensity in zip(source_mz, source_intensity, strict=True):
        aligned[positions[float(mz)]] = float(intensity)
    return aligned


def _collapse_nearby(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    tolerance_ppm: float,
    mz_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Port MetDNA ``MatchSpec`` including its high-to-low merge order."""
    mz = np.asarray(mz, dtype=float).copy()
    intensity = np.asarray(intensity, dtype=float).copy()
    while len(mz) > 1:
        eligible = np.flatnonzero(_difference_ppm(mz, mz_floor) < tolerance_ppm)
        if not len(eligible):
            break
        left = int(eligible[-1])
        if intensity[left + 1] > intensity[left]:
            intensity[left] = intensity[left + 1]
        mz[left] = mz[left + 1]
        mz = np.delete(mz, left + 1)
        intensity = np.delete(intensity, left + 1)
    return mz, intensity


def metdna3_forward_dot(
    left_tensor: np.ndarray,
    right_tensor: np.ndarray,
    *,
    tolerance_ppm: float = 30.0,
    mz_floor: float = 400.0,
) -> float:
    """Calculate the MetDNA3 linked-feature MS2 similarity constraint."""
    left_precursor, left_mz, left_intensity = spectrum_from_dreams_tensor(left_tensor)
    right_precursor, right_mz, right_intensity = spectrum_from_dreams_tensor(right_tensor)

    # MetDNA3 Methods: discard fragments from the higher-precursor spectrum
    # that exceed the precursor m/z of the lower-precursor feature.
    common_ceiling = min(left_precursor, right_precursor)
    left_keep = left_mz <= common_ceiling
    right_keep = right_mz <= common_ceiling
    left_mz, left_intensity = left_mz[left_keep], left_intensity[left_keep]
    right_mz, right_intensity = right_mz[right_keep], right_intensity[right_keep]
    if not len(left_mz) or not len(right_mz):
        return 0.0

    union_mz = np.sort(np.concatenate([left_mz, right_mz]), kind="stable")
    left_aligned = _align_to_union(left_mz, left_intensity, union_mz)
    right_aligned = _align_to_union(right_mz, right_intensity, union_mz)
    left_union, left_aligned = _collapse_nearby(
        union_mz, left_aligned, tolerance_ppm=tolerance_ppm, mz_floor=mz_floor
    )
    right_union, right_aligned = _collapse_nearby(
        union_mz, right_aligned, tolerance_ppm=tolerance_ppm, mz_floor=mz_floor
    )
    if len(left_union) != len(right_union) or not np.allclose(left_union, right_union, rtol=0, atol=0):
        raise RuntimeError("MetDNA alignment invariance failed")
    denominator = float(np.linalg.norm(left_aligned) * np.linalg.norm(right_aligned))
    return float(np.dot(left_aligned, right_aligned) / denominator) if denominator > 0 else 0.0


def metdna3_reverse_dot(
    left_tensor: np.ndarray,
    right_tensor: np.ndarray,
    *,
    tolerance_ppm: float = 25.0,
    mz_floor: float = 400.0,
) -> float:
    """Calculate the current MetDNA3 v3.1.1 feature-edge ``scoreReverse``.

    The public implementation always treats the spectrum with the smaller
    precursor as the reference.  Reverse scoring retains only aligned bins in
    which that reference has signal; extra peaks in the larger-precursor
    experimental spectrum therefore do not lower the score.
    """
    left_precursor, left_mz, left_intensity = spectrum_from_dreams_tensor(left_tensor)
    right_precursor, right_mz, right_intensity = spectrum_from_dreams_tensor(right_tensor)
    if left_precursor >= right_precursor:
        experimental_mz, experimental_intensity = left_mz, left_intensity
        reference_mz, reference_intensity = right_mz, right_intensity
    else:
        experimental_mz, experimental_intensity = right_mz, right_intensity
        reference_mz, reference_intensity = left_mz, left_intensity
    if not len(experimental_mz) or not len(reference_mz):
        return 0.0

    union_mz = np.sort(np.concatenate([experimental_mz, reference_mz]), kind="stable")
    experimental = _align_to_union(experimental_mz, experimental_intensity, union_mz)
    reference = _align_to_union(reference_mz, reference_intensity, union_mz)
    experimental_union, experimental = _collapse_nearby(
        union_mz, experimental, tolerance_ppm=tolerance_ppm, mz_floor=mz_floor
    )
    reference_union, reference = _collapse_nearby(
        union_mz, reference, tolerance_ppm=tolerance_ppm, mz_floor=mz_floor
    )
    if len(experimental_union) != len(reference_union) or not np.allclose(
        experimental_union, reference_union, rtol=0, atol=0
    ):
        raise RuntimeError("MetDNA alignment invariance failed")
    keep = reference > 0
    experimental, reference = experimental[keep], reference[keep]
    denominator = float(np.linalg.norm(experimental) * np.linalg.norm(reference))
    return float(np.dot(experimental, reference) / denominator) if denominator > 0 else 0.0
