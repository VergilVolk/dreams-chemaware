"""Corrected spectral similarity functions for all P0/P2 evaluations.

Replaces the buggy homemade entropy (identical spectra → 0 instead of 1)
with the pinned msentropy backend from the existing score bundle pipeline.
Also fixes greedy cosine to properly handle sorted m/z arrays.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("tasks"))

# Import the CORRECT entropy implementation from the validated pipeline
from noise_gnps_article_spectral_scores import weighted_entropy_similarity  # noqa: E402

TOLERANCE = 0.02


def norm_peaks_sorted(peaks, max_peaks=100):
    """Normalize peaks: top-intensity slice, then SORT BY M/Z (critical for searchsorted)."""
    p = sorted(sorted(peaks, key=lambda x: -x[1])[:max_peaks], key=lambda x: x[0])
    mzs = np.array([x[0] for x in p], dtype=np.float64)
    ints = np.array([x[1] for x in p], dtype=np.float64)
    if ints.max() > 0:
        ints /= ints.max()
    return mzs, ints


def greedy_cosine_correct(a_mz, a_int, b_mz, b_int):
    """Greedy cosine on m/z-SORTED arrays (fixes the searchsorted-on-unsorted bug)."""
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, val in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - TOLERANCE, mz + TOLERANCE))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]
                used[best] = True
                matched += float(val * b_int[best])
    na = float((a_int ** 2).sum())
    nb = float((b_int ** 2).sum())
    return matched / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0


def entropy_similarity_correct(a_mz, a_int, b_mz, b_int):
    """Proper entropy similarity using the pinned msentropy backend.

    For identical spectra, returns ~1.0 (not 0 like the buggy version).
    """
    spec_a = np.vstack([a_mz, a_int])
    spec_b = np.vstack([b_mz, b_int])
    return float(weighted_entropy_similarity(spec_a, spec_b, TOLERANCE))


def score_pair_correct(spec_a_peaks, spec_b_peaks):
    """Score a pair of spectra with both corrected metrics.

    Returns dict with cosine, entropy, and combined (average).
    """
    a_mz, a_int = norm_peaks_sorted(spec_a_peaks)
    b_mz, b_int = norm_peaks_sorted(spec_b_peaks)
    cos = greedy_cosine_correct(a_mz, a_int, b_mz, b_int)
    ent = entropy_similarity_correct(a_mz, a_int, b_mz, b_int)
    return {"cosine": cos, "entropy": ent, "combined": (cos + ent) / 2}


def self_test():
    """Verify identical spectra return ~1.0 for entropy (the bug fix)."""
    peaks = [(100.0, 50.0), (150.0, 80.0), (200.0, 90.0), (250.0, 30.0)]
    a_mz, a_int = norm_peaks_sorted(peaks)
    b_mz, b_int = norm_peaks_sorted(peaks)

    cos = greedy_cosine_correct(a_mz, a_int, b_mz, b_int)
    ent = entropy_similarity_correct(a_mz, a_int, b_mz, b_int)

    print(f"identical spectra: cosine={cos:.4f}, entropy={ent:.4f}")
    assert cos > 0.99, f"cosine should be ~1.0, got {cos}"
    assert ent > 0.99, f"entropy should be ~1.0, got {ent}"  # THIS IS THE FIX

    # Dissimilar spectra should score low
    peaks_c = [(50.0, 60.0), (75.0, 70.0), (300.0, 90.0)]
    c_mz, c_int = norm_peaks_sorted(peaks_c)
    ent_diff = entropy_similarity_correct(a_mz, a_int, c_mz, c_int)
    cos_diff = greedy_cosine_correct(a_mz, a_int, c_mz, c_int)
    print(f"dissimilar: cosine={cos_diff:.4f}, entropy={ent_diff:.4f}")

    print("SELF-TEST PASSED")


if __name__ == "__main__":
    self_test()
