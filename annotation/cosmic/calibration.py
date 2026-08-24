"""Decoy E-value + truth calibration (comparable confidence).

COSMIC (Hoffmann et al. 2022, DOI 10.1038/s41587-021-01045-9) obtains an E-value
from *structure-space* proxy decoys. Step-1 uses *spectrum-space* decoys (see
:mod:`annotation.cosmic.decoys`); the E-value is the number of decoys scoring >=
the target, +1 pseudo-count (BLAST / Elias-Gygi convention).

**Measured caveat (library self-retrieval, neg mode, 2026-08-19):** shuffle
decoy-TDA is NOT calibrated for the DreaMS embedder -- it over-estimates FDR ~4x
(and precursor-swap under-estimates ~5x). The defensible, comparable scale is the
self-retrieval *truth* FDR curve, through which the raw score / E-value is
remapped (:func:`build_truth_fdr_curve`, :func:`calibrated_fdr`). A flat curve is
a legitimate finding (the score carries no correctness signal) and is reported.
"""
from __future__ import annotations

import numpy as np


def decoy_evalue(target_score: float, decoy_scores: np.ndarray) -> float:
    """E-value = (# decoys scoring >= target) + 1 (BLAST / Elias-Gygi convention)."""
    return float(np.sum(np.asarray(decoy_scores, dtype=np.float64) >= target_score) + 1.0)


def decoy_fraction_at_least(target_score: float, decoy_scores: np.ndarray) -> float:
    """Empirical decoy P-value = fraction of decoys scoring >= the target."""
    return float(np.mean(np.asarray(decoy_scores, dtype=np.float64) >= target_score))


def build_truth_fdr_curve(
    scores: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Quantile-binned truth-FDR curve: score -> fraction of incorrect annotations.

    ``correct`` is the boolean self-retrieval truth (top-1 InChIKey == query's own,
    under the m/z hard constraint). Returns bin edges, per-bin FDR and counts. Every
    input spectrum lands in exactly one bin (including scores equal to the maximum).

    Binning rule (shared with :func:`calibrated_fdr`): ``bin_idx = digitize(scores,
    edges[1:-1], right=False)`` -- interior edges only, so the last bin is closed on
    the max. Using ``edges[1:]`` would assign ``score == max`` to an out-of-range bin
    and silently drop those spectra (regression-tested).
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if len(scores) == 0:
        return {"edges": [], "centers": [], "fdr": [], "count": [], "n_wrong": []}

    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantiles)
    n_bin = len(edges) - 1
    if n_bin < 1:
        # all scores equal -> single bin covering the whole range
        pad = 1e-9
        edges = np.asarray([scores.min() - pad, scores.max() + pad])
        n_bin = 1
        bin_idx = np.zeros(len(scores), dtype=np.int64)
    else:
        bin_idx = np.digitize(scores, edges[1:-1], right=False)

    centers, fdr, counts, wrongs = [], [], [], []
    for b in range(n_bin):
        mask = bin_idx == b
        counts.append(int(mask.sum()))
        nw = int((~correct[mask]).sum())
        wrongs.append(nw)
        fdr.append(nw / max(counts[-1], 1))
        centers.append(0.5 * (edges[b] + edges[b + 1]))
    return {
        "edges": edges.tolist(),
        "centers": centers,
        "fdr": fdr,
        "count": counts,
        "n_wrong": wrongs,
    }


def calibrated_fdr(score: float, edges: np.ndarray, fdr: np.ndarray) -> float:
    """Look a score up in a truth-FDR table (same binning as :func:`build_truth_fdr_curve`)."""
    edges = np.asarray(edges, dtype=np.float64)
    fdr = np.asarray(fdr, dtype=np.float64)
    if len(fdr) == 0:
        raise ValueError("fdr is empty")
    if len(edges) != len(fdr) + 1:
        raise ValueError("edges must be one element longer than fdr")
    b = int(np.digitize(score, edges[1:-1], right=False))
    return float(fdr[min(b, len(fdr) - 1)])
