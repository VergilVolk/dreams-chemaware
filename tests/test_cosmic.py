"""Unit tests for the COSMIC confidence layer (annotation/cosmic/).

Covers the deterministic, import-light core: rule-coherence score, spectrum-space
decoy generators, decoy E-value, and truth-FDR calibration (incl. a regression for
the binning bug that dropped max-score spectra). Nothing touches the network or a
trained model, so the suite runs on CPU in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation.cosmic import (  # noqa: E402
    StructureSpaceDecoy,
    ShuffleIntensityDecoy,
    ShuffleMZDecoy,
    build_truth_fdr_curve,
    calibrated_fdr,
    decoy_evalue,
    decoy_fraction_at_least,
    roc_auc,
    rule_coherence_scores,
)


# --------------------------------------------------------------------------- #
# score.roc_auc
# --------------------------------------------------------------------------- #
def test_roc_auc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_roc_auc_reverse_separation():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_roc_auc_degenerate_returns_uninformative():
    assert roc_auc(np.array([1, 1, 1]), np.array([0.5, 0.6, 0.7])) == 0.5  # no negatives
    assert roc_auc(np.array([0, 0, 0]), np.array([0.5, 0.6, 0.7])) == 0.5  # no positives


def test_roc_auc_handles_ties():
    # two positives tied at score 0.5, two negatives tied at 0.5 -> tied ranks -> 0.5
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# score.rule_coherence_scores
# --------------------------------------------------------------------------- #
def test_rule_coherence_scores_shape_and_range():
    rng = np.random.default_rng(0)
    n, d, k = 8, 16, 6
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    hits = rng.integers(0, 2, (n, k)).astype(np.uint8)
    idx = np.arange(k)
    w = rng.standard_normal((k, d)).astype(np.float32)
    b = np.zeros(k, np.float32)
    mean = np.zeros(d, np.float32)
    std = np.ones(d, np.float32)
    s = rule_coherence_scores(emb, hits, idx, w, b, mean, std)
    assert s.shape == (n,)
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_rule_coherence_scores_zero_positive_is_uninformative():
    emb = np.full((1, 8), 1.0 / np.sqrt(8), dtype=np.float32)
    hits = np.array([[0, 0, 0]], dtype=np.uint8)
    idx = np.arange(3)
    w = np.zeros((3, 8), np.float32)
    b = np.zeros(3, np.float32)
    mean = np.zeros(8, np.float32)
    std = np.ones(8, np.float32)
    s = rule_coherence_scores(emb, hits, idx, w, b, mean, std)
    assert s[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# decoys
# --------------------------------------------------------------------------- #
def test_shuffle_intensity_preserves_mz_permutes_intensity():
    peaks = np.vstack([np.array([10.0, 20.0, 30.0, 40.0]),
                       np.array([1.0, 2.0, 3.0, 4.0])]).astype(np.float32)
    decoys = ShuffleIntensityDecoy().generate(peaks, 500.0, n=1, seed=0)
    assert len(decoys) == 1
    d = decoys[0]
    assert d["precursor_mz"] == 500.0
    assert np.array_equal(d["peaks"][0], peaks[0])  # m/z axis intact
    assert np.array_equal(np.sort(d["peaks"][1]), np.sort(peaks[1]))  # intensities permuted


def test_shuffle_mz_permutes_mz_preserves_intensity():
    peaks = np.vstack([np.array([10.0, 20.0, 30.0, 40.0]),
                       np.array([1.0, 2.0, 3.0, 4.0])]).astype(np.float32)
    decoys = ShuffleMZDecoy().generate(peaks, 500.0, n=1, seed=0)
    d = decoys[0]
    assert d["precursor_mz"] == 500.0
    assert np.array_equal(d["peaks"][1], peaks[1])  # intensities intact
    assert np.array_equal(np.sort(d["peaks"][0]), np.sort(peaks[0]))  # m/z permuted


def test_decoy_seed_determinism_and_independence():
    peaks = np.vstack([np.arange(1, 11), np.arange(11, 21)]).astype(np.float32)
    a = ShuffleIntensityDecoy().generate(peaks, 100.0, n=2, seed=7)
    b = ShuffleIntensityDecoy().generate(peaks, 100.0, n=2, seed=7)
    c = ShuffleIntensityDecoy().generate(peaks, 100.0, n=2, seed=8)
    assert np.array_equal(a[0]["peaks"], b[0]["peaks"])  # same seed -> same permutation
    assert not np.array_equal(a[0]["peaks"][1], c[0]["peaks"][1])  # different seed


def test_structure_space_decoy_is_reserved():
    with pytest.raises(NotImplementedError):
        StructureSpaceDecoy().generate(np.zeros((2, 3), np.float32), 100.0)


# --------------------------------------------------------------------------- #
# decoy E-value
# --------------------------------------------------------------------------- #
def test_decoy_evalue_count_plus_one():
    assert decoy_evalue(0.9, np.array([0.1, 0.2, 0.95])) == 2.0  # 1 decoy >= 0.9
    assert decoy_evalue(0.99, np.array([0.1, 0.2, 0.3])) == 1.0  # 0 decoys >= 0.99


def test_decoy_fraction_at_least():
    assert decoy_fraction_at_least(0.9, np.array([0.1, 0.2, 0.95])) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# truth calibration
# --------------------------------------------------------------------------- #
def test_truth_fdr_curve_drops_no_spectra():
    # regression: scores equal to the max (1.0) and min (0.0) must not be dropped
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, 100)
    scores[0:5] = 1.0
    scores[5:10] = 0.0
    correct = rng.integers(0, 2, 100).astype(bool)
    out = build_truth_fdr_curve(scores, correct, n_bins=10)
    assert sum(out["count"]) == 100
    assert len(out["edges"]) == len(out["fdr"]) + 1
    assert all(0.0 <= f <= 1.0 for f in out["fdr"])
    assert out["n_wrong"] == [int(w) for w in out["n_wrong"]]


def test_truth_fdr_curve_all_equal_single_bin():
    scores = np.full(50, 0.5)
    correct = np.zeros(50, dtype=bool)
    out = build_truth_fdr_curve(scores, correct, n_bins=10)
    assert sum(out["count"]) == 50
    assert len(out["fdr"]) == 1
    assert out["fdr"][0] == 1.0  # all wrong


def test_truth_fdr_curve_empty():
    out = build_truth_fdr_curve(np.array([]), np.array([]))
    assert out["count"] == []


def test_calibrated_fdr_max_lands_in_top_bin():
    scores = np.array([0.1, 0.5, 1.0, 1.0])
    correct = np.array([False, True, True, False])
    out = build_truth_fdr_curve(scores, correct, n_bins=3)
    edges = np.asarray(out["edges"])
    fdr = np.asarray(out["fdr"])
    assert calibrated_fdr(1.0, edges, fdr) == fdr[-1]  # max -> top bin, not out of range


def test_calibrated_fdr_validation():
    with pytest.raises(ValueError):
        calibrated_fdr(0.5, np.array([]), np.array([]))
    with pytest.raises(ValueError):
        calibrated_fdr(0.5, np.array([0.0, 1.0]), np.array([0.5, 0.5]))  # length mismatch
