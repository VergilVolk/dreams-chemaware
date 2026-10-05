"""Focused tests for the low-cost multi-probe coordinate pilot."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

import pilot_reference_anchored_multi_probe_a1 as pilot  # noqa: E402
from pilot_reference_anchored_multi_probe_a1 import (  # noqa: E402
    CHANNELS,
    average_ranks,
    multi_anchor_score,
    ndcg_at_k,
    rank_concordance,
    spearman,
)


def main() -> None:
    ranks = average_ranks(np.asarray([3.0, 1.0, 1.0, 2.0]))
    assert np.allclose(ranks, [3.0, 0.5, 0.5, 2.0])
    assert math.isclose(rank_concordance(np.arange(8), np.arange(8)), 1.0)
    assert rank_concordance(np.arange(8), np.arange(7, -1, -1)) < 0.5
    assert math.isclose(spearman(np.arange(8), np.arange(8)), 1.0)
    assert math.isclose(spearman(np.arange(8), np.arange(7, -1, -1)), -1.0)
    relevance = np.asarray([1.0, 0.7, 0.1, 0.0])
    ideal = ndcg_at_k(relevance, np.asarray([0, 1, 2, 3]), 3)
    bad = ndcg_at_k(relevance, np.asarray([3, 2, 1, 0]), 3)
    assert math.isclose(ideal, 1.0)
    assert bad < ideal

    # Changing only the direct q-c edge must not change their multi-anchor
    # score.  This guards against silently collapsing the method to search.
    base = np.asarray([
        [1.0, 0.2, 0.8, 0.1],
        [0.2, 1.0, 0.7, 0.2],
        [0.8, 0.7, 1.0, 0.4],
        [0.1, 0.2, 0.4, 1.0],
    ])
    matrices = {name: base.copy() for name in CHANNELS}
    before = multi_anchor_score(matrices, 0, 1)
    for matrix in matrices.values():
        matrix[0, 1] = matrix[1, 0] = 0.99
    after = multi_anchor_score(matrices, 0, 1)
    assert math.isclose(before, after), "direct q-c edge leaked into the landmark profile"

    # Exercise the complete evaluation path without requiring RDKit in the
    # lightweight local runtime.
    original_fingerprint, original_tanimoto = pilot.fingerprint, pilot.tanimoto
    try:
        pilot.fingerprint = lambda value: float(value)
        pilot.tanimoto = lambda left, right: 1.0 - abs(left - right) / 9.0
        n = 10
        frame = pd.DataFrame({
            "ik14": [f"ID{i:012d}" for i in range(n)],
            "representative_name": [f"compound_{i}" for i in range(n)],
            "smiles": [str(i) for i in range(n)],
            "precursor_mz": np.linspace(100.0, 190.0, n),
            "reference_spectra": np.ones(n, dtype=int),
        })
        synthetic = 1.0 - np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 9.0
        synthetic_matrices = {name: synthetic.copy() for name in CHANNELS}
        evaluated, report = pilot.evaluate_panel(
            "synthetic", frame, synthetic_matrices, minimum_anchors=8
        )
        assert len(evaluated) == n
        assert report["minimum_landmarks"] == 8
        assert set(report["metrics"]) == set(pilot.SCORE_NAMES)
    finally:
        pilot.fingerprint, pilot.tanimoto = original_fingerprint, original_tanimoto
    print("[test_reference_anchored_multi_probe_a1] PASS")


if __name__ == "__main__":
    main()
