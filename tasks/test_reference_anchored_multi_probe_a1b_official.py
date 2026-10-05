"""Focused tests for A1b official-DreaMS multi-probe confirmation."""

from __future__ import annotations

import math
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

import pilot_reference_anchored_multi_probe_a1b_official as pilot  # noqa: E402


def main() -> None:
    values = pilot.normalize_rows(np.asarray([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)

    direct = np.asarray([
        [1.0, 0.2, 0.8, 0.1],
        [0.2, 1.0, 0.7, 0.2],
        [0.8, 0.7, 1.0, 0.4],
        [0.1, 0.2, 0.4, 1.0],
    ], dtype=np.float32)
    before, landmarks = pilot.build_profile(direct, minimum_anchors=2)
    changed = direct.copy()
    changed[0, 1] = changed[1, 0] = 0.99
    after, _ = pilot.build_profile(changed, minimum_anchors=2)
    assert landmarks == 2
    assert math.isclose(float(before[0, 1]), float(after[0, 1]))

    # Cache rows must be aligned by the frozen reference-spectrum IDs rather
    # than by incidental manifest order. Multiple stereo SMILES per IK14 are
    # legal because the experiment evaluates connectivity.
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        cache = tmp / "cache"
        cache.mkdir()
        reference = pd.DataFrame({
            "reference_spectrum_id": ["R2", "R1", "R3"],
            "panel": ["neg_rp"] * 3,
            "ik14": ["A", "A", "B"],
            "smiles": ["A@", "A@@", "B"],
            "name": ["a2", "a1", "b"],
            "precursor_mz": [100.0, 100.0, 120.0],
        })
        alignment = pd.DataFrame({
            "panel": ["neg_rp"] * 3,
            "subset_row": [0, 1, 2],
            "reference_spectrum_id": ["R1", "R2", "R3"],
        })
        pd.DataFrame({"ik14": ["A", "A", "B"]}).to_csv(cache / "manifest.csv", index=False)
        np.save(cache / "embeddings.npy", np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32))
        manifest_sha = pilot.sha256(cache / "manifest.csv")
        (cache / "report.json").write_text(
            json.dumps({
                "status": "unified_library_p2b_cache_complete",
                "model": {"kind": "official_dreams"},
                "provenance": {
                    "official_checkpoint_sha256": "abc",
                    "mgf_sha256": "def",
                    "manifest_sha256": manifest_sha,
                },
            }),
            encoding="utf-8",
        )
        table, matrix, info = pilot.build_identity_cosine(
            reference, alignment, cache, "neg_rp", "abc", "def"
        )
        assert table.ik14.tolist() == ["A", "B"]
        assert table.distinct_stereo_smiles.tolist() == [2, 1]
        assert matrix.shape == (2, 2)
        assert info["spectra"] == 3

    original_fingerprint, original_tanimoto = pilot.fingerprint, pilot.tanimoto
    try:
        pilot.fingerprint = lambda value: float(value)
        pilot.tanimoto = lambda left, right: 1.0 - abs(left - right) / 9.0
        n = 10
        identities = pd.DataFrame({
            "ik14": [f"ID{i:012d}" for i in range(n)],
            "representative_name": [f"compound_{i}" for i in range(n)],
            "smiles": [str(i) for i in range(n)],
            "precursor_mz": np.linspace(100.0, 190.0, n),
            "reference_spectra": np.ones(n, dtype=int),
        })
        synthetic = 1.0 - np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / 9.0
        per_query, report = pilot.evaluate_panel(
            "synthetic", identities, synthetic.astype(np.float32), minimum_anchors=8
        )
        assert len(per_query) == n
        assert report["landmarks_per_comparison"] == 8
        assert set(report["metrics"]) == set(pilot.SCORE_NAMES)
    finally:
        pilot.fingerprint, pilot.tanimoto = original_fingerprint, original_tanimoto
    print("[test_reference_anchored_multi_probe_a1b_official] PASS")


if __name__ == "__main__":
    main()
