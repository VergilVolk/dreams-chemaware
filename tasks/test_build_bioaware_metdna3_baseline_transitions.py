#!/usr/bin/env python
"""Deterministic smoke test for baseline rotation expansion and tie handling."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scores = root / "scores.csv"
        splits = root / "splits.csv"
        output = root / "out"
        pd.DataFrame([
            {"query_id": "q1", "candidate_id": "A", "spectral_score": .5, "truth_candidate_id": "A", "truth_formula": "F1"},
            {"query_id": "q1", "candidate_id": "B", "spectral_score": .4, "truth_candidate_id": "A", "truth_formula": "F1"},
            {"query_id": "q2", "candidate_id": "C", "spectral_score": .2, "truth_candidate_id": "C", "truth_formula": "F2"},
            {"query_id": "q2", "candidate_id": "D", "spectral_score": .2, "truth_candidate_id": "C", "truth_formula": "F2"},
        ]).to_csv(scores, index=False)
        pd.DataFrame([
            {"fold": 0, "ik14": "A", "role": "heldout"},
            {"fold": 1, "ik14": "A", "role": "heldout"},
            {"fold": 0, "ik14": "C", "role": "heldout"},
        ]).to_csv(splits, index=False)
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("build_bioaware_metdna3_baseline_transitions.py")),
            "--scores", str(scores), "--splits", str(splits), "--output-dir", str(output),
        ], check=True, capture_output=True, text=True)
        result = pd.read_csv(output / "raw_transitions.csv.gz")
        assert len(result) == 3
        assert result[result.query_id.eq("q1")].baseline_correct.all()
        assert not result[result.query_id.eq("q2")].baseline_correct.any()
    print("[test_build_bioaware_metdna3_baseline_transitions] PASS")


if __name__ == "__main__":
    main()
