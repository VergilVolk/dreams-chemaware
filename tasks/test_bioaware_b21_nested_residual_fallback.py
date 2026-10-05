#!/usr/bin/env python
"""Unit checks for B21 abstention-only fallback."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b21_nested_residual_fallback import apply_fallback


def main() -> None:
    base = pd.DataFrame({
        "query_id": ["q1", "q2"], "source": ["s", "s"],
        "truth_candidate_id": ["b", "d"], "truth_formula": ["f1", "f2"],
        "baseline_candidate_id": ["a", "c"], "baseline_correct": [False, False],
        "baseline_gap": [0.01, 0.01], "intervene": [False, True],
        "final_candidate_id": ["a", "d"], "final_correct": [False, True],
        "corrected": [False, True], "introduced": [False, False], "delta": [0, 1],
    })
    candidates = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2"],
        "candidate_id": ["a", "b", "c", "d"],
        "known_mass_candidate_fraction": [0.0, 1.0, 0.0, 1.0],
        "baseline_gap": [0.01, 0.01, 0.01, 0.01],
    })
    result = apply_fallback(base, candidates, 0.05)
    assert result["final_candidate_id"].tolist() == ["b", "d"]
    assert result["fallback_intervene"].tolist() == [True, False]
    assert result["corrected"].tolist() == [True, True]
    print("[test_bioaware_b21_nested_residual_fallback] PASS")


if __name__ == "__main__":
    main()
