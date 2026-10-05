#!/usr/bin/env python
"""Unit check the B19 action evaluator."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b19_residual_action_atlas import evaluate_action


def main() -> None:
    candidates = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2"],
        "source": ["s", "s", "s", "s"],
        "candidate_id": ["a", "b", "c", "d"],
        "truth_candidate_id": ["b", "b", "c", "c"],
        "truth_formula": ["f1", "f1", "f2", "f2"],
        "baseline_candidate_id": ["a", "a", "c", "c"],
        "baseline_correct": [False, False, True, True],
        "baseline_gap": [0.01, 0.01, 0.01, 0.01],
        "feature": [0.0, 1.0, 0.0, 1.0],
    })
    b17 = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "corrected": [False, False],
        "introduced": [False, False],
        "final_correct": [False, True],
    })
    report, rows = evaluate_action(candidates, b17, "feature", "max", 0.05)
    assert report["corrected"] == 1
    assert report["introduced"] == 1
    assert report["risk_net_lambda2"] == -1
    assert report["new_corrected_beyond_B17"] == 1
    assert rows["intervene"].all()
    print("[test_bioaware_b19_residual_action_atlas] PASS")


if __name__ == "__main__":
    main()
