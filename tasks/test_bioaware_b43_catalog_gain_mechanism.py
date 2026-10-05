#!/usr/bin/env python
"""Unit checks for B43 membership strata."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b43_catalog_gain_mechanism import attach_query_features  # noqa: E402


def main() -> None:
    transitions = pd.DataFrame({
        "arm": ["x", "x"], "query_id": ["q1", "q2"],
        "truth_candidate_id": ["t1", "t2"],
        "baseline_candidate_id": ["b1", "b2"],
        "final_candidate_id": ["t1", "b2"],
    })
    candidates = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2"],
        "candidate_id": ["t1", "b1", "t2", "b2"],
        "member": [1.0, 0.0, 1.0, 1.0],
    })
    result = attach_query_features(transitions, candidates, "x", "member")
    assert result["membership_stratum"].tolist() == ["truth_only", "both_truth_and_baseline"]
    assert result["all_candidates_member"].tolist() == [0.0, 1.0]
    print("[test_bioaware_b43_catalog_gain_mechanism] PASS", flush=True)


if __name__ == "__main__":
    main()
