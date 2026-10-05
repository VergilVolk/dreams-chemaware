#!/usr/bin/env python
"""Unit checks for B22 alternative construction and gating."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b22_crossfit_action_selector import apply_gate


def main() -> None:
    scored = pd.DataFrame({
        "query_id": ["q1", "q2"], "source": ["s", "s"],
        "truth_candidate_id": ["b", "c"], "truth_formula": ["f1", "f2"],
        "baseline_candidate_id": ["a", "c"], "baseline_correct": [False, True],
        "B17_final_candidate_id": ["a", "c"],
        "proposed_candidate_id": ["b", "d"], "proposal_unique": [True, True],
        "proposal_probability": [0.9, 0.6], "B17_probability": [0.5, 0.59],
        "proposal_advantage": [0.4, 0.01], "baseline_gap": [0.02, 0.02],
    })
    result = apply_gate(scored, 0.05, 0.05)
    assert result["final_candidate_id"].tolist() == ["b", "c"]
    assert result["corrected"].tolist() == [True, False]
    assert not result["introduced"].any()
    print("[test_bioaware_b22_crossfit_action_selector] PASS")


if __name__ == "__main__":
    main()
