#!/usr/bin/env python
"""Unit checks for the B17 policy combiner."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b17_nested_union_action import combine


def table(final: list[str], intervene: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": ["q1", "q2"], "held_label": ["h", "h"],
        "source": ["s", "s"], "truth_candidate_id": ["t1", "t2"],
        "truth_formula": ["f1", "f2"], "baseline_candidate_id": ["w1", "t2"],
        "baseline_correct": [False, True], "intervene": intervene,
        "final_candidate_id": final,
    })


def main() -> None:
    linear = table(["t1", "t2"], [True, False])
    nonlinear = table(["w1", "x2"], [False, True])
    union = combine(linear, nonlinear, "union_b12_priority")
    assert union["final_candidate_id"].tolist() == ["t1", "x2"]
    assert int(union["corrected"].sum()) == 1
    assert int(union["introduced"].sum()) == 1
    agreement = combine(linear, nonlinear, "agreement_only")
    assert agreement["final_candidate_id"].tolist() == ["w1", "t2"]
    print("[test_bioaware_b17_nested_union_action] PASS")


if __name__ == "__main__":
    main()
