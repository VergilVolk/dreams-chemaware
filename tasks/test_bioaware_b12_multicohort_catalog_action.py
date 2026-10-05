#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    GATE_GRID,
    choose,
    prefix_domain,
)


def main() -> None:
    assert len(EXPECTED_DOMAINS) == 6
    assert len(GATE_GRID) == 15
    frame = pd.DataFrame({"query_id": ["q1", "q2"], "source": ["x", "x"]})
    prefixed = prefix_domain(frame, "D")
    assert prefixed["query_id"].tolist() == ["D::q1", "D::q2"]
    assert prefixed["source"].eq("D").all()
    assert prefixed["polarity"].eq("negative").all()

    rows = pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4"],
        "held_label": ["u"] * 4,
        "source": ["s1", "s1", "s2", "s2"],
        "truth_candidate_id": ["T1", "T2", "T3", "T4"],
        "truth_formula": ["F1", "F2", "F3", "F4"],
        "baseline_candidate_id": ["W1", "T2", "W3", "T4"],
        "proposed_candidate_id": ["T1", "W2", "T3", "T4"],
        "baseline_correct": [False, True, False, True],
        "proposal_unique": [True, True, True, True],
        "proposal_probability": [0.80, 0.60, 0.80, 0.90],
        "baseline_gap": [0.03, 0.03, 0.03, 0.01],
    })
    selected, ledger = choose({"linear_b4_replay": rows})
    assert selected["corrected"] == 2
    assert selected["introduced"] == 0
    assert len(ledger) == 15
    print("[BioAware B12 multicohort action unit checks] PASS", flush=True)


if __name__ == "__main__":
    main()
