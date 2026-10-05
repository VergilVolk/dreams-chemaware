#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b13_hub_veto_action import apply_veto  # noqa: E402


def main() -> None:
    scored = pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "held_label": ["u"] * 3,
        "source": ["s"] * 3,
        "truth_candidate_id": ["T1", "T2", "T3"],
        "truth_formula": ["F1", "F2", "F3"],
        "baseline_candidate_id": ["T1", "W2", "W3"],
        "proposed_candidate_id": ["W1", "T2", "T3"],
        "baseline_correct": [True, False, False],
        "proposal_unique": [True, True, True],
        "proposal_probability": [0.8, 0.8, 0.8],
        "baseline_gap": [0.02, 0.02, 0.02],
        "delta_network_member": [0.0, 1.0, 0.0],
        "delta_known_log_degree": [2.1, 2.1, 1.0],
        "delta_known_mass_candidate_fraction": [0.0, 0.0, 0.0],
    })
    result = apply_veto(
        scored, 0.05, 0.7, "degree_only_jump_ge_2_0", 2.0
    )
    assert result["veto_applied"].tolist() == [True, False, False]
    assert int(result["corrected"].sum()) == 2
    assert int(result["introduced"].sum()) == 0
    print("[BioAware B13 hub-veto unit checks] PASS", flush=True)


if __name__ == "__main__":
    main()
