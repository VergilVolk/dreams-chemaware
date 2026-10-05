#!/usr/bin/env python
"""Small deterministic checks for B40-M2 cohort and action logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from evaluate_bioaware_b40_m2_context_strata import apply_action  # noqa: E402


def main() -> None:
    base = pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "source": ["ST001154_same_formula_10ppm", "KGMN200STD_hidden_seed", "BV2cell"],
        "common_risk_eligible": [True, False, True],
        "topology_candidate_id": ["a", "a", "a"],
        "truth_candidate_id": ["b", "b", "a"],
    })
    proposal = pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "mapped_candidate_count": [2, 2, 1],
        "graph_max_score": [1.0, 1.0, 1.0],
        "graph_proposal_unique": [True, True, True],
        "graph_proposed_candidate_id": ["b", "b", "b"],
    })
    out = apply_action(base, proposal)
    assert out["graph_intervene"].tolist() == [True, False, False]
    assert out["graph_final_correct"].tolist() == [True, False, True]
    assert out["context_stratum"].tolist() == [
        "sample_local_leave_one_seed_out", "hidden_standard", "synthetic_rotation"
    ]
    print("[test_bioaware_b40_m2_context_strata] PASS", flush=True)


if __name__ == "__main__":
    main()
