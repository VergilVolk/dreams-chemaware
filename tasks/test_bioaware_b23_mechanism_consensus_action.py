#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b23_mechanism_consensus_action import (  # noqa: E402
    apply_consensus, consensus_proposals,
)


def main() -> None:
    candidates = pd.DataFrame([
        {"query_id": "q1", "candidate_id": "wrong", "spectral_score": .80,
         "baseline_gap": .03, "mass": 0, "member": 0, "degree": 0,
         "path": 0, "edge0": 0, "edge1": 0, "pred": 0, "co": 0, "multi": 0},
        {"query_id": "q1", "candidate_id": "truth", "spectral_score": .77,
         "baseline_gap": .03, "mass": 1, "member": 1, "degree": 1,
         "path": 1, "edge0": 1, "edge1": 1, "pred": 1, "co": 1, "multi": 1},
    ]).rename(columns={
        "mass": "known_mass_candidate_fraction", "member": "network_member",
        "degree": "known_log_degree", "path": "known_path_fraction",
        "edge0": "edge0_reliability", "edge1": "edge1_bottleneck_mean",
        "pred": "predicted_edge_increment",
        "co": "coabundance_log_neighbours_mean",
        "multi": "coabundance_multiwitness_fraction",
    })
    base = pd.DataFrame([{
        "query_id": "q1", "source": "d", "truth_candidate_id": "truth",
        "truth_formula": "F", "baseline_candidate_id": "wrong",
        "baseline_correct": False, "final_candidate_id": "wrong",
        "final_correct": False, "corrected": False, "introduced": False,
        "intervene": False, "delta": 0,
    }])
    proposal = consensus_proposals(candidates, base).iloc[0]
    assert proposal["proposal_candidate_id"] == "truth"
    assert proposal["proposal_families"] == 3
    result = apply_consensus(base, candidates, {
        "minimum_families": 2, "minimum_votes": 3,
        "maximum_baseline_gap": .05, "maximum_spectral_loss": .05,
    }).iloc[0]
    assert bool(result["corrected"])
    assert result["final_candidate_id"] == "truth"
    print("[test_bioaware_b23_mechanism_consensus_action] PASS")


if __name__ == "__main__":
    main()
