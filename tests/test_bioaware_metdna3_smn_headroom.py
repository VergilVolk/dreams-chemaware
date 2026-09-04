from __future__ import annotations

import pandas as pd

from tasks.audit_bioaware_metdna3_smn_headroom import (
    aggregate_majority,
    evidence_key,
    structure_path_evidence,
)


def test_structure_path_prefers_direct_then_best_bottleneck() -> None:
    candidate_to_nodes = {
        "candidate": [("seed_a", 0.41), ("middle", 0.80)],
    }
    node_to_seeds = {"middle": [("seed_a", 0.70)]}
    direct = structure_path_evidence(
        "candidate", {"seed_a"}, candidate_to_nodes, node_to_seeds
    )
    assert direct["minimum_depth"] == 1
    assert direct["best_bottleneck"] == 0.41

    indirect = structure_path_evidence(
        "candidate", {"seed_b"},
        {"candidate": [("middle", 0.80)]},
        {"middle": [("seed_b", 0.70)]},
    )
    assert indirect["minimum_depth"] == 2
    assert indirect["best_bottleneck"] == 0.70
    assert evidence_key(direct) > evidence_key(indirect)


def test_majority_requires_four_identity_isolated_votes() -> None:
    rows = []
    for fold in range(7):
        rows.append({
            "fold": fold,
            "query_id": "q",
            "truth_candidate_id": "truth",
            "truth_formula": "C2H4O2",
            "baseline_candidate_id": "wrong",
            "network_candidate_id": "truth" if fold < 4 else "wrong",
            "intervene": fold < 4,
            "truth_strict_advantage": fold < 4,
        })
    result = aggregate_majority(pd.DataFrame(rows)).iloc[0]
    assert bool(result.corrected)
    assert result.final_candidate_id == "truth"
    assert result.truth_strict_advantage_votes == 4
    assert bool(result.truth_headroom)


def test_tie_or_three_votes_abstains() -> None:
    rows = []
    for fold in range(7):
        rows.append({
            "fold": fold,
            "query_id": "q",
            "truth_candidate_id": "truth",
            "truth_formula": "C2H4O2",
            "baseline_candidate_id": "wrong",
            "network_candidate_id": "truth" if fold < 3 else "wrong",
            "intervene": fold < 3,
            "truth_strict_advantage": fold < 3,
        })
    result = aggregate_majority(pd.DataFrame(rows)).iloc[0]
    assert result.final_candidate_id == "wrong"
    assert not bool(result.corrected)
    assert not bool(result.truth_headroom)
