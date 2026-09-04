from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_bioaware_metdna3_failure_decomposition",
    ROOT / "tasks" / "audit_bioaware_metdna3_failure_decomposition.py",
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_official_top1_counts_ties_against_truth() -> None:
    scores = pd.DataFrame([
        {"query_id": "q", "candidate_id": "truth", "spectral_score": 0.8,
         "truth_candidate_id": "truth", "truth_formula": "C2H4"},
        {"query_id": "q", "candidate_id": "wrong", "spectral_score": 0.8,
         "truth_candidate_id": "truth", "truth_formula": "C2H4"},
    ])
    result = AUDIT.official_top1(scores).iloc[0]
    assert not result.baseline_correct
    assert result.baseline_tie_size == 2


def test_neighbor_map_uses_opposite_side_and_filters_bad_seeds() -> None:
    participants = pd.DataFrame([
        {"reaction_id": "r1", "side": "left", "compound_id": "truth", "is_currency": False},
        {"reaction_id": "r1", "side": "right", "compound_id": "good", "is_currency": False},
        {"reaction_id": "r1", "side": "right", "compound_id": "currency", "is_currency": True},
        {"reaction_id": "r2", "side": "left", "compound_id": "truth", "is_currency": False},
        {"reaction_id": "r2", "side": "right", "compound_id": "high_degree", "is_currency": False},
        {"reaction_id": "r3", "side": "left", "compound_id": "other", "is_currency": False},
        {"reaction_id": "r3", "side": "right", "compound_id": "high_degree", "is_currency": False},
    ])
    nodes, neighbors, eligible = AUDIT.eligible_opposite_side_neighbors(
        participants, {"good", "currency", "high_degree"}, maximum_seed_degree=1
    )
    assert "truth" in nodes
    assert eligible == {"good"}
    assert neighbors["truth"] == {"good"}


def test_primary_bottleneck_prioritizes_coverage_before_score() -> None:
    row = pd.Series({
        "truth_in_rhea": False, "eligible_level1_seed_neighbors": 0,
        "truth_network_path_rotations": 0, "truth_raw_path_rotations": 0,
        "paired_raw_rotations": 0, "mean_raw_truth_minus_wrong": -1.0,
        "network_can_rank_truth_first": False,
    })
    assert AUDIT.primary_bottleneck(row) == "A_truth_absent_from_rhea"
