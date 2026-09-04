from __future__ import annotations

import numpy as np
import pandas as pd

from annotation.bioaware import (
    BioAwareConfig,
    aggregate_query_support,
    build_one_hop_evidence,
    compound_reaction_degree,
    degree_preserving_reaction_decoy,
    fuse_candidates,
    top1_transition_table,
)


def participants() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"reaction_id": "r1", "side": "left", "compound_id": "A"},
            {"reaction_id": "r1", "side": "right", "compound_id": "B"},
            {"reaction_id": "r2", "side": "left", "compound_id": "B"},
            {"reaction_id": "r2", "side": "right", "compound_id": "C"},
            {"reaction_id": "r3", "side": "left", "compound_id": "D"},
            {"reaction_id": "r3", "side": "right", "compound_id": "E"},
        ]
    )


def test_one_hop_support_and_truth_exclusion():
    seeds = pd.DataFrame(
        [
            {"seed_query_id": "external-A", "seed_compound_id": "A", "seed_score": 0.9},
            {"seed_query_id": "q1", "seed_compound_id": "B", "seed_score": 0.99},
            {"seed_query_id": "external-C", "seed_compound_id": "C", "seed_score": 0.9},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"query_id": "q1", "candidate_id": "A", "spectral_score": 0.61, "truth": "B"},
            {"query_id": "q1", "candidate_id": "B", "spectral_score": 0.60, "truth": "B"},
        ]
    )
    paths = build_one_hop_evidence(participants(), seeds)
    supported, explanations = aggregate_query_support(
        candidates, paths, truth_col="truth", exclude_same_query=True, exclude_truth_identity=True
    )
    b = supported[supported.candidate_id == "B"].iloc[0]
    a = supported[supported.candidate_id == "A"].iloc[0]
    assert b.network_support > 0
    assert a.network_support == 0
    assert set(explanations.seed_compound_id) == {"A", "C"}


def test_low_margin_network_can_correct_but_high_margin_conflict_abstains():
    seeds = pd.DataFrame(
        [
            {"seed_query_id": "sA", "seed_compound_id": "A", "seed_score": 0.95},
            {"seed_query_id": "sC", "seed_compound_id": "C", "seed_score": 0.95},
        ]
    )
    low = pd.DataFrame(
        [
            {"query_id": "low", "candidate_id": "A", "spectral_score": 0.61, "truth": "B"},
            {"query_id": "low", "candidate_id": "B", "spectral_score": 0.60, "truth": "B"},
        ]
    )
    high = pd.DataFrame(
        [
            {"query_id": "high", "candidate_id": "A", "spectral_score": 0.90, "truth": "B"},
            {"query_id": "high", "candidate_id": "B", "spectral_score": 0.60, "truth": "B"},
        ]
    )
    candidates = pd.concat([low, high], ignore_index=True)
    paths = build_one_hop_evidence(participants(), seeds)
    supported, _ = aggregate_query_support(
        candidates, paths, truth_col="truth", exclude_truth_identity=True
    )
    scored, decisions = fuse_candidates(supported)
    per_query, summary = top1_transition_table(scored, truth_col="truth")
    low_decision = decisions.set_index("query_id").loc["low"]
    high_decision = decisions.set_index("query_id").loc["high"]
    assert low_decision.bioaware_applied
    assert low_decision.final_top_candidate == "B"
    assert not high_decision.bioaware_applied
    assert high_decision.evidence_state == "spectral_strong_network_conflict"
    assert summary["corrected"] == 1
    assert summary["introduced"] == 0
    assert np.isclose(summary["delta_recall1"], 0.5)


def test_no_network_evidence_is_exact_noop():
    candidates = pd.DataFrame(
        [
            {"query_id": "q", "candidate_id": "X", "spectral_score": 0.7},
            {"query_id": "q", "candidate_id": "Y", "spectral_score": 0.6},
        ]
    )
    candidates["network_support"] = 0.0
    candidates["network_path_count"] = 0
    scored, decisions = fuse_candidates(candidates)
    assert np.array_equal(scored.spectral_score.to_numpy(), scored.final_score.to_numpy())
    assert decisions.iloc[0].evidence_state == "no_network_evidence"


def test_currency_and_high_degree_seeds_are_excluded():
    p = participants()
    p.loc[p.compound_id == "A", "is_currency"] = True
    seeds = pd.DataFrame(
        [{"seed_query_id": "s", "seed_compound_id": "A", "seed_score": 1.0}]
    )
    paths = build_one_hop_evidence(p, seeds)
    assert paths.empty


def test_degree_preserving_decoy_preserves_compound_and_side_degrees():
    p = participants()
    original_compound_degree = p.groupby("compound_id").size().sort_index()
    original_side_size = (
        p.assign(side_node=p.reaction_id + "|" + p.side)
        .groupby("side_node").size().sort_index()
    )
    decoy = degree_preserving_reaction_decoy(p, seed=7, swaps_per_edge=2)
    decoy_compound_degree = decoy.groupby("compound_id").size().sort_index()
    decoy_side_size = (
        decoy.assign(side_node=decoy.reaction_id + "|" + decoy.side)
        .groupby("side_node").size().sort_index()
    )
    pd.testing.assert_series_equal(original_compound_degree, decoy_compound_degree)
    pd.testing.assert_series_equal(original_side_size, decoy_side_size)


def test_directed_mode_only_propagates_left_to_right():
    p = participants().query("reaction_id == 'r1'")
    seeds = pd.DataFrame(
        [
            {"seed_query_id": "left", "seed_compound_id": "A", "seed_score": 1.0},
            {"seed_query_id": "right", "seed_compound_id": "B", "seed_score": 1.0},
        ]
    )
    paths = build_one_hop_evidence(p, seeds, BioAwareConfig(directed=True))
    assert set(paths.candidate_id) == {"B"}
    assert set(paths.seed_compound_id) == {"A"}

