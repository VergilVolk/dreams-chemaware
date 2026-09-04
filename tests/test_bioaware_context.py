from __future__ import annotations

import numpy as np
import pandas as pd

from annotation.bioaware_context import extract_reaction_context_features


def test_complete_and_incomplete_reaction_context_are_separated() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["Q", "Q"],
            "candidate_id": ["A", "B"],
            "spectral_score": [0.8, 0.7],
            "truth": ["A", "A"],
        }
    )
    participants = pd.DataFrame(
        {
            "reaction_id": ["R1", "R1", "R1", "R2", "R2", "R2", "R2"],
            "side": ["left", "left", "right", "left", "left", "right", "right"],
            "compound_id": ["S", "WATER", "A", "T", "MISSING", "B", "P"],
            "is_currency": [False, True, False, False, False, False, False],
            "reaction_weight": [1.0] * 7,
        }
    )
    seeds = pd.DataFrame(
        {
            "seed_query_id": ["S_QUERY", "T_QUERY"],
            "seed_compound_id": ["S", "T"],
            "seed_score": [0.9, 0.9],
        }
    )
    paths = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "seed_compound_id": ["S", "T"],
            "seed_query_id": ["S_QUERY", "T_QUERY"],
            "reaction_id": ["R1", "R2"],
            "seed_side": ["left", "left"],
            "contribution": [0.5, 0.6],
        }
    )
    features, details = extract_reaction_context_features(
        candidates,
        paths,
        participants,
        seeds,
        truth_col="truth",
        exclude_truth_identity=True,
    )
    a = features[features.candidate_id == "A"].iloc[0]
    b = features[features.candidate_id == "B"].iloc[0]
    assert a.complete_path_count == 1 and np.isclose(a.complete_network_support, 0.5)
    assert b.incomplete_path_count == 1 and np.isclose(b.incomplete_network_support, 0.6)
    assert b.unique_missing_signatures == 1
    assert details.loc[details.candidate_id == "B", "missing_source_signature"].iloc[0] == "MISSING"


def test_missing_cosubstrate_becomes_complete_when_independently_seeded() -> None:
    candidates = pd.DataFrame(
        {"query_id": ["Q"], "candidate_id": ["B"], "spectral_score": [0.7]}
    )
    participants = pd.DataFrame(
        {
            "reaction_id": ["R", "R", "R"],
            "side": ["left", "left", "right"],
            "compound_id": ["T", "M", "B"],
            "is_currency": [False, False, False],
            "reaction_weight": [1.0, 1.0, 1.0],
        }
    )
    seeds = pd.DataFrame(
        {
            "seed_query_id": ["TQ", "MQ"],
            "seed_compound_id": ["T", "M"],
            "seed_score": [0.9, 0.9],
        }
    )
    paths = pd.DataFrame(
        {
            "candidate_id": ["B"],
            "seed_compound_id": ["T"],
            "seed_query_id": ["TQ"],
            "reaction_id": ["R"],
            "seed_side": ["left"],
            "contribution": [0.6],
        }
    )
    features, _ = extract_reaction_context_features(
        candidates, paths, participants, seeds
    )
    assert features.iloc[0].complete_path_count == 1


def test_shared_missing_cosubstrate_is_counted_once() -> None:
    candidates = pd.DataFrame(
        {"query_id": ["Q"], "candidate_id": ["B"], "spectral_score": [0.7]}
    )
    participants = pd.DataFrame(
        {
            "reaction_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
            "side": ["left", "left", "right", "left", "left", "right"],
            "compound_id": ["S1", "M", "B", "S2", "M", "B"],
            "is_currency": [False] * 6,
            "reaction_weight": [1.0] * 6,
        }
    )
    seeds = pd.DataFrame(
        {
            "seed_query_id": ["S1Q", "S2Q"],
            "seed_compound_id": ["S1", "S2"],
            "seed_score": [0.9, 0.9],
        }
    )
    paths = pd.DataFrame(
        {
            "candidate_id": ["B", "B"],
            "seed_compound_id": ["S1", "S2"],
            "seed_query_id": ["S1Q", "S2Q"],
            "reaction_id": ["R1", "R2"],
            "seed_side": ["left", "left"],
            "contribution": [0.4, 0.3],
        }
    )
    features, _ = extract_reaction_context_features(
        candidates, paths, participants, seeds
    )
    row = features.iloc[0]
    assert np.isclose(row.raw_network_support, 1 - (1 - 0.4) * (1 - 0.3))
    assert np.isclose(row.dependency_corrected_network_support, 0.4)
    assert row.dependency_group_count == 1
