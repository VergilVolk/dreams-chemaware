#!/usr/bin/env python
"""Deterministic tests for candidate-specific typed BioAware context fields."""
from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware_context import extract_reaction_context_features


def main() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["q", "q"],
            "candidate_id": ["c1", "c2"],
            "spectral_score": [0.5, 0.4],
        }
    )
    seeds = pd.DataFrame(
        {
            "seed_query_id": ["seed-row", "cofactor-row", "coproduct-row"],
            "seed_compound_id": ["s", "cofactor", "coproduct"],
            "seed_score": [0.9, 0.9, 0.9],
        }
    )
    participants = pd.DataFrame(
        {
            "reaction_id": ["r", "r", "r", "r", "noop", "noop"],
            "side": ["left", "left", "right", "right", "left", "right"],
            "compound_id": ["s", "cofactor", "c1", "coproduct", "s", "s"],
            "stoichiometry": [1, 2, 1, 1, 1, 1],
            "direction_semantics": ["physiological_lr"] * 6,
            "is_currency": [False] * 6,
        }
    )
    # The same seed/reaction supports two competing candidates.  c2 is not a
    # participant in the synthetic hyperedge, but represents an intentionally
    # ambiguous path cache entry for testing query-level specificity.
    paths = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c1"],
            "seed_compound_id": ["s", "s", "s"],
            "seed_query_id": ["seed-row", "seed-row", "seed-row"],
            "reaction_id": ["r", "r", "noop"],
            "seed_side": ["left", "left", "left"],
            "candidate_side": ["right", "right", "right"],
            "contribution": [0.8, 0.8, 0.95],
        }
    )
    reaction_directions = pd.DataFrame(
        {"MASTER_ID": ["r"], "DIRECTION": ["LR"]}
    )
    features, details = extract_reaction_context_features(
        candidates,
        paths,
        participants,
        seeds,
        reaction_directions=reaction_directions,
    )
    assert len(features) == 2
    assert set(details["competing_query_candidate_count"]) == {2}
    assert np.allclose(details["candidate_specificity"], 0.5)
    c1 = details[details["query_candidate_id"] == "c1"].iloc[0]
    assert bool(c1["source_side_complete"])
    assert bool(c1["target_side_complete"])
    assert bool(c1["physiological_direction_available"])
    assert bool(c1["curated_direction_supported"])
    assert not bool(c1["curated_direction_conflicted"])
    assert np.isclose(c1["source_side_noncurrency_stoichiometry"], 3.0)
    assert np.isclose(c1["specificity_weighted_contribution"], 0.4)
    c1_feature = features[features["candidate_id"] == "c1"].iloc[0]
    assert c1_feature["raw_path_count"] == 1
    assert c1_feature["excluded_identity_noop_path_count"] == 1
    assert "noop" not in set(details["reaction_id"].astype(str))
    print("[test_bioaware_typed_context] PASS", flush=True)


if __name__ == "__main__":
    main()
