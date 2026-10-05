#!/usr/bin/env python
"""Fast unit checks for BioAware B39-M0 pure helpers."""
from __future__ import annotations

import math

import pandas as pd

from build_bioaware_b39_atomic_event_ledger import (
    add_candidate_specificity,
    hyperedge_features,
    source_semantics,
)


def main() -> None:
    assert source_semantics("BV2cell")["context_class"] == "S"
    assert source_semantics("ST001154_same_formula_10ppm")["context_class"] == "L"
    assert source_semantics("KGMN200STD_hidden_seed")["context_class"] == "H"
    assert not source_semantics("ST001154_same_formula_10ppm")["prospective_unknown"]

    lookup = {
        "100": {
            "identity_noop": False,
            "noncurrency_left": {"SEED": 1.0, "CO_SUB": 2.0},
            "noncurrency_right": {"CAND": 1.0, "CO_PROD": 1.0},
        }
    }
    complete = hyperedge_features(
        "100", "SEED", "CAND", "left", "right", {"SEED", "CO_SUB", "CO_PROD"}, lookup
    )
    assert complete["required_context_participant_count"] == 2
    assert complete["observed_context_participant_count"] == 2
    assert complete["hyperedge_context_complete"]
    incomplete = hyperedge_features(
        "100", "SEED", "CAND", "left", "right", {"SEED", "CO_SUB"}, lookup
    )
    assert incomplete["hyperedge_context_completeness"] == 0.5
    assert incomplete["missing_context_signature"] == "CO_PROD"
    missing = hyperedge_features(
        "", "SEED", "CAND", "", "", {"SEED"}, lookup
    )
    assert math.isnan(missing["hyperedge_context_completeness"])

    events = pd.DataFrame(
        [
            {"source": "S", "query_id": "q", "seed_stratum": "r", "seed_identity": "s", "edge_key": "e", "candidate_id": "a"},
            {"source": "S", "query_id": "q", "seed_stratum": "r", "seed_identity": "s", "edge_key": "e", "candidate_id": "b"},
            {"source": "S", "query_id": "q", "seed_stratum": "r", "seed_identity": "s2", "edge_key": "e2", "candidate_id": "a"},
        ]
    )
    scored = add_candidate_specificity(events)
    assert scored.loc[scored["edge_key"].eq("e"), "event_supported_candidate_count"].eq(2).all()
    assert scored.loc[scored["edge_key"].eq("e2"), "event_candidate_specific"].all()
    print("[test_bioaware_b39_atomic_event_ledger] PASS", flush=True)


if __name__ == "__main__":
    main()
