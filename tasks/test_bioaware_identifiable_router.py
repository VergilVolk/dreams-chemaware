#!/usr/bin/env python
"""Contract tests for the BioAware biological-identifiability gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bioaware_identifiable_router import apply_identifiable_router
from develop_bioaware_rank_consensus_fusion import FAMILY_FEATURES, RAW_FAMILIES


def row(query: str, candidate: str, truth: str, spectral: float, *, reaction: float,
        structure: float, rt: float = 0.0, rules: float = 0.0) -> dict:
    result = {
        "query_id": query, "candidate_id": candidate,
        "truth_candidate_id": truth, "truth_formula": "C6H12O6",
        "spectral_score": spectral,
    }
    for columns in RAW_FAMILIES.values():
        for column in columns:
            result[column] = 0.0
    result["predicted_edge_best_bottleneck"] = reaction
    result["smn_best_bottleneck"] = structure
    result["rt_score"] = rt
    result["rule_jaccard_idf"] = rules
    return result


def main() -> None:
    # The fusion score favours B in both queries.  In q_unique, B is the
    # unique reaction-network winner and may replace A.  In q_tied, A and B
    # have identical biological evidence; RT alone must not break the tie.
    ledger = pd.DataFrame([
        row("q_unique", "A", "B", 0.80, reaction=0.0, structure=0.0),
        row("q_unique", "B", "B", 0.79, reaction=1.0, structure=0.0, rt=1.0),
        row("q_tied", "A", "A", 0.80, reaction=1.0, structure=1.0),
        row("q_tied", "B", "A", 0.79, reaction=1.0, structure=1.0, rt=1.0),
        # Equal spectral scores: deterministic display top is truth A, but the
        # frozen strict baseline must still count the query as wrong.
        row("q_spectral_tie", "A", "A", 0.80, reaction=0.0, structure=0.0),
        row("q_spectral_tie", "B", "A", 0.80, reaction=0.0, structure=0.0),
    ])
    weights = np.asarray([
        0.0 if name not in {"family_predicted_reaction", "family_retention_time"}
        else 0.2 for name in FAMILY_FEATURES
    ])
    result, _ = apply_identifiable_router(
        ledger, weights, maximum_spectral_margin=0.05,
        minimum_fusion_advantage=0.0, minimum_support_families=1,
    )
    result = result.set_index("query_id")
    assert bool(result.loc["q_unique", "intervene"])
    assert bool(result.loc["q_unique", "corrected"])
    assert result.loc["q_unique", "unique_biological_mechanisms"] == "reaction_network"
    assert not bool(result.loc["q_tied", "biologically_identifiable"])
    assert not bool(result.loc["q_tied", "intervene"])
    assert bool(result.loc["q_tied", "final_correct"])
    assert not bool(result.loc["q_spectral_tie", "baseline_correct"])
    assert not bool(result.loc["q_spectral_tie", "intervene"])
    assert not bool(result.loc["q_spectral_tie", "final_correct"])
    print("[test_bioaware_identifiable_router] PASS")


if __name__ == "__main__":
    main()
