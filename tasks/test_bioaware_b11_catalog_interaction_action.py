#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES,
    GATE_GRID,
    apply_gate,
    choose_configuration,
)


def scored(recipe_offset: float) -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4"],
        "held_label": ["u"] * 4,
        "source": ["s1", "s1", "s2", "s2"],
        "truth_candidate_id": ["T1", "T2", "T3", "T4"],
        "truth_formula": ["F1", "F2", "F3", "F4"],
        "baseline_candidate_id": ["W1", "T2", "W3", "T4"],
        "proposed_candidate_id": ["T1", "W2", "T3", "T4"],
        "baseline_correct": [False, True, False, True],
        "proposal_unique": [True, True, True, True],
        "proposal_probability": [0.80 + recipe_offset, 0.60, 0.80, 0.90],
        "baseline_gap": [0.03, 0.03, 0.03, 0.01],
    })


def main() -> None:
    assert set(FEATURE_RECIPES) == {
        "linear_b4_replay",
        "spectral_catalog_interactions",
        "ambiguity_density_interactions",
    }
    assert len(GATE_GRID) == 9
    result = apply_gate(scored(0.0), 0.05, 0.75)
    assert int(result["corrected"].sum()) == 2
    assert int(result["introduced"].sum()) == 0
    assert int(result["intervene"].sum()) == 2
    tables = {
        "linear_b4_replay": scored(-0.10),
        "spectral_catalog_interactions": scored(0.00),
        "ambiguity_density_interactions": scored(-0.05),
    }
    selected, ledger = choose_configuration(tables)
    # Equal safe objectives deliberately fall back to the simpler exact B4
    # replay; an interaction recipe must earn a strictly better inner score.
    assert selected["recipe"] == "linear_b4_replay"
    assert selected["risk_net_lambda2"] == 2
    assert len(ledger) == 27
    print("[BioAware B11 catalog-interaction unit checks] PASS", flush=True)


if __name__ == "__main__":
    main()
