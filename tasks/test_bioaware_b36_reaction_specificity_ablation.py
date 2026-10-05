#!/usr/bin/env python
"""Unit checks for B36 null construction and gate safety."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b36_reaction_specificity_ablation import (  # noqa: E402
    CATALOG_FEATURES,
    REACTION_FEATURES,
    REAL_ARMS,
    choose_gate,
    derange_reaction_context,
)


def fixture() -> pd.DataFrame:
    rows = []
    for query_index, query in enumerate(("q0", "q1", "q2")):
        for candidate_index in range(3):
            row = {
                "query_id": query,
                "source": "s",
                "network_member": 1.0,
                "known_log_degree": 2.2,
                "known_mass_candidate_fraction": 0.4,
                "log_reference_spectra": 1.5,
                "truth_marker": f"truth-{query}",
            }
            for feature_index, feature in enumerate(REACTION_FEATURES):
                row[feature] = float(100 * query_index + 10 * candidate_index + feature_index)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    frame = fixture()
    for method in ("within_query", "catalog_stratum"):
        left, audit = derange_reaction_context(frame, method, 17)
        right, _ = derange_reaction_context(frame, method, 17)
        assert left[REACTION_FEATURES].equals(right[REACTION_FEATURES])
        assert left["truth_marker"].equals(frame["truth_marker"])
        assert audit["assigned_from_other_row"] > 0
        assert audit["maximum_group_feature_sum_error"] <= 1e-9
        assert not np.array_equal(
            left[REACTION_FEATURES].to_numpy(), frame[REACTION_FEATURES].to_numpy()
        )
    assert not set(CATALOG_FEATURES) & set(REACTION_FEATURES)
    assert set(REAL_ARMS) == {
        "spectral_only", "spectral_plus_catalog", "spectral_plus_reaction",
        "spectral_plus_catalog_plus_reaction",
    }
    scored = pd.DataFrame({
        "query_id": ["q0", "q1"],
        "source": ["s0", "s1"],
        "truth_candidate_id": ["t0", "t1"],
        "truth_formula": ["f0", "f1"],
        "baseline_candidate_id": ["t0", "t1"],
        "proposed_candidate_id": ["x0", "x1"],
        "baseline_correct": [True, True],
        "proposal_unique": [True, True],
        "proposal_probability": [0.99, 0.99],
        "baseline_gap": [0.01, 0.01],
        "candidate_count": [2, 2],
    })
    selected, ledger = choose_gate(scored)
    assert selected["gate_name"] == "no_op"
    assert len(ledger) == 16
    print("[test_bioaware_b36_reaction_specificity_ablation] PASS")


if __name__ == "__main__":
    main()
