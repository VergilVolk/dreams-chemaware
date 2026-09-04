from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.audit_bioaware_metdna3_negative_loso_ablation import (
    ABLATIONS,
    paired_formula_bootstrap,
)


def test_ablation_feature_families_are_distinct_and_full_is_complete() -> None:
    assert ABLATIONS["spectral_only_same_edge_gate"]["features"] == ["spectral_score"]
    assert "spectral_score" not in ABLATIONS["network_only_same_edge_gate"]["features"]
    assert set(ABLATIONS["full_bioaware"]["features"]) > set(
        ABLATIONS["spectral_only_same_edge_gate"]["features"]
    )
    assert ABLATIONS["full_bioaware"]["require_raw_step0_edge"] is True
    assert ABLATIONS["mass_membership_only_no_edge_gate"]["features"] == [
        "known_mass_candidate_fraction"
    ]
    assert "known_mass_candidate_fraction" not in ABLATIONS[
        "known_topology_without_mass_same_edge_gate"
    ]["features"]


def test_paired_formula_bootstrap_uses_query_paired_difference() -> None:
    left = pd.DataFrame({
        "query_id": [1, 2, 3], "truth_formula": ["A", "A", "B"],
        "final_correct": [True, True, False],
    })
    right = pd.DataFrame({
        "query_id": [1, 2, 3], "final_correct": [False, True, False],
    })
    result = paired_formula_bootstrap(left, right, repeats=100, seed=7)
    assert np.isclose(result["mean"], 1 / 3)
    assert result["left_better"] == 1
    assert result["right_better"] == 0
