import pandas as pd

from tasks.develop_bioaware_same_formula_network_expert_v3 import (
    FEATURES,
    filter_same_formula_candidates,
)


def candidate_row(query, candidate, truth, truth_formula, row):
    value = {
        "query_id": query,
        "candidate_id": candidate,
        "truth_candidate_id": truth,
        "truth_formula": truth_formula,
        "best_library_row": row,
        "unit_id": "unit",
    }
    value.update({feature: 0.1 for feature in FEATURES})
    return value


def test_same_formula_filter_keeps_only_valid_competing_groups():
    candidates = pd.DataFrame([
        candidate_row("q1", "truth", "truth", "C2H4O2", 0),
        candidate_row("q1", "isomer", "truth", "C2H4O2", 1),
        candidate_row("q1", "wrong_formula", "truth", "C2H4O2", 2),
        candidate_row("q2", "truth2", "truth2", "CH4O", 3),
        candidate_row("q2", "other", "truth2", "CH4O", 4),
    ])
    integrity = pd.DataFrame({
        "library_row": [0, 1, 2, 3, 4],
        "calculated_formula": ["C2H4O2", "C2H4O2", "C3H6O2", "CH4O", "CH4O"],
        "approved_m_h_reference": [True, True, True, True, False],
        "structure_identity_consistent": [True] * 5,
    })
    result, report = filter_same_formula_candidates(candidates, integrity)
    assert set(result["query_id"]) == {"q1"}
    assert set(result["candidate_id"]) == {"truth", "isomer"}
    assert report["same_formula_queries"] == 1


def test_mass_recovery_feature_is_not_in_v3_recipe():
    assert "known_mass_candidate_fraction" not in FEATURES
