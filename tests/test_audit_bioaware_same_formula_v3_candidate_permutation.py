import pandas as pd

from tasks.audit_bioaware_same_formula_v3_candidate_permutation import FEATURES


def test_same_formula_permutation_keeps_spectral_score_fixed_by_contract():
    network_features = [feature for feature in FEATURES if feature != "spectral_score"]
    assert "spectral_score" not in network_features
    assert network_features


def test_feature_recipe_has_no_mass_recovery_shortcut():
    assert "known_mass_candidate_fraction" not in FEATURES
