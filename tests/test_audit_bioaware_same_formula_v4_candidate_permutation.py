from tasks.develop_bioaware_same_formula_network_expert_v3 import FEATURES


def test_v4_permutation_has_network_features_to_destroy():
    assert [feature for feature in FEATURES if feature != "spectral_score"]
    assert "known_mass_candidate_fraction" not in FEATURES
