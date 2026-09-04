from tasks.develop_bioaware_same_formula_network_expert_v3 import FEATURES


def test_v3_frozen_recipe_is_same_formula_compatible():
    assert "spectral_score" in FEATURES
    assert "known_mass_candidate_fraction" not in FEATURES
    assert len(FEATURES) == len(set(FEATURES))
