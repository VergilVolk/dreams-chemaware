import math

from audit_ibd200_candidate_adduct_compatibility import compatibility_status


def test_compatibility_status_abstains_without_neutral_mass():
    status, delta = compatibility_status(float("nan"), 100.0, 10.0)
    assert status == "adduct_unresolved"
    assert math.isnan(delta)


def test_compatibility_status_uses_neutral_mass_not_precursor_mass():
    assert compatibility_status(100.0, 100.0005, 10.0)[0] == "explicit_neutral_compatible"
    assert compatibility_status(100.0, 101.0, 10.0)[0] == "explicit_neutral_incompatible"
