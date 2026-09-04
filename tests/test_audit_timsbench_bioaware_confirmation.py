import pandas as pd

from tasks.audit_timsbench_bioaware_confirmation import adduct_sign, ik14


def test_ik14_normalizes_full_and_connectivity_keys():
    values = pd.Series(["abcdefghijklmN-AAAA-B", "ABCDEFGHIJKLMN", None])
    assert ik14(values) == {"ABCDEFGHIJKLMN"}


def test_adduct_sign_handles_charge_magnitude():
    assert adduct_sign("[2M+2Na]+2") == 1
    assert adduct_sign("[M-H]-") == -1
    assert adduct_sign(None) == 0
