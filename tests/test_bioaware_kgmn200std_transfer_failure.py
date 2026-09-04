import pandas as pd

from tasks.diagnose_bioaware_kgmn200std_transfer_failure import transition_counts


def test_transition_counts_are_paired() -> None:
    baseline = pd.Series([False, True, False, True])
    proposal = pd.Series([True, False, False, True])
    result = transition_counts(baseline, proposal)
    assert result == {
        "corrected": 1, "introduced": 1, "net": 0, "risk_weighted_net": -1,
    }
