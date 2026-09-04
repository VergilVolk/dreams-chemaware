from __future__ import annotations

import pandas as pd
import pytest

from tasks.audit_kgmn_oep003284_author_identifiers import reconcile


def raw_tables() -> dict[str, pd.DataFrame]:
    return {
        "positive": pd.DataFrame({"peak_name": ["M100T10"], "raw_mz": [100.0], "raw_rt": [10.0]}),
        "negative": pd.DataFrame({"peak_name": ["M200T20"], "raw_mz": [200.0], "raw_rt": [20.0]}),
    }


def test_reconcile_accepts_author_rounding_only() -> None:
    truth = pd.DataFrame({
        "polarity": ["positive", "negative"], "peak_name": ["M100T10", "M200T20"],
        "mz": [100.00005, 199.99995], "rt": [10.05, 19.95],
    })
    result = reconcile(truth, raw_tables(), label="test", mz_tolerance_da=1e-4, rt_tolerance_sec=0.1)
    assert len(result) == 2
    assert result["abs_mz_delta_da"].max() <= 1e-4


def test_reconcile_rejects_missing_or_coordinate_mismatch() -> None:
    missing = pd.DataFrame({"polarity": ["positive"], "peak_name": ["missing"], "mz": [100.0], "rt": [10.0]})
    with pytest.raises(RuntimeError, match="missing from author peak tables"):
        reconcile(missing, raw_tables(), label="test", mz_tolerance_da=1e-4, rt_tolerance_sec=0.1)
    mismatch = pd.DataFrame({"polarity": ["positive"], "peak_name": ["M100T10"], "mz": [100.01], "rt": [10.0]})
    with pytest.raises(RuntimeError, match="author-coordinate mismatch"):
        reconcile(mismatch, raw_tables(), label="test", mz_tolerance_da=1e-4, rt_tolerance_sec=0.1)
