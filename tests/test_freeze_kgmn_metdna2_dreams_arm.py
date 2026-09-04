from __future__ import annotations

from pathlib import Path

import pandas as pd

from tasks.freeze_kgmn_metdna2_dreams_arm import compare_csv_multisets


def test_compare_csv_multisets_ignores_row_order(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"id": ["a", "b"], "score": [0.1, 0.2]}).to_csv(left, index=False)
    pd.DataFrame({"id": ["b", "a"], "score": [0.2, 0.1]}).to_csv(right, index=False)
    equal, detail = compare_csv_multisets(left, right)
    assert equal
    assert detail is None


def test_compare_csv_multisets_detects_score_change(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"id": ["a"], "score": [0.1]}).to_csv(left, index=False)
    pd.DataFrame({"id": ["a"], "score": [0.11]}).to_csv(right, index=False)
    equal, detail = compare_csv_multisets(left, right)
    assert not equal
    assert detail
