from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks" / "prepare_kgmn_oep003284_author_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_oep", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frame_for(polarity: str) -> pd.DataFrame:
    meta = MODULE.PANELS[polarity]
    samples = MODULE.expected_samples()
    count = meta["rows"]
    values = {
        "name": [f"M{i}T{i}" for i in range(count)],
        "mz": np.arange(count, dtype=float) + 70.0,
        "rt": np.arange(count, dtype=float),
    }
    values.update({sample: np.ones(count) for sample in samples})
    return pd.DataFrame(values)


def test_published_peak_table_contract_is_exact() -> None:
    frame = frame_for("positive")
    exported, sample_info = MODULE.validate_peak_table(frame, "positive")
    assert len(exported) == 15942
    assert list(sample_info.columns) == ["sample.name", "group"]
    assert sample_info["group"].value_counts().to_dict() == {"g1": 4, "g2": 4, "g4": 4}


def test_feature_name_or_sample_drift_fails() -> None:
    duplicated = frame_for("negative")
    duplicated.loc[1, "name"] = duplicated.loc[0, "name"]
    with pytest.raises(RuntimeError, match="empty or duplicated"):
        MODULE.validate_peak_table(duplicated, "negative")
    wrong_sample = frame_for("positive").rename(columns={"g1_46std_1": "wrong"})
    with pytest.raises(RuntimeError, match="sample columns drift"):
        MODULE.validate_peak_table(wrong_sample, "positive")
