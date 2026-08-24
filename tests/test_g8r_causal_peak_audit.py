import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_g8r_causal_peak_audit", ROOT / "tasks/run_g8r_causal_peak_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_cases_keeps_all_nonprotected_and_caps_controls():
    frame = pd.DataFrame({
        "query_index": range(7),
        "query_row": range(100, 107),
        "transition": [
            "corrected", "introduced", "persistent_wrong",
            "protected_correct", "protected_correct", "protected_correct", "protected_correct",
        ],
    })
    selected = MODULE.select_cases(frame, protected_cap=2, seed=7)
    assert set(selected.loc[selected["transition"] != "protected_correct", "query_index"]) == {0, 1, 2}
    assert int((selected["transition"] == "protected_correct").sum()) == 2
