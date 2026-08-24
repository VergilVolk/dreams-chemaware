import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_g8r_causal_peak_specificity",
    ROOT / "tasks/analyze_g8r_causal_peak_specificity.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_control_matching_removes_generic_geometry_effect():
    rows = []
    for transition, offset in (("protected_correct", 0.0), ("persistent_wrong", 0.2)):
        for i in range(30):
            rows.append({
                "arm": "positive_deficit",
                "direction": "query_to_reference",
                "transition": transition,
                "removed_count": 5,
                "clean_similarity": 0.2 + i / 100,
                "directional_support": 0.1 + offset,
                "query_ik14": f"{transition}-{i}",
                "query_formula": f"F{i}",
            })
    frame = MODULE.attach_control_expectation(MODULE.add_matching_strata(pd.DataFrame(rows)), 5)
    controls = frame[frame["transition"] == "protected_correct"]
    cases = frame[frame["transition"] == "persistent_wrong"]
    assert abs(float(controls["specific_excess_support"].mean())) < 1e-12
    assert abs(float(cases["specific_excess_support"].mean()) - 0.2) < 1e-12
