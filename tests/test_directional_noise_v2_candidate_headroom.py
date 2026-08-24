import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_directional_noise_v2_candidate_headroom",
    ROOT / "tasks/run_directional_noise_v2_candidate_headroom.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_best_of_k_compares_equal_target_and_random_search_multiplicity():
    base = {
        "query_row": 1, "positive_row": 2, "negative_row": 3,
        "ik14": "A", "formula": "F", "adduct": "M+H",
        "cross_condition_positive": True, "baseline_margin": -0.1,
        "candidate_count": 2,
    }
    rows = [
        base | {"condition": "targeted", "repeat": -1, "target_slot": 0, "target_token": 4, "target_mz": 100.0, "perturbed_margin": -0.05},
        base | {"condition": "targeted", "repeat": -1, "target_slot": 1, "target_token": 5, "target_mz": 120.0, "perturbed_margin": 0.10},
    ]
    random_margins = [[-0.04, -0.02], [-0.03, 0.02], [-0.01, -0.05]]
    for repeat, margins in enumerate(random_margins):
        for slot, margin in enumerate(margins):
            rows.append(base | {
                "condition": "matched_random", "repeat": repeat,
                "target_slot": slot, "target_token": 10 + slot,
                "target_mz": 200.0 + slot, "perturbed_margin": margin,
            })
    paired = MODULE.aggregate_best_of_k(pd.DataFrame(rows), random_repeats=3)
    assert len(paired) == 1
    result = paired.iloc[0]
    assert result["best_target_slot"] == 1
    assert result["target_top1"] == 1
    assert abs(result["mean_best_random_margin"] - (-1 / 300)) < 1e-12
    assert abs(result["random_top1_mean"] - (1 / 3)) < 1e-12
