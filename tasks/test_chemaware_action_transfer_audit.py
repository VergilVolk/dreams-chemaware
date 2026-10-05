"""CPU contracts for action/control reconstruction and late-stage summary."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_action_transfer_gradients import (  # noqa: E402
    ARMS,
    ROUTES,
    formula_stratified_positions,
    permuted_prediction_row,
    summarize,
    swapped_prediction_row,
)
from audit_chemaware_iceberg_synthetic_embedding import peak_permute  # noqa: E402


def test_memory_bounded_permutation_exactly_matches_action_bank_implementation() -> None:
    prediction = np.asarray([
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 0.0, 5.0, 6.0],
        [7.0, 8.0, 0.0, 9.0],
    ], dtype=np.float32)
    seed = 20260976
    expected = peak_permute(prediction, seed)
    observed = np.stack([
        permuted_prediction_row(prediction, index, seed)
        for index in range(len(prediction))
    ])
    assert np.array_equal(observed, expected)


def test_memory_bounded_candidate_swap_exactly_matches_np_roll() -> None:
    prediction = np.arange(24, dtype=np.float32).reshape(6, 4)
    expected = np.roll(prediction[1:5], 1, axis=0)
    observed = np.stack([
        swapped_prediction_row(prediction, index, 1, 5) for index in range(1, 5)
    ])
    assert np.array_equal(observed, expected)


def test_formula_stratified_screen_prioritizes_distinct_formulas() -> None:
    eligible = np.arange(8)
    formula = np.asarray(["A", "A", "B", "C", "D", "E", "F", "G"])
    selected = formula_stratified_positions(eligible, formula, maximum=6, seed=7)
    assert len(selected) == 6
    assert len(np.unique(formula[selected])) == 6
    assert set(selected).issubset(set(eligible))


def test_summary_passes_only_specific_positive_route_and_handles_zero_norm() -> None:
    records = []
    for position in range(20):
        for route in ROUTES:
            for arm in ARMS:
                if route == "action_forward_clean_backward":
                    value = {"correct": 1.0, "candidate_swapped": -0.5, "peak_permuted": -0.25}[arm]
                elif route == "naive_action_minus_clean" and position == 0:
                    value = None
                else:
                    value = -1.0
                records.append({
                    "action_position": position,
                    "formula": f"F{position}",
                    "route": route,
                    "arm": arm,
                    "clean_margin_gain_per_unit_update_norm": value,
                })
    report = summarize(records, draws=500, seed=11, minimum_fraction=0.60)
    assert report["action_forward_clean_backward"]["pass"] is True
    assert report["action_query_only"]["pass"] is False
    assert report["action_routed_clean_pair"]["pass"] is False
    assert report["naive_action_minus_clean"]["pass"] is False
    assert report["naive_action_minus_clean"]["finite_fraction"] < 1.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware action-transfer audit contracts")
