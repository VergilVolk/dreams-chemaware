from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_formula_crosscalibration_core import (  # noqa: E402
    crosscalibrate,
    ranks_at_threshold,
    reward,
    stable_formula_roles,
)


def main() -> None:
    baseline = np.asarray([2, 1, 2, 1, 2, 1, 2, 1], dtype=np.int16)
    proposal = np.tile(np.asarray([[1, 3]], dtype=np.int16), (8, 1))
    correct = np.asarray([[0.9, 0.1], [0.1, 0.8]] * 4, dtype=np.float64)
    control = np.asarray([[0.1, 0.8], [0.8, 0.1]] * 4, dtype=np.float64)
    formula = np.asarray([f"F{i}" for i in range(8)])
    roles = stable_formula_roles(formula, 2, 11)
    assert roles.shape == (8,) and np.array_equal(roles, stable_formula_roles(formula, 2, 11))
    rank, selected, _ = ranks_at_threshold(baseline, proposal, correct, 0.5)
    assert selected.shape == baseline.shape and rank.shape == baseline.shape
    assert reward(baseline, rank, 2.0).shape == baseline.shape
    result = crosscalibrate(
        baseline, proposal, {"correct": correct, "control": control}, formula,
        roles=2, seed=11, risk_penalty=2.0, min_selected_formulas=1,
        bootstrap_draws=200,
    )
    assert len(result["folds"]) == 2
    assert np.isfinite(result["deployment_threshold"])
    assert set(result["oof_gate"]) == {
        "correct_risk_ci_positive",
        "corrected_exceeds_risk_weighted_introduced",
        "correct_beats_control_risk_ci",
    }
    print("PASS: ChemAware formula cross-calibration contracts")


if __name__ == "__main__":
    main()
