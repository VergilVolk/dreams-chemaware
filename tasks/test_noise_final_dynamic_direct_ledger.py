"""Synthetic and static tests for the unified dynamic N/P ledger."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))


def main() -> None:
    path = ROOT / "tasks/build_noise_final_dynamic_direct_ledger.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "P_INTENSITY_FAMILIES", "P_TRANSFER_FAMILIES", "fit_crossfit",
        "all_30_cells_retained", "outer_held_formulas_absent",
        "raw_P_outcomes_published", "historical transfer matrix",
        "stable_control_index", "build_action_weights",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"dynamic ledger contract drifted: {missing}")
    forbidden = ("passing_cells", "best_fixed_cell", "oracle_per_query", "P2b_score")
    present = [token for token in forbidden if token in source]
    if present:
        raise RuntimeError(f"dynamic ledger performs forbidden post-outcome cell selection: {present}")

    # Exercise formula-crossfit behavior in the server environment before any
    # model is loaded. Every held prediction must come from other formulas.
    try:
        from build_noise_final_dynamic_direct_ledger import fit_crossfit
    except ModuleNotFoundError as error:
        if error.name in {"h5py", "sklearn"}:
            print(
                "[test_noise_final_dynamic_direct_ledger] static PASS; "
                f"{error.name} numeric test deferred"
            )
            return
        raise
    rng = np.random.default_rng(19)
    formulas = np.repeat([f"f{i}" for i in range(80)], 4)
    folds = np.repeat(np.arange(80) % 5, 4).astype(np.int8)
    x = rng.normal(size=(len(formulas), 8)).astype(np.float32)
    signal = x[:, 0] - 0.5 * x[:, 1]
    gain = (0.02 * signal + rng.normal(scale=0.004, size=len(signal))).astype(np.float32)
    positive = gain >= 0.01
    harmful = gain <= -0.01
    pred_gain, p_positive, p_harmful, metrics = fit_crossfit(
        x, formulas, folds, gain, positive, harmful, outer_fold=4, seed=29,
    )
    active = folds != 4
    if not np.isfinite(np.column_stack([pred_gain[active], p_positive[active], p_harmful[active]])).all():
        raise RuntimeError("synthetic crossfit produced missing predictions")
    if metrics["positive_auprc"] <= metrics["positive_prevalence"]:
        raise RuntimeError("synthetic crossfit failed to recover a strong clean-visible signal")
    if np.isfinite(pred_gain[~active]).any():
        raise RuntimeError("outer-held formulas received action predictions")
    print("[test_noise_final_dynamic_direct_ledger] PASS")


if __name__ == "__main__":
    main()
