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
    formulas = np.repeat([f"f{i}" for i in range(125)], 4)
    folds = np.repeat(np.arange(125) % 5, 4).astype(np.int8)
    x = rng.normal(size=(len(formulas), 8)).astype(np.float32)
    signal = x[:, 0] - 0.5 * x[:, 1]
    gain = (0.03 * signal + rng.normal(scale=0.002, size=len(signal))).astype(np.float32)
    positive = signal >= 0.25
    harmful = signal <= -0.25
    # Exercise every possible outer fold. Production support thresholds remain
    # the defaults; only the tree leaf size is reduced for this compact fixture.
    for outer_fold in range(5):
        pred_gain, p_positive, p_harmful, metrics = fit_crossfit(
            x, formulas, folds, gain, positive, harmful,
            outer_fold=outer_fold, seed=29, min_samples_leaf=10,
        )
        active = folds != outer_fold
        values = np.column_stack([pred_gain[active], p_positive[active], p_harmful[active]])
        if not np.isfinite(values).all():
            raise RuntimeError(f"synthetic crossfit produced missing predictions for outer fold {outer_fold}")
        if metrics["positive_auprc"] <= metrics["positive_prevalence"]:
            raise RuntimeError(f"synthetic crossfit missed clean-visible signal for outer fold {outer_fold}")
        if np.isfinite(pred_gain[~active]).any():
            raise RuntimeError(f"outer-held formulas received predictions for outer fold {outer_fold}")

    # Contract failures must be explicit before sklearn is entered.
    try:
        fit_crossfit(
            x[:20], formulas[:20], folds[:20], gain[:20], positive[:20], harmful[:20],
            outer_fold=4, seed=29, minimum_train_formulas=50, min_samples_leaf=2,
        )
    except RuntimeError as error:
        if "insufficient formula support" not in str(error):
            raise
    else:
        raise RuntimeError("undersized crossfit fixture was not rejected")
    print("[test_noise_final_dynamic_direct_ledger] PASS")


if __name__ == "__main__":
    main()
