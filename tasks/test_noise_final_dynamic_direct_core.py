"""Numerical and contract tests for dynamic conditional direct training."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_dynamic_direct_core import (  # noqa: E402
    N_CELLS, PHASE_A_ARMS, WeightConfig, assert_outer_formula_disjoint,
    build_action_weights, formula_equal_weights, stable_control_index,
    stratified_action_epoch, validate_n_cells,
)


def synthetic_actions() -> pd.DataFrame:
    rows = []
    action = 0
    for formula_index in range(8):
        for identity_index in range(4):
            query = formula_index * 4 + identity_index
            for family_index, family in enumerate(("candidate_gradient", "role_confounder", "p_recurrent")):
                rows.append({
                    "action_id": f"a{action}", "query_index": query,
                    "identity": f"i{formula_index}_{identity_index}",
                    "formula": f"f{formula_index}", "family": family,
                    "p_clean": 0.2 + 0.25 * family_index + 0.02 * (query % 4),
                    "lagged_advantage": -0.02 + 0.02 * family_index + 0.004 * ((query % 3) - 1),
                    "risk": 0.05 * family_index,
                })
                action += 1
    return pd.DataFrame(rows)


def main() -> None:
    source = (ROOT / "tasks/noise_final_dynamic_direct_core.py").read_text(encoding="utf-8")
    ast.parse(source)
    if PHASE_A_ARMS != ("clean_continuation", "matched_random", "static_target", "dynamic_np"):
        raise RuntimeError("Phase A arm contract drifted")
    frame = synthetic_actions()
    config = WeightConfig(minimum_family_ess=4.0)
    dynamic, dynamic_report = build_action_weights(frame, "dynamic", config)
    static, static_report = build_action_weights(frame, "static", config)
    if np.allclose(dynamic["weight"], static["weight"]):
        raise RuntimeError("dynamic weights collapsed to the static control")
    if not dynamic_report["all_family_ess_pass"] or not static_report["all_family_ess_pass"]:
        raise RuntimeError("synthetic family ESS unexpectedly failed")
    if dynamic.groupby("query_index")["weight"].sum().max() > 1.000002:
        raise RuntimeError("query exposure cap failed")
    candidate = dynamic.loc[dynamic["family"].eq("candidate_gradient")].copy()
    raw_order_signal = candidate["raw_utility"].to_numpy()
    if np.corrcoef(raw_order_signal, candidate["weight"].to_numpy())[0, 1] <= 0.5:
        raise RuntimeError("dynamic evidence lost all positive association after balancing")
    choices = [stable_control_index(f"action-{index}") for index in range(100)]
    if set(choices) != {0, 1} or choices != [stable_control_index(f"action-{index}") for index in range(100)]:
        raise RuntimeError("matched-control selection is not frozen and deterministic")
    formula_weights = formula_equal_weights(["a", "a", "b", "c", "c", "c"])
    totals = pd.DataFrame({"formula": ["a", "a", "b", "c", "c", "c"],
                           "weight": formula_weights}).groupby("formula")["weight"].sum()
    if not np.allclose(totals, totals.iloc[0]):
        raise RuntimeError("formula-equal weights are not equal")
    sampled = stratified_action_epoch(dynamic, "weight", 17, 1)
    if sampled["action_id"].duplicated().any():
        raise RuntimeError("stratified sampler recycled an action")
    per_identity_family = sampled.groupby(["identity", "family"]).size()
    if int(per_identity_family.max()) > 1:
        raise RuntimeError("stratified sampler violated its exposure cap")

    n_rows = pd.DataFrame([
        {"selector": selector, "attenuation": attenuation, "step": step}
        for selector, attenuation, step in sorted(N_CELLS)
    ])
    validate_n_cells(n_rows)
    split = pd.DataFrame({
        "formula": ["f0", "f1", "f2"], "formula_fold": [0, 1, 1],
        "split": ["held", "train", "train"],
    })
    assert_outer_formula_disjoint(split, 0)

    # Fail-closed regression checks.
    duplicate = frame.copy()
    duplicate.loc[1, "action_id"] = duplicate.loc[0, "action_id"]
    try:
        build_action_weights(duplicate, "dynamic", config)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("duplicate action ids were accepted")
    leaked = split.copy(); leaked.loc[len(leaked)] = ["f0", 1, "train"]
    try:
        assert_outer_formula_disjoint(leaked, 0)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("formula overlap was accepted")
    print("[test_noise_final_dynamic_direct_core] PASS")


if __name__ == "__main__":
    main()
