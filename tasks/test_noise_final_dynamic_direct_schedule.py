"""Static and numerical tests for the common Phase-A schedule."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from noise_final_dynamic_direct_core import stratified_action_epoch, stratified_action_schedule
from build_noise_final_dynamic_direct_schedule import assign_epoch_weights, json_default


def main() -> None:
    for filename in (
        "build_noise_final_dynamic_direct_schedule.py",
        "validate_noise_final_dynamic_direct_schedule.py",
    ):
        ast.parse((ROOT / "tasks" / filename).read_text(encoding="utf-8"))
    rows = []
    for query in range(24):
        for family in ("N:a", "N:b", "P:c"):
            for action in range(3):
                rows.append({
                    "action_id": f"{query}|{family}|{action}", "query_index": query,
                    "identity": f"i{query // 2}", "formula": f"f{query // 4}",
                    "family": family, "cell_id": f"{family}|{action}",
                    "dynamic_weight": 0.1 + 0.01 * action,
                })
    frame = pd.DataFrame(rows)
    first = stratified_action_epoch(frame, "dynamic_weight", 29, 1)
    second = stratified_action_epoch(frame, "dynamic_weight", 29, 1)
    if first["action_id"].tolist() != second["action_id"].tolist():
        raise RuntimeError("common schedule is not deterministic")
    if first["action_id"].duplicated().any():
        raise RuntimeError("schedule recycled an action")
    if int(first.groupby(["identity", "family"]).size().max()) != 1:
        raise RuntimeError("identity-family cap failed")
    if len(first) != 12 * 3:
        raise RuntimeError("unexpected bounded schedule size")
    if not np.isfinite(first["dynamic_weight"]).all():
        raise RuntimeError("schedule contains invalid weights")
    cycling = stratified_action_schedule(frame, 29, 3, 1)
    if cycling.duplicated(["epoch", "action_id"]).any():
        raise RuntimeError("cycling schedule repeats an action within an epoch")
    if int(cycling.groupby(["epoch", "identity", "family"]).size().max()) != 1:
        raise RuntimeError("cycling identity-family cap failed")
    for (_, _), block in cycling.groupby(["identity", "family"]):
        if block["action_id"].nunique() != 3:
            raise RuntimeError("cycling schedule failed to traverse available actions")
    weighted_input = cycling.loc[cycling["epoch"].eq(1)].rename(
        columns={"dynamic_weight": "m2_dynamic_weight"}
    )
    full_mass = frame.groupby("query_index")["dynamic_weight"].sum()
    weighted, report = assign_epoch_weights(weighted_input, full_mass, 0.45, 0.65)
    dynamic_mass = weighted.groupby("query_index")["dynamic_weight"].sum().sort_index()
    static_mass = weighted.groupby("query_index")["static_weight"].sum().sort_index()
    if not np.allclose(dynamic_mass, static_mass, atol=1e-8):
        raise RuntimeError("static schedule does not match conditional query dose")
    if abs(float(dynamic_mass.mean()) - 0.45) > 1e-8:
        raise RuntimeError("epoch curriculum mean dose is incorrect")
    if weighted[["dynamic_selected_no_op_weight", "static_selected_no_op_weight"]].min().min() < 0.35 - 1e-8:
        raise RuntimeError("minimum no-op mass was violated")
    if float(report["static_dynamic_mass_max_abs_difference"]) > 1e-8:
        raise RuntimeError("reported dose mismatch is nonzero")
    serialized = json.loads(json.dumps({
        "bool": np.bool_(True),
        "integer": np.int64(7),
        "float": np.float32(0.25),
    }, default=json_default))
    if serialized != {"bool": True, "integer": 7, "float": 0.25}:
        raise RuntimeError("NumPy report scalar JSON normalization failed")
    try:
        json.dumps({"unsupported": object()}, default=json_default)
    except TypeError:
        pass
    else:
        raise RuntimeError("JSON fallback silently accepted an unsupported object")
    print("[test_noise_final_dynamic_direct_schedule] PASS")


if __name__ == "__main__":
    main()
