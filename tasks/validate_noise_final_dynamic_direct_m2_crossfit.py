"""Independent validation of current-geometry formula-OOF action weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [args.output_dir / "report.json", args.output_dir / "training_actions.csv.gz"]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    frame = pd.read_csv(required[1], low_memory=False)
    forbidden = {"clean_rank", "target_rank", "control_rank", "corrected", "introduced", "positive", "harmful"}
    grouped = frame.groupby("query_index", sort=False)
    gates = {
        "status": report.get("status") == "noise_final_dynamic_direct_m2_current_crossfit_complete",
        "formal": report.get("formal") is True,
        "rows": len(frame) == int(report.get("actions", -1)),
        "unique": not frame["action_id"].duplicated().any(),
        "sources": set(frame["source"]) == {"N", "P_intensity", "P_transfer"},
        "cells": frame["cell_id"].nunique() == 30,
        "probabilities": frame[["p_clean", "risk"]].ge(0).all().all() and frame[["p_clean", "risk"]].le(1).all().all(),
        "finite": np.isfinite(frame[["p_clean", "risk", "lagged_advantage", "dynamic_weight"]].to_numpy(float)).all(),
        "no_raw_outcomes": not bool(forbidden & set(frame.columns)),
        "dynamic_mass": np.allclose(grouped["dynamic_weight"].sum() + grouped["dynamic_no_op_weight"].first(), 1.0, atol=2e-6),
        "static_mass": np.allclose(grouped["static_weight"].sum() + grouped["static_no_op_weight"].first(), 1.0, atol=2e-6),
        "positive_no_op": grouped["dynamic_no_op_weight"].first().gt(0).all(),
        "ablations": all(
            {"full", "cell_only", "permuted_clean"} <= set(value.get("models", {}))
            for value in report.get("source_crossfit", {}).values()
        ),
        "P2b": report.get("contracts", {}).get("P2b") == "forbidden",
        "P3": report.get("contracts", {}).get("P3_consumed") is False,
        "pass": report.get("pass_to_schedule") is True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"M2 current-crossfit validation failed: {gates}")
    print(f"[validate_noise_final_dynamic_direct_m2_crossfit] PASS actions={len(frame):,}")


if __name__ == "__main__":
    main()
