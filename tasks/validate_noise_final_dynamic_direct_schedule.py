"""Independent validation of the Phase-A arm-invariant schedule."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from noise_final_dynamic_direct_core import PHASE_A_ARMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [args.output_dir / name for name in ("report.json", "epoch_schedule.csv.gz", "arm_manifest.csv")]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    schedule = pd.read_csv(required[1], low_memory=False)
    arms = pd.read_csv(required[2])
    gates = {
        "status": report.get("status") == "noise_final_dynamic_direct_phase_a_schedule_complete",
        "formal": report.get("formal") is True,
        "rows": len(schedule) == int(report.get("scheduled_actions", -1)),
        "order": schedule["schedule_index"].tolist() == list(range(len(schedule))),
        "unique_within_epoch": not schedule.duplicated(["epoch", "action_id"]).any(),
        "arms": set(arms["arm"]) == set(PHASE_A_ARMS),
        "identical_membership": arms["membership_sha256"].nunique() == 1,
        "cells": schedule["cell_id"].nunique() == 30,
        "cells_each_epoch": schedule.groupby("epoch")["cell_id"].nunique().eq(30).all(),
        "no_op": schedule[["dynamic_selected_no_op_weight", "static_selected_no_op_weight"]].gt(0).all().all(),
        "dynamic_mass": np.allclose(
            schedule.groupby(["epoch", "query_index"])["dynamic_weight"].sum().to_numpy()
            + schedule.groupby(["epoch", "query_index"])["dynamic_selected_no_op_weight"].first().to_numpy(),
            1.0, atol=1e-6,
        ),
        "static_mass": np.allclose(
            schedule.groupby(["epoch", "query_index"])["static_weight"].sum().to_numpy()
            + schedule.groupby(["epoch", "query_index"])["static_selected_no_op_weight"].first().to_numpy(),
            1.0, atol=1e-6,
        ),
        "dose_matched": np.allclose(
            schedule.groupby(["epoch", "query_index"])["dynamic_weight"].sum().to_numpy(),
            schedule.groupby(["epoch", "query_index"])["static_weight"].sum().to_numpy(),
            atol=1e-7,
        ),
        "epochs": sorted(schedule["epoch"].unique().tolist()) == list(range(1, int(report.get("epochs", 0)) + 1)),
        "source": set(schedule["source"]) == {"N", "P_intensity", "P_transfer"},
        "pass": report.get("pass_to_gpu_replay") is True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Phase-A schedule validation failed: {gates}")
    print(f"[validate_noise_final_dynamic_direct_schedule] PASS actions={len(schedule):,}")


if __name__ == "__main__":
    main()
