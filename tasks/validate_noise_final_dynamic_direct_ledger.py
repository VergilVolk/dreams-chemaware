"""Independent validator for the unified dynamic-direct N/P action ledger."""
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
    required = [args.output_dir / "report.json", args.output_dir / "training_actions.csv.gz",
                args.output_dir / "cell_summary.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    actions = pd.read_csv(required[1], low_memory=False)
    cells = pd.read_csv(required[2])
    contracts = report.get("contracts", {})
    outcome_columns = {
        "corrected", "introduced", "target_rank", "control_rank",
        "positive_guided_oracle_recoverable", "transfer_oracle_recoverable",
    }
    gates = {
        "status": report.get("status") == "noise_final_dynamic_direct_action_ledger_complete",
        "formal": report.get("formal") is True,
        "action_count": len(actions) == int(report.get("actions", -1)),
        "unique_actions": not actions["action_id"].duplicated().any(),
        "all_cells": len(cells) == 30 and contracts.get("all_30_cells_retained") is True,
        "N_and_P": set(actions["source"].astype(str)) == {"N", "P_intensity", "P_transfer"},
        "positive_weights": actions[["dynamic_weight", "static_weight"]].gt(0).all().all(),
        "query_caps": bool(
            actions.groupby("query_index")["dynamic_weight"].sum().max() <= 1.00001
            and actions.groupby("query_index")["static_weight"].sum().max() <= 1.00001
        ),
        "no_outcome_columns": not bool(outcome_columns & set(actions.columns)),
        "held_absent": contracts.get("outer_held_formulas_absent") is True,
        "multiple_actions": contracts.get("multiple_actions_per_query_retained") is True,
        "no_op": contracts.get("no_op_implicit_and_always_available") is True,
        "P2b": contracts.get("P2b") == "forbidden",
        "P3": contracts.get("P3_consumed") is False,
        "pass": report.get("pass_to_gpu_replay") is True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"dynamic-direct ledger validation failed: {gates}")
    print(f"[validate_noise_final_dynamic_direct_ledger] PASS actions={len(actions):,}")


if __name__ == "__main__":
    main()
