"""Fail-closed validator for E15-M1 replay/calibration output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "report": args.output_dir / "report.json",
        "corrective": args.output_dir / "calibrated_corrective_actions.csv.gz",
        "harmful": args.output_dir / "calibrated_harmful_actions.csv.gz",
        "calibration": args.output_dir / "source_local_calibration.csv",
        "panel": args.output_dir / "gradient_calibration_panel.csv.gz",
        "replay": args.output_dir / "source_replay_ledger.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e15_m1_replay_calibration_complete"
        or not report.get("formal") or not report.get("pass_to_32_query_overfit")
        or not all(report.get("gates", {}).values())
    ):
        raise RuntimeError("E15-M1 report did not pass every gate")
    corrective = pd.read_csv(required["corrective"])
    harmful = pd.read_csv(required["harmful"])
    panel = pd.read_csv(required["panel"])
    replay = pd.read_csv(required["replay"])
    for name, frame, kind in (
        ("corrective", corrective, "corrective"), ("harmful", harmful, "harmful"),
    ):
        if not frame["supervision_kind"].astype(str).eq(kind).all():
            raise RuntimeError(f"E15-M1 {name} branch is contaminated")
        if frame[["calibrated_strength", "source_kind_percentile"]].isna().any().any():
            raise RuntimeError(f"E15-M1 {name} branch is not calibrated")
    if len(panel) != 128 or panel.duplicated(
        ["source", "query_index", "action_id", "supervision_kind"]
    ).any():
        raise RuntimeError("E15-M1 gradient panel is not 128 unique actions")
    if not panel.groupby(["source", "supervision_kind"]).size().eq(16).all():
        raise RuntimeError("E15-M1 gradient panel is not source/branch balanced")
    if len(replay) != len(corrective) + len(harmful):
        raise RuntimeError("E15-M1 replay ledger does not cover every selected action")
    if report.get("contracts", {}).get("P2b") != "forbidden" or report.get("contracts", {}).get("P3_consumed"):
        raise RuntimeError("E15-M1 violates P2b/P3 isolation")
    print(
        f"[validate_noise_final_e15_replay_calibration] PASS actions={len(replay):,} panel={len(panel)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
