"""Fail-closed validation for the R2 transfer audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    table_path = args.output_dir / "per_query.csv.gz"
    if not report_path.is_file() or not table_path.is_file():
        raise FileNotFoundError("incomplete R2 transfer audit")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)
    if report.get("status") != "noise_final_r2_transfer_audit_complete":
        raise RuntimeError("wrong audit status")
    if len(table) != 882 or not table["recorded_teacher_rank"].eq(1).all():
        raise RuntimeError("R1 recorded teacher table drifted")
    replay = report.get("fixed_teacher_replay", {})
    if int(replay.get("mismatches", 883)) != int(
        table["fixed_teacher_rank"].ne(table["recorded_teacher_rank"]).sum()
    ):
        raise RuntimeError("fixed teacher replay mismatch count is inconsistent")
    if float(replay.get("mismatch_fraction", 1.0)) > 0.005:
        raise RuntimeError("fixed teacher replay drift exceeds 0.5%")
    if report.get("contracts", {}).get("P2b") != "forbidden" or report["contracts"].get("P3_consumed"):
        raise RuntimeError("protocol contamination")
    print(
        "[validate_noise_final_r2_transfer_audit] PASS "
        f"train={report['train_formula_actions']['queries']} "
        f"held={report['held_formula_actions']['queries']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
