"""Fail-closed validator for E12-A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path, detail_path = args.output_dir / "report.json", args.output_dir / "residual_detail.csv.gz"
    if not report_path.is_file() or not detail_path.is_file():
        raise FileNotFoundError("E12-A report/detail incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    detail = pd.read_csv(detail_path)
    if report.get("status") != "noise_final_e12a_residual_reachability_complete" or not report.get("formal"):
        raise RuntimeError("E12-A status/formal failed")
    if report.get("held_queries") != 5923 or report.get("residual_errors") != 133 or len(detail) != 133:
        raise RuntimeError("E12-A task/residual count failed")
    if detail["query_index"].duplicated().any():
        raise RuntimeError("E12-A residual detail contains duplicate queries")
    contracts = report.get("contracts", {})
    if contracts.get("self_match_excluded_from_positive_teacher") is not True:
        raise RuntimeError("E12-A self-match exclusion contract failed")
    if contracts.get("no_new_action_outcomes_scored") is not True:
        raise RuntimeError("E12-A action-free audit contract failed")
    if contracts.get("P2b") != "forbidden" or contracts.get("P3_consumed") is not False:
        raise RuntimeError("E12-A isolation contract failed")
    expected_pass = all(report["gates"].values())
    if report["pass_to_e12b_relaxed_recurrence_matrix"] is not expected_pass:
        raise RuntimeError("E12-A gate decision is inconsistent")
    print("[validate_noise_final_e12a_residual_reachability] PASS", flush=True)


if __name__ == "__main__":
    main()
