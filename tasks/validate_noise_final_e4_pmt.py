"""Fail-closed artifact validator for E4-PMT manifest or summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = (
        "noise_final_e4_pmt_summary_complete" if args.summary
        else "noise_final_e4_pmt_manifest_complete"
    )
    if report.get("status") != expected or report.get("formal") is not True:
        raise RuntimeError(f"unexpected PMT artifact status: {report.get('status')}")
    if args.summary:
        if set(report.get("arms", {})) != {"clean_duplicate", "matched_random", "alpha025", "alpha050"}:
            raise RuntimeError("PMT summary does not contain exactly four arms")
        if not (args.output_dir / "paired_per_query.csv.gz").is_file():
            raise FileNotFoundError("PMT paired per-query ledger is absent")
    else:
        contracts = report.get("contracts", {})
        for key in (
            "all_mature_n_actions_routed", "nine_mature_n_cells_only",
            "noncorrective_target_weight_exact_zero", "outer_fold_absent",
            "current_geometry_full_candidate_replay",
            "selected_control_reproduces_frozen_R0_path",
        ):
            if contracts.get(key) is not True:
                raise RuntimeError(f"PMT manifest contract failed: {key}")
        if contracts.get("M2_predictions_used") is not False or contracts.get("P_actions_used") is not False:
            raise RuntimeError("PMT manifest used forbidden M2/P information")
        for name in ("all_routed_actions.csv.gz", "corrective_actions.csv.gz"):
            if not (args.output_dir / name).is_file():
                raise FileNotFoundError(name)
    print(f"[validate_noise_final_e4_pmt] PASS status={expected}")


if __name__ == "__main__":
    main()
