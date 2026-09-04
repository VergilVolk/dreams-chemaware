"""Fail-closed validation of the complete B1-C0 error-space audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [
        "decision.json", "query_error_space.csv.gz",
        "priority_teacher_rescue_missed.csv.gz", "introduced_errors.csv.gz",
    ]
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing B1-C0 outputs: {missing}")
    decision = json.loads((args.output_dir / "decision.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.output_dir / "query_error_space.csv.gz")
    if len(frame) != 4998 or int(decision["queries"]) != 4998:
        raise RuntimeError("formal B1-C0 requires 4,998 queries")
    counts = decision["outcome_counts"]
    if sum(map(int, counts.values())) != 4998:
        raise RuntimeError("B1-C0 outcomes are not exhaustive")
    teacher_rescue = (
        int(counts.get("teacher_rescue_student_recovered", 0))
        + int(counts.get("teacher_rescue_student_missed", 0))
    )
    if teacher_rescue != 542:
        raise RuntimeError(f"B1-C0 lost B0 teacher rescues: {teacher_rescue}")
    if int(counts.get("student_introduced_error", 0)) != 45:
        raise RuntimeError("B1-C0 introduced-error count differs from frozen B1")
    corrected = (
        int(counts.get("teacher_rescue_student_recovered", 0))
        + int(counts.get("student_independent_correction", 0))
    )
    if corrected != 98:
        raise RuntimeError("B1-C0 corrected count differs from frozen B1")
    for name, probe in decision["formula_group_oof_diagnostic_probes"].items():
        if probe.get("status") != "formula_group_oof":
            raise RuntimeError(f"diagnostic probe did not run: {name}")
    print(f"[validate_noise_v3_a4b_b1_error_space] PASS: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
