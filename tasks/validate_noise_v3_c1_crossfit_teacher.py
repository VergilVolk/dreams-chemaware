"""Fail-closed validator for formal C1 support-disjoint expansion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.output_dir / "decision.json"
    examples_path = args.output_dir / "crossfit_examples.csv.gz"
    rescues_path = args.output_dir / "crossfit_teacher_rescues.csv.gz"
    for path in (decision_path, examples_path, rescues_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(examples_path)
    if int(decision["graph_queries_considered"]) != 23876:
        raise RuntimeError("C1 did not cover the full graph")
    if len(frame) != int(decision["examples"]):
        raise RuntimeError("C1 example count mismatch")
    if frame.empty or frame["query_index"].nunique() < 1000:
        raise RuntimeError("C1 expansion is unexpectedly small")
    overlap = frame.apply(
        lambda row: str(int(row.evaluation_positive_row)) in str(row.teacher_rows).split(";"),
        axis=1,
    )
    if bool(overlap.any()):
        raise RuntimeError("C1 teacher/evaluation positive overlap detected")
    if int(frame["corrected"].sum()) != int(decision["corrected"]):
        raise RuntimeError("C1 corrected count mismatch")
    print(f"[validate_noise_v3_c1_crossfit_teacher] PASS: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
