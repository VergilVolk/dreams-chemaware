#!/usr/bin/env python
"""Independent validator for B40-M2 context-stratified evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from evaluate_bioaware_b39_m2_fixed_action import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.input_dir / "report.json"
    transitions_path = args.input_dir / "cell_context_transitions.csv.gz"
    for path in (report_path, transitions_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transitions = pd.read_csv(transitions_path, low_memory=False)
    if report.get("status") != "bioaware_b40_m2_context_strata_evaluation_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B40-M2 status")
    if report.get("evaluation_queries") != 860:
        raise RuntimeError("B40-M2 query panel drift")
    if transitions.duplicated(["cell_id", "query_id"]).any():
        raise RuntimeError("duplicate B40-M2 cell/query row")
    counts = transitions.groupby("cell_id")["query_id"].nunique()
    if set(counts.index) != {f"B40-0{i}" for i in range(6)} or counts.ne(860).any():
        raise RuntimeError(f"incomplete B40-M2 cells: {counts.to_dict()}")
    expected_strata = {
        "hidden_standard": 162,
        "sample_local_leave_one_seed_out": 150,
        "synthetic_rotation": 548,
    }
    observed = (
        transitions.loc[transitions["cell_id"].eq("B40-00")]
        .groupby("context_stratum")["query_id"].nunique().astype(int).to_dict()
    )
    if observed != expected_strata:
        raise RuntimeError(f"B40-M2 context strata drift: {observed}")
    if report.get("pass_to_context_representation"):
        raise RuntimeError("B40-M2 cannot directly authorize context representation")
    if report.get("provenance", {}).get("context_transitions") != sha256(transitions_path):
        raise RuntimeError("B40-M2 transition provenance mismatch")
    print(
        "[validate_bioaware_b40_m2_context_strata] PASS",
        {"prospective_reconstruction": report["pass_to_prospective_context_reconstruction"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
