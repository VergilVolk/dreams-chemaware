#!/usr/bin/env python
"""Independent validator for B40-M1 graph-completion evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_bioaware_b39_m2_fixed_action import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.input_dir / "report.json"
    transitions_path = args.input_dir / "cell_transitions.csv.gz"
    for path in (report_path, transitions_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transitions = pd.read_csv(transitions_path, low_memory=False)
    if report.get("status") != "bioaware_b40_m1_graph_completion_evaluation_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B40-M1 status")
    if report.get("evaluation_queries") != 548 or report.get("mapped_competition_primary_queries") != 298:
        raise RuntimeError("B40-M1 evaluation panel drift")
    if transitions.groupby("cell_id")["query_id"].nunique().ne(548).any() or transitions.duplicated(["cell_id", "query_id"]).any():
        raise RuntimeError("incomplete or duplicate B40-M1 transitions")
    if set(report["cell_reports"]) != {f"B40-0{index}" for index in range(6)}:
        raise RuntimeError("B40-M1 cell set drift")
    if any(report["cell_reports"][cell]["candidate_for_external_reconstruction"] for cell in ("B40-02", "B40-03", "B40-04", "B40-05")):
        raise RuntimeError("negative control was promoted")
    if report.get("pass_to_context_representation"):
        raise RuntimeError("B40-M1 may not pass directly to context representation")
    if report.get("provenance", {}).get("cell_transitions") != sha256(transitions_path):
        raise RuntimeError("B40-M1 transition provenance mismatch")
    print("[validate_bioaware_b40_m1_graph_completion] PASS", {"selected": report["cells_candidate_for_external_reconstruction"]}, flush=True)


if __name__ == "__main__":
    main()
