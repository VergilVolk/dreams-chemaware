#!/usr/bin/env python
"""Independent validator for BioAware B39-M2 fixed-action results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if report.get("status") != "bioaware_b39_m2_internal_fixed_action_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B39-M2 report status")
    if report.get("evaluation_queries") != 548 or set(report.get("evaluation_sources", [])) != {"BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"}:
        raise RuntimeError("B39-M2 evaluation panel drift")
    evaluated = [item for item in report["cell_reports"].values() if item["status"] == "evaluated"]
    unavailable = [item for item in report["cell_reports"].values() if item["status"] == "not_constructible_from_m1"]
    if len(evaluated) != 10 or len(unavailable) != 2:
        raise RuntimeError("B39-M2 cell execution count drift")
    if transitions.groupby("cell_id")["query_id"].nunique().ne(548).any():
        raise RuntimeError("incomplete per-cell query pairing")
    if transitions.duplicated(["cell_id", "query_id"]).any():
        raise RuntimeError("duplicate per-cell transition")
    if not transitions["atomic_final_correct"].astype(str).str.lower().isin({"true", "false"}).all():
        raise RuntimeError("invalid final correctness values")
    if report.get("pass_to_context_representation"):
        raise RuntimeError("M2 must never directly pass to context representation")
    if report.get("provenance", {}).get("cell_transitions") != sha256(transitions_path):
        raise RuntimeError("B39-M2 transition provenance mismatch")
    nonidentifying = report["cell_reports"]["B39-07"]
    if nonidentifying.get("identifiability") != "nonidentifying_all_events_candidate_specific":
        raise RuntimeError("candidate-specificity ablation limitation was hidden")
    print(
        "[validate_bioaware_b39_m2_fixed_action] PASS",
        {
            "queries": report["evaluation_queries"],
            "evaluated_cells": len(evaluated),
            "external_candidates": report["cells_candidate_for_external_reconstruction"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
