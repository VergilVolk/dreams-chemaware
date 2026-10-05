#!/usr/bin/env python
"""Independent validator for the sealed B44 MassBank panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b11_catalog_interaction_action import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "report": args.input_dir / "report.json",
        "queries": args.input_dir / "queries.csv.gz",
        "candidate_references": args.input_dir / "candidate_references.csv.gz",
        "reference_library": args.input_dir / "reference_library.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    queries = pd.read_csv(paths["queries"])
    candidates = pd.read_csv(paths["candidate_references"])
    library = pd.read_csv(paths["reference_library"])
    if report.get("status") != "bioaware_b44_massbank_panel_sealed" or not report.get("formal"):
        raise RuntimeError("unexpected B44 panel status")
    if report.get("outcomes_computed") or report.get("embedding_values_read") or report.get("model_fitted"):
        raise RuntimeError("B44 panel lock is outcome contaminated")
    if not report.get("pass_to_one_time_evaluation") or not all(report.get("gates", {}).values()):
        raise RuntimeError("B44 panel did not pass")
    if len(queries) != report["panel"]["queries"] or queries["truth_ik14"].nunique() != len(queries):
        raise RuntimeError("B44 query count or identity uniqueness drift")
    if set(queries["query_hdf5_row"]) & set(candidates["reference_hdf5_row"]):
        raise RuntimeError("B44 query/reference row leakage")
    if not candidates.groupby("query_id")["candidate_ik14"].nunique().ge(2).all():
        raise RuntimeError("B44 candidate identity multiplicity drift")
    if not candidates.groupby("query_id")["is_positive"].sum().ge(1).all():
        raise RuntimeError("B44 positive reference coverage drift")
    if set(candidates["reference_hdf5_row"]) != set(library["hdf5_row"]):
        raise RuntimeError("B44 reference library row coverage mismatch")
    for key in ("queries", "candidate_references", "reference_library"):
        if report.get("provenance", {}).get(key) != sha256(paths[key]):
            raise RuntimeError(f"B44 provenance mismatch: {key}")
    print("[validate_bioaware_b44_massbank_blind_panel] PASS", report["panel"], flush=True)


if __name__ == "__main__":
    main()
