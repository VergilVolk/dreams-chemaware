#!/usr/bin/env python
"""Independent validator for B43 catalogue-gain decomposition."""
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
    report_path = args.input_dir / "report.json"
    per_query_path = args.input_dir / "catalog_gain_mechanism_per_query.csv.gz"
    for path in (report_path, per_query_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(per_query_path, low_memory=False)
    if report.get("status") != "bioaware_b43_catalog_gain_mechanism_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B43 status")
    if report.get("pass_to_shared_embedding"):
        raise RuntimeError("B43 cannot authorize shared embedding")
    counts = frame.groupby("arm")["query_id"].nunique().astype(int)
    if set(counts.index) != set(report.get("arm_reports", {})) or counts.ne(860).any():
        raise RuntimeError(f"B43 per-query coverage drift: {counts.to_dict()}")
    if frame.duplicated(["arm", "query_id"]).any():
        raise RuntimeError("B43 arm/query rows are not unique")
    for arm, values in report["arm_reports"].items():
        local = frame.loc[frame["arm"].eq(arm)]
        if int(local["corrected"].sum()) != values["overall"]["corrected"]:
            raise RuntimeError(f"B43 corrected count mismatch: {arm}")
        if int(local["introduced"].sum()) != values["overall"]["introduced"]:
            raise RuntimeError(f"B43 introduced count mismatch: {arm}")
    if report.get("provenance", {}).get("per_query") != sha256(per_query_path):
        raise RuntimeError("B43 per-query provenance mismatch")
    print("[validate_bioaware_b43_catalog_gain_mechanism] PASS", {
        arm: gates["within_catalogue_signal_pass"]
        for arm, gates in report["mechanism_gates"].items()
    }, flush=True)


if __name__ == "__main__":
    main()
