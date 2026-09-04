#!/usr/bin/env python
"""Fail-closed validation for the frozen BioAware candidate evidence ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(
        "data/validation/bioaware_candidate_evidence_ledger_v1"))
    args = parser.parse_args()
    report_path = args.input_dir / "report.json"
    ledger_path = args.input_dir / "candidate_evidence.csv.gz"
    if not report_path.exists() or not ledger_path.exists():
        raise FileNotFoundError(f"incomplete evidence ledger: {args.input_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_candidate_evidence_ledger_complete":
        raise RuntimeError("unexpected evidence-ledger status")
    if not report.get("formal"):
        raise RuntimeError("evidence ledger is not formal")
    if report.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("P2b contract violated")
    if sha256(ledger_path) != report.get("ledger_sha256"):
        raise RuntimeError("evidence-ledger SHA256 mismatch")
    frame = pd.read_csv(ledger_path)
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula", "spectral_score",
        *report.get("evidence_columns", []),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing ledger columns: {missing}")
    observed = {
        "queries": int(frame["query_id"].nunique()),
        "candidates": int(len(frame)),
        "identities": int(frame["truth_candidate_id"].nunique()),
        "formulas": int(frame["truth_formula"].nunique()),
    }
    for key, value in observed.items():
        if value != int(report[key]):
            raise RuntimeError(f"ledger {key} mismatch: {value} != {report[key]}")
    print(json.dumps({
        "status": "bioaware_candidate_evidence_ledger_validation_passed",
        **observed,
        "ledger_sha256": report["ledger_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
