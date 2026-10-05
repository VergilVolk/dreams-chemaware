#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b47_u2_catalog_coverage import route_decision, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    report_path = directory / "report.json"
    ledger_path = directory / "consensus_identity_coverage.csv.gz"
    if not report_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError(directory)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b47_u2_catalog_coverage_complete"
        or report.get("formal") is not True
        or report.get("truth_blind") is not True
        or report.get("provenance", {}).get("identity_ledger_sha256")
        != sha256_file(ledger_path)
    ):
        raise RuntimeError("invalid U2 report or provenance")
    ledger = pd.read_csv(ledger_path)
    if len(ledger) != 238 or ledger["candidate_id"].nunique() != 238:
        raise RuntimeError("U2 consensus identity denominator changed")
    coverage = report["coverage"]
    if sum(int(value) for value in coverage["coverage_class_counts"].values()) != 238:
        raise RuntimeError("U2 coverage classes do not partition the denominator")
    if not coverage.get("by_study") or any(
        int(row["identities"]) <= 0 for row in coverage["by_study"].values()
    ):
        raise RuntimeError("U2 source-stratified coverage is missing")
    expected = route_decision(
        int(coverage["rhea_safe"]),
        int(coverage["rhea_or_strict_kegg_safe_provisional"]),
        int(coverage["rhea_or_kegg_or_emrn_edge"]),
    )
    if report.get("fixed_route_decision") != expected:
        raise RuntimeError("U2 fixed route decision is inconsistent")
    if report.get("contracts", {}).get("reaction_event_scored") is not False:
        raise RuntimeError("U2 must not score reaction events")
    if report.get("contracts", {}).get("strict_kegg_only_currency_status_provisional") is not True:
        raise RuntimeError("U2 lost the strict-KEGG safety boundary")
    print(
        "[validate_bioaware_b47_u2_catalog_coverage] PASS "
        f"Rhea={coverage['rhea_safe']} "
        f"Rhea+KEGG={coverage['rhea_or_strict_kegg_safe_provisional']} "
        f"route={expected['code']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
