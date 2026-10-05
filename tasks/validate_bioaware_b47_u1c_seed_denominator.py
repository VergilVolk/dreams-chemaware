#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_u1_core import sha256_file  # noqa: E402
from audit_bioaware_b47_u1c_seed_denominator import (  # noqa: E402
    first_stage_below, fixed_route,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    report_path, ledger_path = directory / "report.json", directory / "query_stage_ledger.csv.gz"
    if not report_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError(directory)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b47_u1c_seed_denominator_audit_complete"
        or report.get("formal") is not True
        or report.get("truth_blind") is not True
        or report.get("provenance", {}).get("query_stage_ledger_sha256")
        != sha256_file(ledger_path)
    ):
        raise RuntimeError("U1c report/provenance is invalid")
    contracts = report.get("contracts", {})
    if any(contracts.get(name) is not False for name in (
        "truth_opened", "phenotype_used", "model_fitted", "threshold_retuned",
        "reaction_network_scored", "P2b_used",
    )):
        raise RuntimeError("U1c truth-blind contract changed")
    ledger = pd.read_csv(ledger_path)
    if len(ledger) != 51976 or ledger["query_id"].nunique() != 51976:
        raise RuntimeError("U1c query denominator changed")
    stages = report.get("stages", {})
    order = (
        "all_queries", "unique_top1", "primary_absolute_gate",
        "absolute_and_feature_consensus",
        "reaction_graph_eligible_after_spectral_consensus",
        "after_sample_candidate_collapse",
    )
    events = [int(stages[name]["query_events"]) for name in order]
    if any(right > left for left, right in zip(events, events[1:])):
        raise RuntimeError("U1c stage counts are not monotone")
    expected_first = first_stage_below(stages, int(report["identity_gate"]["required"]))
    if report["identity_gate"]["first_stage_below_200_identities"] != expected_first:
        raise RuntimeError("U1c first-failure stage is inconsistent")
    expected_route = fixed_route(
        expected_first,
        int(report["candidate_universe"]["rhea_safe_degree_noncurrency"]),
    )
    if report.get("fixed_route_decision") != expected_route:
        raise RuntimeError("U1c fixed route decision is inconsistent")
    print(
        "[validate_bioaware_b47_u1c_seed_denominator] PASS "
        f"identities={report['identity_gate']['observed']} "
        f"shortfall={report['identity_gate']['shortfall']} "
        f"first_below={report['identity_gate']['first_stage_below_200_identities']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
