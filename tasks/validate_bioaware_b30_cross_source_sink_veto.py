#!/usr/bin/env python
"""Fail-closed artifact validator for BioAware B30."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report_path = args.output / "report.json"
    transition_path = args.output / "nested_sink_veto_transitions.csv.gz"
    history_path = args.output / "cross_source_candidate_history.csv.gz"
    for path in (report_path, transition_path, history_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(transition_path)
    if report.get("status") != "bioaware_b30_cross_source_sink_veto_complete":
        raise RuntimeError("B30 status changed")
    if len(frame) != 860 or frame["query_id"].nunique() != 860:
        raise RuntimeError("B30 transition coverage changed")
    if frame[["b30_candidate_sink", "b30_veto"]].isna().any().any():
        raise RuntimeError("B30 decision fields contain missing values")
    contracts = report.get("contracts", {})
    expected_true = (
        "candidate_sink_history_excludes_outer_source",
        "physical_duplicates_have_one_history_vote", "router_can_only_revert_B17",
        "unseen_candidate_falls_back_to_B17",
        "candidate_identity_is_explicit_known_candidate_memory",
    )
    for key in expected_true:
        if contracts.get(key) is not True:
            raise RuntimeError(f"B30 contract changed: {key}")
    if contracts.get("outer_outcome_used_for_sink_definition") is not False:
        raise RuntimeError("B30 outer outcome leakage contract changed")
    print(
        "[validate_bioaware_b30_cross_source_sink_veto] PASS "
        f"strictly_better={report.get('strictly_better_action_than_B17')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
