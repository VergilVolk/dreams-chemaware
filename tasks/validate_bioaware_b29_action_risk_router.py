#!/usr/bin/env python
"""Fail-closed artifact validator for BioAware B29."""
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
    transition_path = args.output / "nested_action_risk_transitions.csv.gz"
    if not report_path.is_file() or not transition_path.is_file():
        raise FileNotFoundError("B29 report or transitions missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(transition_path)
    if report.get("status") != "bioaware_b29_action_risk_router_complete":
        raise RuntimeError("B29 status changed")
    if len(frame) != 860 or frame["query_id"].nunique() != 860:
        raise RuntimeError("B29 transition coverage changed")
    required = {"b29_probability", "b29_threshold", "b29_execute", "B17_final_candidate_id"}
    if missing := required - set(frame):
        raise RuntimeError(f"B29 transition columns missing: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise RuntimeError("B29 transition table contains missing routed values")
    contracts = report.get("contracts", {})
    for key in (
        "router_can_only_execute_or_revert_B17",
        "outer_outcome_used_for_model_or_threshold_selection",
        "inner_predictions_crossfit_by_formula",
        "held_truth_identity_and_formula_purged",
        "physical_duplicates_have_unit_training_mass",
    ):
        expected = key != "outer_outcome_used_for_model_or_threshold_selection"
        if contracts.get(key) is not expected:
            raise RuntimeError(f"B29 contract changed: {key}")
    print(
        "[validate_bioaware_b29_action_risk_router] PASS "
        f"strictly_better={report.get('strictly_better_action_than_B17')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
