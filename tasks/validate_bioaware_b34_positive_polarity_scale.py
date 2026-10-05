#!/usr/bin/env python
"""Validate the BioAware B34 polarity-transfer scale audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    transitions_path = args.output_dir / "b34_large_scale_transitions.csv.gz"
    if not report_path.is_file() or not transitions_path.is_file():
        raise FileNotFoundError("B34 report or transition table is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(transitions_path)
    if report.get("status") != "bioaware_b34_positive_polarity_scale_complete":
        raise RuntimeError("unexpected B34 status")
    if len(frame) != 1738 or frame["query_id"].nunique() != 1738:
        raise RuntimeError("B34 expected 1,738 unique query rows")
    counts = frame["evaluation_stratum"].value_counts().to_dict()
    if counts != {
        "original_negative_replay": 860,
        "held_positive_transfer": 878,
    }:
        raise RuntimeError(f"B34 stratum counts changed: {counts}")
    required = (
        "B17_original_baseline_replay_exact",
        "B30_original_860_querywise_replay_exact",
        "positive_transfer_rows_eq_878",
    )
    if not all(report["gates"].get(key) is True for key in required):
        raise RuntimeError("B34 implementation/replay gates failed")
    if report["contracts"].get("held_source_positive_outcomes_used_for_fitting_or_selection") is not False:
        raise RuntimeError("B34 held-source positive outcome-selection contract changed")
    if report["contracts"].get("other_source_positive_outcomes_used_for_model_fitting") is not True:
        raise RuntimeError("B34 mixed-polarity fitting disclosure changed")
    if report["contracts"].get("positive_outcomes_used_for_sink_definition") is not False:
        raise RuntimeError("B34 sink contract changed")
    print("[validate_bioaware_b34_positive_polarity_scale] PASS")
    print(json.dumps({
        "positive_primary": report["held_positive_transfer_primary"],
        "mixed_polarity_descriptive_only": report["mixed_polarity_descriptive_only"],
        "scientific_gates": report["gates"],
        "pass_large_scale_polarity_transfer": report["pass_large_scale_polarity_transfer"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
