#!/usr/bin/env python
"""Validate a completed B21 nested residual action."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(args.output_dir / "nested_domain_loso_transitions.csv.gz")
    if len(rows) != 860 or rows["query_id"].nunique() != 860:
        raise RuntimeError("B21 coverage changed")
    summary = report["nested_oof"]
    if int(rows["corrected"].sum()) != int(summary["corrected"]):
        raise RuntimeError("B21 corrected mismatch")
    if int(rows["introduced"].sum()) != int(summary["introduced"]):
        raise RuntimeError("B21 introduced mismatch")
    if any(report["frozen_B17_comparator"]["querywise_replay_mismatches"].values()):
        raise RuntimeError("B21 B17 replay is not exact")
    print(
        "[validate_bioaware_b21_nested_residual_fallback] PASS",
        {key: summary[key] for key in (
            "delta_recall1", "corrected", "introduced", "risk_net_lambda2"
        )},
    )


if __name__ == "__main__":
    main()
