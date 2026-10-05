#!/usr/bin/env python
"""Validate a completed B18 contextual action artifact."""
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
        raise RuntimeError("B18 row/query coverage changed")
    if set(rows["source"].astype(str)) != set(report["nested_oof"]["by_domain"]):
        raise RuntimeError("B18 domain coverage changed")
    summary = report["nested_oof"]
    if int(rows["corrected"].sum()) != int(summary["corrected"]):
        raise RuntimeError("B18 corrected count mismatch")
    if int(rows["introduced"].sum()) != int(summary["introduced"]):
        raise RuntimeError("B18 introduced count mismatch")
    if abs(float(rows["delta"].mean()) - float(summary["delta_recall1"])) > 1e-15:
        raise RuntimeError("B18 delta mismatch")
    print(
        "[validate_bioaware_b18_contextual_pairwise_action] PASS",
        {key: summary[key] for key in (
            "delta_recall1", "corrected", "introduced", "risk_net_lambda2"
        )},
    )


if __name__ == "__main__":
    main()
