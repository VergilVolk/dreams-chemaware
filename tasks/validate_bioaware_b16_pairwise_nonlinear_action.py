#!/usr/bin/env python
"""Validate a B16 action-mining artifact without refitting."""
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
    frame = pd.read_csv(args.output_dir / "nested_domain_loso_transitions.csv.gz")
    if len(frame) != 860 or frame["query_id"].nunique() != 860:
        raise RuntimeError("B16 query coverage changed")
    if not set(frame["source"].astype(str)) == set(report["nested_oof"]["by_domain"]):
        raise RuntimeError("B16 domain report is incomplete")
    print(
        "[validate_bioaware_b16_pairwise_nonlinear_action]",
        "PASS" if report.get("strictly_better_action_than_B12") else "SCIENTIFIC_FAIL",
        report["nested_oof"],
    )


if __name__ == "__main__":
    main()
