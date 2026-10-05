#!/usr/bin/env python
"""Validate B17 output without refitting."""
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
        raise RuntimeError("B17 query coverage changed")
    print(
        "[validate_bioaware_b17_nested_union_action]",
        "PASS" if report.get("strictly_better_action_than_B12") else "SCIENTIFIC_FAIL",
        report["nested_oof"],
    )


if __name__ == "__main__":
    main()
