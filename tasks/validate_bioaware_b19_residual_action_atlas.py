#!/usr/bin/env python
"""Validate a B19 residual action atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    ledger = json.loads((args.output_dir / "action_ledger.json").read_text(encoding="utf-8"))
    if report.get("actions_audited") != 150 or len(ledger.get("actions", [])) != 150:
        raise RuntimeError("B19 action coverage changed")
    if report["B17_comparator"] != {
        "corrected": 57,
        "introduced": 7,
        "residual_official_errors": 235,
        "risk_net_lambda2": 43,
    }:
        raise RuntimeError("B19 B17 comparator changed")
    if len(report.get("top_20_by_incremental_headroom_then_risk", [])) != 20:
        raise RuntimeError("B19 top-action ledger incomplete")
    print("[validate_bioaware_b19_residual_action_atlas] PASS")


if __name__ == "__main__":
    main()
