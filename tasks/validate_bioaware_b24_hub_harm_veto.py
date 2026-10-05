#!/usr/bin/env python
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
    transitions = pd.read_csv(args.output_dir / "nested_domain_loso_transitions.csv.gz")
    if report["status"] != "bioaware_b24_hub_harm_veto_complete":
        raise RuntimeError("B24 status mismatch")
    if len(transitions) != 860 or transitions["query_id"].nunique() != 860:
        raise RuntimeError("B24 coverage mismatch")
    if any(report["frozen_B17_comparator"]["querywise_replay_mismatches"].values()):
        raise RuntimeError("B24 B17 replay mismatch")
    if report["contracts"]["P2b_used"] or report["contracts"]["phenotype_used"]:
        raise RuntimeError("B24 forbidden dependency")
    print("[validate_bioaware_b24_hub_harm_veto] PASS", {
        key: report["nested_oof"][key]
        for key in ("delta_recall1", "corrected", "introduced", "risk_net_lambda2")
    })


if __name__ == "__main__":
    main()
