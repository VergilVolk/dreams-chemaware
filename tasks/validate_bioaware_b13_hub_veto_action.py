#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    transitions = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    if not report_path.is_file() or not transitions.is_file():
        raise FileNotFoundError("B13 output is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b13_hub_veto_action_complete":
        raise RuntimeError("unexpected B13 status")
    if not report.get("formal") or not all(report.get("gates", {}).values()):
        raise RuntimeError(f"B13 scientific gate failed: {report.get('gates')}")
    if not report.get("pass_to_hub_safe_action_construction"):
        raise RuntimeError("B13 did not pass to action construction")
    print("[BioAware B13 validation] PASS", report["nested_oof"], flush=True)


if __name__ == "__main__":
    main()
