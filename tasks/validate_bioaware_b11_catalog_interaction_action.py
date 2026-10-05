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
    transitions = args.output_dir / "nested_source_loso_transitions.csv.gz"
    if not report_path.is_file() or not transitions.is_file():
        raise FileNotFoundError("B11 output is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b11_catalog_interaction_action_complete":
        raise RuntimeError("unexpected B11 status")
    if not report.get("formal"):
        raise RuntimeError("B11 result is not formal")
    if not report.get("pass_to_external_catalog_action_validation"):
        raise RuntimeError(f"B11 scientific gate failed: {report.get('gates')}")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError("not all B11 gates passed")
    if any(report["b4_exact_replay"]["mismatches"].values()):
        raise RuntimeError("B4 replay mismatch")
    print(
        "[BioAware B11 validation] PASS",
        report["nested_oof"],
        flush=True,
    )


if __name__ == "__main__":
    main()
