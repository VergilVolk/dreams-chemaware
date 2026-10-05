#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b14_reaction_context_action_complete":
        raise RuntimeError("B14 status mismatch")
    if report["universe"]["queries"] != 860:
        raise RuntimeError("B14 query universe mismatch")
    transitions = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    if not transitions.is_file() or transitions.stat().st_size == 0:
        raise RuntimeError("B14 transitions missing")
    passed = bool(report.get("pass_to_reaction_context_action_construction"))
    print(
        "[validate_bioaware_b14_reaction_context_action] PASS "
        f"scientific_gate={passed}"
    )
    if not passed:
        raise RuntimeError(f"B14 scientific gate failed: {report['gates']}")


if __name__ == "__main__":
    main()
