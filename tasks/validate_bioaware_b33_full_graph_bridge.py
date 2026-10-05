#!/usr/bin/env python3
"""Validate a completed B33 full-graph shared-embedding bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report_path = args.directory / "report.json"
    checkpoint = args.directory / "checkpoint.pt"
    per_query = args.directory / "per_query.csv.gz"
    for path in (report_path, checkpoint, per_query):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b33_full_graph_bridge_complete":
        raise RuntimeError("B33 bridge status changed")
    if report.get("pass_to_formula_isolated_training") is not True:
        raise RuntimeError(f"B33 scientific gate failed: {report.get('gates')}")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError("B33 bridge has a failed component gate")
    if report.get("contracts", {}).get(
        "candidate_max_recomputed_over_all_reference_spectra"
    ) is not True:
        raise RuntimeError("B33 post-update full-reference max was not enforced")
    print(
        "[validate_bioaware_b33_full_graph_bridge] PASS "
        f"row_delta={report['row_weighted']['delta_recall1']:+.4f} "
        f"physical_delta={report['physical_query_weighted']['delta_recall1']:+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
