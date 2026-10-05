#!/usr/bin/env python
"""Fail-closed result validator for B4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    protocol = report["candidate_protocol"]
    expected = {
        "all_queries": 1426, "negative_queries": 548,
        "negative_identities": 164, "negative_formulas": 136, "sources": 4,
    }
    for key, value in expected.items():
        if int(protocol[key]) != value:
            raise RuntimeError(f"{key} changed: {protocol[key]} != {value}")
    if set(report["protocols"]) != {"mixed_polarity", "negative_only"}:
        raise RuntimeError("training protocol set changed")
    for protocol_report in report["protocols"].values():
        if set(protocol_report["recipes"]) != {
            "spectral_only", "reference_only", "graph_only", "combined_opportunity"
        }:
            raise RuntimeError("recipe set changed")
        for recipe in protocol_report["recipes"].values():
            if int(recipe["negative"]["queries"]) != 548:
                raise RuntimeError("primary recipe denominator is not 548 negative queries")
    contracts = report["contracts"]
    if contracts["positive_interventions"] != 0 or contracts["P2b_used"] or contracts["phenotype_used"] or contracts["shared_embedding_changed"]:
        raise RuntimeError("forbidden action/dependency detected")
    print(
        "[validate_bioaware_b4_opportunity_decomposition] PASS "
        f"graph_component={report['graph_component_is_bioaware']}", flush=True
    )


if __name__ == "__main__":
    main()
