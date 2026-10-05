#!/usr/bin/env python
"""Fail-closed validator for B3b."""
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
    expected = {"queries": 548, "identities": 164, "formulas": 136, "sources": 4}
    for key, value in expected.items():
        if int(protocol[key]) != value:
            raise RuntimeError(f"{key} changed: {protocol[key]} != {value}")
    contracts = report["contracts"]
    if not contracts["negative_training_only"] or not contracts["single_incremental_reaction_coordinate"]:
        raise RuntimeError("B3b minimal negative-only contract failed")
    if contracts["positive_interventions"] != 0 or contracts["P2b_used"] or contracts["phenotype_used"] or contracts["shared_embedding_changed"]:
        raise RuntimeError("forbidden dependency or action detected")
    left = set(report["recipes"]["reaction_coabundance_minimal"]["features"])
    right = set(report["recipes"]["matched_random_minimal"]["features"])
    if left - right != {"coabundance_abs_excess_top3_mean"} or right - left:
        raise RuntimeError("recipes do not differ by exactly the frozen coordinate")
    print(
        "[validate_bioaware_b3b_negative_coabundance_action] PASS "
        f"queries={protocol['queries']} scientific_pass={report['pass_to_context_model']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
