#!/usr/bin/env python
"""Fail-closed validation for a B26 canary output."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_bioaware_b26_direct_shared_embedding_canary.py OUTPUT")
    directory = Path(sys.argv[1])
    report_path = directory / "report.json"
    action_path = directory / "per_action.csv.gz"
    for path in (report_path, action_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b26_direct_shared_embedding_canary_complete":
        raise RuntimeError("wrong B26 status")
    if report.get("pass_to_formula_isolated_training") is not True:
        raise RuntimeError(f"B26 scientific gate failed: {report.get('gates')}")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError("B26 contains a failed sub-gate")
    contracts = report.get("contracts", {})
    required_true = (
        "query_reference_encoder_shared", "inference_candidate_independent",
        "physical_query_duplicates_collapsed", "identity_balanced_sampling",
        "dropout_disabled",
    )
    required_false = (
        "catalogue_score_distilled", "reaction_neighbour_used_as_identity_positive",
        "P2b_used", "phenotype_used",
    )
    if not all(contracts.get(name) is True for name in required_true):
        raise RuntimeError("B26 positive contract failed")
    if not all(contracts.get(name) is False for name in required_false):
        raise RuntimeError("B26 forbidden-information contract failed")
    print(
        "[validate_bioaware_b26_direct_shared_embedding_canary] PASS "
        f"corrected={report['adapted']['corrected']} "
        f"introduced={report['adapted']['introduced']}"
    )


if __name__ == "__main__":
    main()
