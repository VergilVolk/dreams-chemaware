#!/usr/bin/env python
"""Fail-closed structural validation for B28, independent of scientific pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_bioaware_b28_branch_spectral_arbitration.py OUTPUT")
    directory = Path(sys.argv[1])
    report_path = directory / "report.json"
    transitions = directory / "nested_domain_loso_transitions.csv.gz"
    for path in (report_path, transitions):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b28_branch_spectral_arbitration_complete":
        raise RuntimeError("wrong B28 status")
    if len(report.get("folds", [])) != 6:
        raise RuntimeError("B28 lacks six nested source folds")
    contracts = report.get("contracts", {})
    if not all(contracts.get(name) is True for name in (
        "only_B12_B16_disagreements_changed", "candidate_own_reference_spectrum_used",
    )):
        raise RuntimeError("B28 positive contract failed")
    if not all(contracts.get(name) is False for name in (
        "outer_outcome_used_for_policy_selection", "truth_used_as_action_feature",
        "P2b_used", "phenotype_used", "shared_embedding_changed",
    )):
        raise RuntimeError("B28 forbidden-information contract failed")
    print(
        "[validate_bioaware_b28_branch_spectral_arbitration] PASS audit_complete "
        f"strictly_better={report['strictly_better_action_than_B17']}"
    )


if __name__ == "__main__":
    main()
