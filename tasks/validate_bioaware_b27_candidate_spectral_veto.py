#!/usr/bin/env python
"""Validate a B27 result, retaining a scientific failure as a valid audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_bioaware_b27_candidate_spectral_veto.py OUTPUT")
    directory = Path(sys.argv[1])
    report_path = directory / "report.json"
    transition_path = directory / "nested_domain_loso_transitions.csv.gz"
    for path in (report_path, transition_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b27_candidate_spectral_veto_complete":
        raise RuntimeError("wrong B27 status")
    if len(report.get("folds", [])) != 6:
        raise RuntimeError("B27 lacks six nested source folds")
    contracts = report.get("contracts", {})
    if not all(contracts.get(name) is True for name in (
        "candidate_own_reference_spectrum_used", "veto_only_reverts_to_DreaMS",
    )):
        raise RuntimeError("B27 positive contract failed")
    if not all(contracts.get(name) is False for name in (
        "outer_outcome_used_for_policy_selection", "truth_used_as_action_feature",
        "reaction_neighbour_spectrum_used", "P2b_used", "phenotype_used",
        "shared_embedding_changed",
    )):
        raise RuntimeError("B27 forbidden-information contract failed")
    print(
        "[validate_bioaware_b27_candidate_spectral_veto] PASS audit_complete "
        f"strictly_better={report['strictly_better_action_than_B17']}"
    )


if __name__ == "__main__":
    main()
