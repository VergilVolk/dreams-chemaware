#!/usr/bin/env python
"""Fail-closed validator for the BioAware B2 reaction-transform audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    rotation_path = args.output_dir / "candidate_rotation_features.csv.gz"
    candidate_path = args.output_dir / "candidate_transform_features.csv.gz"
    for path in (report_path, rotation_path, candidate_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidate_path)
    rotations = pd.read_csv(rotation_path)
    if report.get("status") != "bioaware_b2_reaction_transform_action_complete":
        raise RuntimeError("unexpected B2 status")
    if len(candidates) != int(report["candidate_protocol"]["candidate_rows"]):
        raise RuntimeError("candidate row count mismatch")
    if candidates["query_id"].nunique() != int(report["candidate_protocol"]["queries"]):
        raise RuntimeError("candidate query count mismatch")
    if len(rotations) != int(report["rotation_rows"]):
        raise RuntimeError("rotation row count mismatch")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate candidate pair")
    required_contracts = {
        "direct_reactions_only",
        "seed_identity_held_out_by_rotation",
        "exact_adduct_and_polarity_required",
        "candidate_features_truth_blind",
        "held_source_truth_identity_and_formula_purged",
    }
    if not all(report["contracts"].get(key) is True for key in required_contracts):
        raise RuntimeError("B2 scientific contracts failed")
    if report["contracts"].get("P2b_used") is not False:
        raise RuntimeError("P2b is forbidden")
    if report.get("formal"):
        if report["candidate_protocol"]["queries"] != 1426:
            raise RuntimeError("formal query count changed")
        if report["candidate_protocol"]["candidate_rows"] != 5384:
            raise RuntimeError("formal candidate count changed")
        if set(report["recipes"]) != {
            "catalog_opportunity", "generic_edge", "wrong_shift_control", "reaction_transform"
        }:
            raise RuntimeError("formal recipe set changed")
        if set(report["gates"]) != {
            "transform_increment_ge_3pp", "identity_ci_low_positive",
            "formula_ci_low_positive", "corrected_gt_2x_introduced",
            "corrected_identities_ge_25", "at_least_3_of_4_sources_nonnegative",
            "wrong_shift_control_beaten",
        }:
            raise RuntimeError("formal gate set changed")
    print(
        "[validate_bioaware_b2_reaction_transform_action] PASS "
        f"queries={report['candidate_protocol']['queries']:,} "
        f"transform_pairs={report['candidate_pairs_with_transform']:,} "
        f"scientific_pass={report['scientific_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
