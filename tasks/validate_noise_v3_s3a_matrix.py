"""Fail-closed validation for the preregistered Noise-v3 S3A matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_ACTIONS = {
    ("candidate_gradient", 0.50),
    ("role_confounder", 1.00),
    ("role_shared", 0.25),
    ("role_shared", 0.50),
    ("role_shared", 1.00),
    ("role_unmatched", 0.25),
    ("role_unmatched", 0.50),
    ("role_unmatched", 1.00),
}
EXPECTED_STEPS = set(range(1, 7))
EXPECTED_ROLES = {
    "role_confounder": "confounder_only",
    "role_shared": "shared",
    "role_unmatched": "unmatched",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s3a_extended_matrix"),
    )
    return parser.parse_args()


def integers(value: object, separator: str = ",") -> list[int]:
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [int(item) for item in text.split(separator) if item]


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    paired_path = args.output_dir / "paired_interventions.csv.gz"
    sequence_path = args.output_dir / "selected_sequences.csv.gz"
    for path in (report_path, paired_path, sequence_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path)
    sequences = pd.read_csv(sequence_path)
    if report.get("formal") is not True or int(report.get("queries", 0)) != 23876:
        raise RuntimeError("S3A is not the locked full candidate graph")
    matrix = report.get("matrix", {})
    registered = {
        (str(item["selector"]), round(float(item["attenuation"]), 2))
        for item in matrix.get("registered_actions", [])
    }
    if registered != EXPECTED_ACTIONS:
        raise RuntimeError(
            f"registered action mismatch: missing={EXPECTED_ACTIONS - registered}; "
            f"unexpected={registered - EXPECTED_ACTIONS}"
        )
    if int(matrix.get("maximum_steps", 0)) != 6:
        raise RuntimeError("S3A maximum steps must equal six")
    for flag in (
        "dynamic_candidate_recalculation_every_step",
        "dynamic_peak_role_recalculation_every_step",
        "identity_only_peaks_protected",
        "complete_matched_random_paths_only",
        "no_outcome_based_action_selection",
    ):
        if matrix.get(flag) is not True:
            raise RuntimeError(f"matrix integrity flag failed: {flag}")

    required = {
        "query_index", "query_ik14", "query_formula", "selector", "attenuation",
        "step", "target_path", "target_role", "baseline_rank", "target_rank",
        "winner_pair_row", "winner_ik14", "winner_formula", "winner_mces_grade",
        "positive_score", "hardest_negative_score", "random_repeats", "corrected",
        "introduced",
    }
    missing = required - set(paired.columns)
    if missing:
        raise RuntimeError(f"paired table misses {sorted(missing)}")
    if paired.duplicated(["query_index", "selector", "attenuation", "step"]).any():
        raise RuntimeError("duplicate query-action-step cell")
    if (paired["random_repeats"] != int(matrix["control_repeats"])).any():
        raise RuntimeError("incomplete matched controls entered S3A")
    if (paired["step"] < 1).any() or (paired["step"] > 6).any():
        raise RuntimeError("invalid S3A step")
    if (paired["corrected"] & paired["introduced"]).any():
        raise RuntimeError("an action cell cannot be both corrected and introduced")
    for row in paired.itertuples(index=False):
        path = integers(row.target_path)
        if len(path) != int(row.step) or len(path) != len(set(path)):
            raise RuntimeError("target path is malformed")
        expected_role = EXPECTED_ROLES.get(str(row.selector))
        if expected_role is not None and str(row.target_role) != expected_role:
            raise RuntimeError(f"{row.selector} selected {row.target_role}")
        if str(row.target_role) == "identity_only":
            raise RuntimeError("identity-only evidence entered an S3A path")

    if sequences.duplicated(["query_index", "selector", "attenuation"]).any():
        raise RuntimeError("multiple paths registered for one query-action")
    for row in sequences.itertuples(index=False):
        targets = integers(row.target_tokens)
        if len(targets) != int(row.steps) or len(targets) != len(set(targets)):
            raise RuntimeError("registered sequence is malformed")
        controls = [integers(path) for path in str(row.control_paths).split(";")]
        if len(controls) != int(matrix["control_repeats"]):
            raise RuntimeError("wrong number of S3A control paths")
        flattened: list[int] = []
        for path in controls:
            if len(path) != len(targets) or len(path) != len(set(path)):
                raise RuntimeError("control path is incomplete or reuses a token")
            flattened.extend(path)
        if set(flattened) & set(targets) or len(flattened) != len(set(flattened)):
            raise RuntimeError("target/control paths are not disjoint")

    cells = set(zip(
        paired["selector"].astype(str),
        paired["attenuation"].astype(float).round(2),
        paired["step"].astype(int),
    ))
    expected_cells = {
        (selector, dose, step)
        for selector, dose in EXPECTED_ACTIONS for step in EXPECTED_STEPS
    }
    missing_cells = expected_cells - cells
    if missing_cells:
        raise RuntimeError(f"missing preregistered S3A cells: {sorted(missing_cells)}")
    output = {
        "status": "noise_v3_s3a_matrix_validation_passed",
        "formal_queries": int(report["queries"]),
        "registered_actions": len(EXPECTED_ACTIONS),
        "registered_steps": len(EXPECTED_STEPS),
        "expected_cells": len(expected_cells),
        "paired_cells": int(len(paired)),
        "sequences": int(len(sequences)),
        "complete_outcome_and_winner_audit": True,
        "unique_nonidentity_paths": True,
        "complete_disjoint_controls": True,
    }
    destination = args.output_dir / "matrix_validation.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
