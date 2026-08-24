"""Fail-closed validation for the Noise-v3 S2 sequential matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SELECTORS = {"candidate_gradient", "role_confounder"}
EXPECTED_DOSES = {0.50, 1.00}
EXPECTED_STEPS = {1, 2, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s2_sequential"),
    )
    return parser.parse_args()


def integers(value: object, separator: str = ",") -> list[int]:
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [int(item) for item in text.split(separator) if item != ""]


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
    if report.get("formal") is not True or int(report.get("queries", 0)) < 20_000:
        raise RuntimeError("S2 is not a formal full-graph run")
    matrix = report.get("matrix", {})
    if set(map(str, matrix.get("selectors", ()))) != EXPECTED_SELECTORS:
        raise RuntimeError("selector mismatch")
    if set(map(float, matrix.get("attenuations", ()))) != EXPECTED_DOSES:
        raise RuntimeError("dose mismatch")
    if int(matrix.get("maximum_steps", 0)) != 3:
        raise RuntimeError("step mismatch")
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
        "query_index", "selector", "attenuation", "step", "target_path",
        "target_role", "baseline_rank", "target_rank", "random_repeats",
        "corrected", "introduced",
    }
    missing = required - set(paired.columns)
    if missing:
        raise RuntimeError(f"paired table misses {sorted(missing)}")
    if paired.duplicated(["query_index", "selector", "attenuation", "step"]).any():
        raise RuntimeError("duplicate query-selector-dose-step cell")
    if (paired["random_repeats"] != int(matrix["control_repeats"])).any():
        raise RuntimeError("incomplete matched controls entered paired table")
    if (paired["step"] < 1).any() or (paired["step"] > 3).any():
        raise RuntimeError("invalid sequential step")
    for row in paired.itertuples(index=False):
        path = integers(row.target_path)
        if len(path) != int(row.step) or len(set(path)) != len(path):
            raise RuntimeError("target path is not a unique prefix of requested length")
        if any(token <= 0 for token in path):
            raise RuntimeError("precursor/padding entered a target path")
        if str(row.selector) == "role_confounder" and str(row.target_role) != "confounder_only":
            raise RuntimeError("role path contains a non-confounder action")
        if str(row.target_role) == "identity_only":
            raise RuntimeError("identity-only peak entered a target path")

    if sequences.duplicated(["query_index", "selector", "attenuation"]).any():
        raise RuntimeError("multiple registered sequences for one query-action")
    for row in sequences.itertuples(index=False):
        targets = integers(row.target_tokens)
        if len(targets) != int(row.steps) or len(set(targets)) != len(targets):
            raise RuntimeError("selected target sequence is malformed")
        controls = [integers(path) for path in str(row.control_paths).split(";")]
        if len(controls) != int(matrix["control_repeats"]):
            raise RuntimeError("wrong number of control paths")
        flattened = []
        for path in controls:
            if len(path) != len(targets) or len(set(path)) != len(path):
                raise RuntimeError("control path is incomplete or reuses a token")
            flattened.extend(path)
        if set(flattened) & set(targets):
            raise RuntimeError("target and matched-control paths overlap")
        if len(set(flattened)) != len(flattened):
            raise RuntimeError("matched-control repeats are not independent paths")

    actual_cells = set(zip(
        paired["selector"].astype(str),
        paired["attenuation"].astype(float).round(2),
        paired["step"].astype(int),
    ))
    expected_cells = {
        (selector, dose, step)
        for selector in EXPECTED_SELECTORS for dose in EXPECTED_DOSES for step in EXPECTED_STEPS
    }
    if not expected_cells <= actual_cells:
        raise RuntimeError(f"missing S2 cells: {sorted(expected_cells - actual_cells)}")
    output = {
        "status": "noise_v3_s2_sequential_matrix_validation_passed",
        "formal_queries": int(report["queries"]),
        "sequences": int(len(sequences)),
        "paired_cells": int(len(paired)),
        "all_2_selectors_x_2_doses_x_3_steps_present": True,
        "unique_nonidentity_target_paths": True,
        "complete_disjoint_matched_control_paths": True,
        "dynamic_recalculation_flags": True,
    }
    destination = args.output_dir / "matrix_validation.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
