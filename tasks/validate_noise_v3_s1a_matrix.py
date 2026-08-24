"""Fail-closed structural validation of the completed Noise-v3 S1a matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SELECTORS = ("candidate_gradient", "role_confounder", "role_identity")
EXPECTED_DOSES = (0.25, 0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s1a_single_peak_matrix"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    paired_path = args.output_dir / "paired_interventions.csv.gz"
    selected_path = args.output_dir / "selected_peaks.csv.gz"
    for path in (report_path, paired_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path)
    selected = pd.read_csv(selected_path)
    matrix = report.get("orthogonal_matrix", {})
    if tuple(matrix.get("selectors", ())) != EXPECTED_SELECTORS:
        raise RuntimeError(f"selector mismatch: {matrix.get('selectors')}")
    doses = tuple(sorted(map(float, matrix.get("attenuation_doses", ()))))
    if not np.allclose(doses, EXPECTED_DOSES):
        raise RuntimeError(f"dose mismatch: {doses}")
    if report.get("formal") is not True or int(report.get("queries", 0)) < 20_000:
        raise RuntimeError("S1a is not a formal full-graph run")
    if not report.get("strict_same_role_control_results"):
        raise RuntimeError("strict same-role sensitivity analysis is missing")
    if not matrix.get("shared_eligibility_stratified_results"):
        raise RuntimeError("shared-eligibility selector comparison is missing")

    required = {
        "query_index", "selector", "attenuation", "target_token",
        "control_tokens", "baseline_rank", "target_rank",
        "target_minus_random_margin_change",
    }
    missing = required - set(paired.columns)
    if missing:
        raise RuntimeError(f"paired matrix misses columns: {sorted(missing)}")
    if paired.duplicated(["query_index", "selector", "attenuation"]).any():
        raise RuntimeError("duplicate query-selector-dose cells")

    cell_counts: dict[str, int] = {}
    for selector in EXPECTED_SELECTORS:
        selector_frame = paired.loc[paired["selector"] == selector].copy()
        if selector_frame.empty:
            raise RuntimeError(f"empty selector: {selector}")
        reference_queries = None
        for dose in EXPECTED_DOSES:
            cell = selector_frame.loc[np.isclose(selector_frame["attenuation"], dose)]
            if cell.empty:
                raise RuntimeError(f"empty matrix cell: {selector}@{dose}")
            query_set = set(map(int, cell["query_index"]))
            if reference_queries is None:
                reference_queries = query_set
            elif query_set != reference_queries:
                raise RuntimeError(f"dose-dependent query eligibility for {selector}")
            cell_counts[f"{selector}|a={dose:.2f}"] = int(len(cell))

        invariants = selector_frame.groupby("query_index", sort=False).agg(
            n_target=("target_token", "nunique"),
            n_controls=("control_tokens", "nunique"),
        )
        if (invariants["n_target"] != 1).any() or (invariants["n_controls"] != 1).any():
            raise RuntimeError(f"target/control changed across doses for {selector}")

    selected_required = {"selector", "target", "control_tokens", "target_role"}
    if selected_required - set(selected.columns):
        raise RuntimeError("selected-peaks audit table is incomplete")
    if (selected["target"] <= 0).any():
        raise RuntimeError("precursor/padding selected as an intervention")
    expected_roles = {
        "role_confounder": "confounder_only",
        "role_identity": "identity_only",
    }
    for selector, role in expected_roles.items():
        values = selected.loc[selected["selector"] == selector, "target_role"]
        if len(values) and not values.eq(role).all():
            raise RuntimeError(f"{selector} selected a non-{role} token")

    output = {
        "status": "noise_v3_s1a_matrix_validation_passed",
        "formal_queries": int(report["queries"]),
        "paired_cells": int(len(paired)),
        "cell_counts": cell_counts,
        "all_selectors_and_doses_present": True,
        "same_query_set_within_selector_across_doses": True,
        "same_target_and_controls_reused_across_doses": True,
        "precursor_protection": True,
        "role_direction_controls_valid": True,
        "strict_same_role_sensitivity_present": True,
        "shared_eligibility_comparison_present": True,
    }
    destination = args.output_dir / "matrix_validation.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
