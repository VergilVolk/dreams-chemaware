"""Fail-closed validation of the expanded Noise-v3 S1c top-k matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SELECTORS = {
    "candidate_gradient", "candidate_gradient_r2", "candidate_gradient_r3",
    "candidate_gradient_r4", "candidate_gradient_r5",
    "role_confounder", "role_confounder_r2", "role_confounder_r3",
}
EXPECTED_DOSES = (0.25, 0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s1c_topk_matrix"),
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
    actual = set(map(str, matrix.get("selectors", ())))
    if actual != EXPECTED_SELECTORS:
        raise RuntimeError(
            f"selector mismatch; missing={sorted(EXPECTED_SELECTORS-actual)} "
            f"unexpected={sorted(actual-EXPECTED_SELECTORS)}"
        )
    if not np.allclose(
        sorted(map(float, matrix.get("attenuation_doses", ()))), EXPECTED_DOSES,
    ):
        raise RuntimeError("dose mismatch")
    if report.get("formal") is not True or int(report.get("queries", 0)) < 20_000:
        raise RuntimeError("S1c is not a formal full-graph run")

    required = {
        "query_index", "selector", "selector_family", "selector_rank",
        "attenuation", "target_token", "control_tokens", "target_rank",
    }
    missing = required - set(paired.columns)
    if missing:
        raise RuntimeError(f"paired table misses: {sorted(missing)}")
    if paired.duplicated(["query_index", "selector", "attenuation"]).any():
        raise RuntimeError("duplicate query-selector-dose cell")

    coverage = {}
    for selector in sorted(EXPECTED_SELECTORS):
        frame = paired.loc[paired["selector"] == selector]
        reference = None
        for dose in EXPECTED_DOSES:
            cell = frame.loc[np.isclose(frame["attenuation"], dose)]
            if cell.empty:
                raise RuntimeError(f"empty cell {selector}@{dose}")
            queries = set(map(int, cell["query_index"]))
            reference = queries if reference is None else reference
            if queries != reference:
                raise RuntimeError(f"dose-dependent query set for {selector}")
        invariant = frame.groupby("query_index", sort=False).agg(
            target=("target_token", "nunique"), controls=("control_tokens", "nunique"),
        )
        if (invariant != 1).any().any():
            raise RuntimeError(f"target/control drift across doses for {selector}")
        coverage[selector] = int(len(reference))

    family_rank = selected[["selector", "selector_family", "selector_rank"]].drop_duplicates()
    if family_rank.duplicated("selector").any():
        raise RuntimeError("selector maps to multiple family/rank definitions")
    if (selected["target"] <= 0).any():
        raise RuntimeError("precursor/padding was selected")
    conf = selected.loc[selected["selector_family"] == "role_confounder"]
    if len(conf) and not conf["target_role"].eq("confounder_only").all():
        raise RuntimeError("role-confounder rank selected an invalid role")

    output = {
        "status": "noise_v3_s1c_topk_matrix_validation_passed",
        "formal_queries": int(report["queries"]),
        "paired_cells": int(len(paired)),
        "selector_coverage": coverage,
        "all_8_selectors_x_4_doses_present": True,
        "same_query_set_within_selector_across_doses": True,
        "same_target_and_controls_reused_across_doses": True,
        "role_and_precursor_protection": True,
    }
    destination = args.output_dir / "matrix_validation.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
