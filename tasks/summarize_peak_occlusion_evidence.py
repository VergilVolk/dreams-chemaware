"""Summarize supporting and misleading peaks from the case-level occlusion audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/validation/mass_dense_peak_occlusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def first_number(value):
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return float(parsed[0]) if parsed else np.nan


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input_dir / "occlusion_results.csv")
    cases = pd.read_csv(args.input_dir / "selected_cases.csv")
    singles = results.loc[results["variant_kind"] == "single_peak"].copy()
    singles["peak_mz"] = singles["removed_mz"].map(first_number)
    singles["peak_relative_intensity"] = singles["removed_relative_intensity"].map(first_number)
    case_lookup = cases.set_index("pair_id")
    rows = []
    for (model, case_id), subset in singles.groupby(["model", "case_id"]):
        supporting = subset.loc[subset["margin_drop"].idxmax()]
        misleading = subset.loc[subset["margin_drop"].idxmin()]
        case = case_lookup.loc[int(supporting["pair_id"])]
        random_values = results.loc[
            (results["model"] == model)
            & (results["case_id"] == case_id)
            & (results["variant_kind"] == "random_fraction"),
            "margin_drop",
        ]
        top_fraction = results.loc[
            (results["model"] == model)
            & (results["case_id"] == case_id)
            & (results["variant_kind"] == "top_intensity_fraction"),
            "margin_drop",
        ]
        rows.append({
            "model": model,
            "case_id": int(case_id),
            "group": supporting["group"],
            "pair_id": int(supporting["pair_id"]),
            "ik14": case["ik14"],
            "query_formula": case.get("query_formula", ""),
            "negative_formula": case.get("negative_formula", ""),
            "same_formula": case.get("same_formula", ""),
            "scaffold_relation": case.get("scaffold_relation", ""),
            "mces": case.get("mces", np.nan),
            "original_margin": case["official_margin"] if model == "official_finetuned" else case["raw_margin"],
            "top_supporting_peak_mz": supporting["peak_mz"],
            "top_supporting_neutral_loss": case["query_precursor_mz"] - supporting["peak_mz"],
            "top_supporting_peak_relative_intensity": supporting["peak_relative_intensity"],
            "top_supporting_peak_margin_drop": supporting["margin_drop"],
            "top_misleading_peak_mz": misleading["peak_mz"],
            "top_misleading_neutral_loss": case["query_precursor_mz"] - misleading["peak_mz"],
            "top_misleading_peak_relative_intensity": misleading["peak_relative_intensity"],
            "top_misleading_peak_margin_drop": misleading["margin_drop"],
            "top_intensity_fraction_margin_drop": float(top_fraction.iloc[0]),
            "random_fraction_margin_drop_median": float(random_values.median()),
            "random_fraction_margin_drop_q25": float(random_values.quantile(0.25)),
            "random_fraction_margin_drop_q75": float(random_values.quantile(0.75)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.input_dir / "case_peak_evidence.csv", index=False)

    official = summary.loc[summary["model"] == "official_finetuned"]
    group_summary = official.groupby("group").agg(
        n_cases=("case_id", "count"),
        median_top_peak_drop=("top_supporting_peak_margin_drop", "median"),
        median_top_peak_intensity=("top_supporting_peak_relative_intensity", "median"),
        median_top_fraction_drop=("top_intensity_fraction_margin_drop", "median"),
        median_random_fraction_drop=("random_fraction_margin_drop_median", "median"),
    ).reset_index()
    group_summary.to_csv(args.input_dir / "case_peak_group_summary.csv", index=False)
    report = {
        "status": "case_peak_evidence_summary",
        "n_cases": int(official["case_id"].nunique()),
        "official_group_summary": group_summary.to_dict("records"),
        "rule_expansion_gate": (
            "No peak or neutral-loss value is promoted to a chemical rule unless the "
            "same evidence recurs in multiple molecule-disjoint cases, survives random "
            "mask controls, and links to a reproducible embedding direction."
        ),
    }
    (args.input_dir / "evidence_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
