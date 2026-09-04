#!/usr/bin/env python3
"""Fail-closed validation for the GSE178341 composition diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    required = {
        "report.json",
        "patient_epithelial_composition.csv",
        "patient_composition_expression_matrix.csv",
        "composition_group_results.csv",
        "composition_matched_results.csv",
        "composition_adjusted_models.csv",
        "composition_expression_correlations.csv",
        "epithelial_composition_diagnostic.png",
    }
    missing = sorted(name for name in required if not (args.output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "gse178341_epithelial_composition_diagnostic_complete":
        raise RuntimeError("status mismatch")
    if report.get("formal") is not False or report.get("analysis_type") != "post-result patient-level diagnostic":
        raise RuntimeError("post-result status was not preserved")
    composition = pd.read_csv(args.output_dir / "patient_epithelial_composition.csv")
    if len(composition) != 59 or composition["PID"].duplicated().any():
        raise RuntimeError("patient composition is not one row per 59 patients")
    if not composition["goblet_fraction"].between(0, 1).all():
        raise RuntimeError("invalid goblet fractions")
    models = pd.read_csv(args.output_dir / "composition_adjusted_models.csv")
    if len(models) != 21 or set(models["model"]) != {"histology_only", "plus_goblet_fraction", "plus_goblet_right_mmr"}:
        raise RuntimeError("expected 7 endpoints x 3 models")
    if (models["condition_number"] > 100).any():
        raise RuntimeError("unstable composition-adjusted model")
    if "not confirmatory" not in report["claim_limit"]:
        raise RuntimeError("claim boundary missing")
    print(f"[validate_gse178341_epithelial_composition_v1] PASS patients={len(composition)} models={len(models)}")


if __name__ == "__main__":
    main()
