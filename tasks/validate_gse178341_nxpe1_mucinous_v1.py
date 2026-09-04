#!/usr/bin/env python3
"""Fail-closed validator for the preregistered GSE178341 patient-pseudobulk audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_H5_SHA256 = "f435bb2651ff5297d0c24a99daf58850ed67ae1ed6c5ef05fad48fa3f0186670"
FIXED = {"NXPE1", "MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2", "GNE", "NANS", "CMAS", "SLC35A1", "CASD1", "SIAE"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/GSE178341_mucinous_secretory_audit/nxpe1_mucinous_patient_pseudobulk_v1"),
    )
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "gse178341_nxpe1_mucinous_patient_pseudobulk_complete" or not report.get("formal"):
        raise RuntimeError("status/formal gate failed")
    if report["patients"] != {"pure_mucinous": 6, "pure_adenocarcinoma": 53}:
        raise RuntimeError(f"patient counts changed: {report['patients']}")
    if not set(report["available_fixed_genes"]) <= FIXED or "NXPE1" not in report["available_fixed_genes"]:
        raise RuntimeError("fixed gene panel changed or NXPE1 unavailable")
    if report["provenance"]["h5_sha256"] != EXPECTED_H5_SHA256:
        raise RuntimeError("H5 provenance changed")
    primary = report["primary_NXPE1"]
    if len(primary) != 2 or {row["cohort"] for row in primary} != {"all_pure_tumours", "right_colon_MMR_stratified"}:
        raise RuntimeError("primary NXPE1 cohorts changed")
    matched = report["matched_NXPE1"]
    if matched["n_cases"] != 6 or len(matched["per_case"]) != 6:
        raise RuntimeError("matched mucinous case count changed")
    required_files = {
        "patient_gene_pseudobulk.csv",
        "broad_epithelial_patient_matrix.csv",
        "goblet_family_patient_matrix.csv",
        "fixed_gene_results.csv",
        "matched_results.csv",
        "nxpe1_leave_one_case_out.csv",
        "nxpe1_regression_sensitivity.csv",
        "nxpe1_mucinous_patient_audit.png",
    }
    missing = sorted(name for name in required_files if not (args.output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"missing output files: {missing}")
    print(f"[validate] PASS patients={report['patients']} genes={len(report['available_fixed_genes'])}")


if __name__ == "__main__":
    main()
