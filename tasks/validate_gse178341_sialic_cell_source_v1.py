#!/usr/bin/env python3
"""Fail-closed validator for the GSE178341 sialic-cell-source audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_H5_SHA256 = "f435bb2651ff5297d0c24a99daf58850ed67ae1ed6c5ef05fad48fa3f0186670"
EXPECTED_ENDPOINTS = {
    "Epi|secretory_carrier",
    "Epi|cmp_neu5ac_capacity",
    "Myeloid|cmp_neu5ac_capacity",
    "Epi|glycoconjugate_release",
    "Myeloid|glycoconjugate_release",
    "Epi|salvage_catabolism",
    "Myeloid|salvage_catabolism",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/GSE178341_mucinous_secretory_audit/"
            "sialic_cell_source_patient_pseudobulk_v1"
        ),
    )
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "gse178341_sialic_cell_source_patient_pseudobulk_complete":
        raise RuntimeError("unexpected report status")
    if report.get("formal") is not True:
        raise RuntimeError("formal flag missing")
    if report.get("patients") != {"pure_mucinous": 6, "pure_adenocarcinoma": 53}:
        raise RuntimeError("patient counts changed")
    endpoints = {
        f"{row['compartment']}|{row['module']}" for row in report["fixed_endpoints"]
    }
    if endpoints != EXPECTED_ENDPOINTS or len(report["fixed_endpoints"]) != 7:
        raise RuntimeError(f"fixed endpoint set changed: {endpoints}")
    if report["provenance"]["h5_sha256"] != EXPECTED_H5_SHA256:
        raise RuntimeError("H5 provenance changed")
    for row in report["fixed_endpoints"]:
        if len(row["primary_bootstrap_95ci"]) != 2:
            raise RuntimeError("missing endpoint bootstrap interval")
        if not 0 <= row["primary_BH_q_across_7"] <= 1:
            raise RuntimeError("invalid multiplicity-adjusted q value")
    required = {
        "patient_gene_pseudobulk.csv",
        "epi_patient_matrix.csv",
        "myeloid_patient_matrix.csv",
        "fixed_endpoint_results.csv",
        "matched_endpoint_results.csv",
        "leave_one_case_out.csv",
        "sialic_cell_source_audit.png",
    }
    missing = sorted(name for name in required if not (args.output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"missing output files: {missing}")
    print(
        "[validate_gse178341_sialic_cell_source_v1] PASS "
        f"endpoints={len(endpoints)} support={len(report['supporting_endpoints'])}"
    )


if __name__ == "__main__":
    main()

