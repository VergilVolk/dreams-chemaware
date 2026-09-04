#!/usr/bin/env python
"""Fail-closed validation for the GSE178341 portal sensitivity artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/GSE178341_mucinous_secretory_audit/scp_portal_normalized_sensitivity_v1"
GENES = ("NXPE1", "MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2", "GNE", "NANS", "CMAS", "SLC35A1", "CASD1", "SIAE")


def main() -> None:
    report_path = OUT / "report.json"
    table_path = OUT / "fixed_panel_patient_results.csv"
    patient_path = OUT / "patient_normalized_expression.csv"
    for path in (report_path, table_path, patient_path, OUT / "fixed_panel_patient_directions.png"):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)
    patient = pd.read_csv(patient_path)
    if report.get("formal") is not False or "secondary" not in report.get("role", ""):
        raise RuntimeError("portal analysis is not explicitly marked secondary/non-formal")
    if report.get("portal_subsample_cells") != 100_000 or report.get("portal_cells_exactly_joined_to_geo") != 98_438:
        raise RuntimeError("portal/GEO reconciliation counts changed")
    if set(report.get("portal_geo_mismatch_patients", [])) != {"C121", "C144"}:
        raise RuntimeError("portal/GEO mismatch patient set changed")
    if len(table) != 2 * len(GENES) or set(table["gene"]) != set(GENES):
        raise RuntimeError("fixed endpoint panel is incomplete")
    if set(table["compartment"]) != {"epithelial_portal_subsample", "goblet_family_portal_subsample"}:
        raise RuntimeError("unexpected compartment set")
    broad = patient[patient["compartment"].eq("epithelial_portal_subsample")]
    if (broad["histology"].eq("Adenocarcinoma;Mucinous").sum(), broad["histology"].eq("Adenocarcinoma").sum()) != (6, 53):
        raise RuntimeError("patient counts changed")
    nxpe1 = table[(table["compartment"].eq("epithelial_portal_subsample")) & (table["gene"].eq("NXPE1"))].iloc[0]
    if not (nxpe1["mean_difference"] > 0 and nxpe1["permutation_p"] > 0.05 and nxpe1["matched_exact_sign_flip_p"] > 0.05):
        raise RuntimeError("NXPE1 result no longer matches the documented weak/non-significant direction")
    agr2 = table[(table["compartment"].eq("epithelial_portal_subsample")) & (table["gene"].eq("AGR2"))].iloc[0]
    slc = table[(table["compartment"].eq("epithelial_portal_subsample")) & (table["gene"].eq("SLC35A1"))].iloc[0]
    if not (agr2["BH_q_within_compartment"] < 0.05 and slc["BH_q_within_compartment"] < 0.05):
        raise RuntimeError("predefined AGR2/SLC35A1 panel signals are not reproduced")
    if not np.isclose(float(report["NXPE1"]["epithelial"]["mean_difference"]), float(nxpe1["mean_difference"]), atol=1e-12):
        raise RuntimeError("report/CSV NXPE1 mismatch")
    print("[validate_gse178341_scp_portal_sensitivity_v1] PASS")


if __name__ == "__main__":
    main()
