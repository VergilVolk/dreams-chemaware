#!/usr/bin/env python
"""Fail-closed validation of the frozen current-GDC NXPE1 mechanism audit."""

from __future__ import annotations

import json
import math
from pathlib import Path


REPORT = Path(
    "data/external/TCGA_COADREAD_Xena_20260830/"
    "nxpe1_free_donor_v3_secretory/report.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    require(
        report.get("status") == "tcga_nxpe1_free_donor_mechanism_audit_complete",
        "unexpected report status",
    )

    tpm = report["units"]["tpm"]["locked_legacy_371"]
    require(tpm["primary_tumours"] == 371, "locked cohort must contain 371 tumours")
    require(tpm["mucinous"] == 42, "locked cohort must contain 42 mucinous tumours")
    require(tpm["conventional"] == 329, "locked cohort must contain 329 conventional tumours")

    primary = report["primary_nxpe1"]
    sensitivity = report["fpkm_uq_sensitivity_nxpe1"]
    for name, result in (("TPM", primary), ("FPKM-UQ", sensitivity)):
        require(result["lineage_beta"] > 0, f"{name}: lineage coefficient is not positive")
        require(result["lineage_p"] < 0.01, f"{name}: lineage coefficient is not significant")
        require(result["msi_lineage_beta"] > 0, f"{name}: MSI-lineage coefficient is not positive")
        require(result["msi_lineage_p"] < 0.01, f"{name}: MSI-lineage coefficient is not significant")
        require(
            result["secretory_lineage_p"] > 0.05,
            f"{name}: secretory-adjusted coefficient unexpectedly remains significant",
        )
        require(
            result["secretory_msi_lineage_p"] > 0.05,
            f"{name}: secretory+MSI-adjusted coefficient unexpectedly remains significant",
        )

    paired_tpm = tpm["nxpe1_paired_tumour_normal"]
    paired_fpkm = report["units"]["fpkm_uq"]["locked_legacy_371"][
        "nxpe1_paired_tumour_normal"
    ]
    for name, paired in (("TPM", paired_tpm), ("FPKM-UQ", paired_fpkm)):
        require(paired["n"] == 50, f"{name}: expected 50 tumour-normal pairs")
        require(paired["tumour_lower"] == 47, f"{name}: expected 47/50 tumour-lower pairs")
        require(
            paired["mean_tumour_minus_normal_log_expression"] < 0,
            f"{name}: expected lower general-CRC tumour expression",
        )
        require(math.isfinite(paired["sign_p"]) and paired["sign_p"] < 1e-8, f"{name}: paired sign test failed")

    require(report["nxpe1_direction_consistent_across_units"] is True, "unit direction mismatch")
    print(
        json.dumps(
            {
                "status": "tcga_nxpe1_free_donor_mechanism_validation_passed",
                "locked_tumours": 371,
                "mucinous": 42,
                "conventional": 329,
                "lineage_beta_tpm": primary["lineage_beta"],
                "lineage_p_tpm": primary["lineage_p"],
                "secretory_adjusted_beta_tpm": primary["secretory_lineage_beta"],
                "secretory_adjusted_p_tpm": primary["secretory_lineage_p"],
                "paired_tumour_lower": paired_tpm["tumour_lower"],
                "paired_n": paired_tpm["n"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
