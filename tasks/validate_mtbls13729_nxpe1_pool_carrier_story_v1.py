#!/usr/bin/env python3
"""Fail-closed consistency audit for the MTBLS13729 Neu5Ac/NXPE1 story.

This validator does not estimate a new biological effect.  It checks that the
machine-readable results supporting the manuscript still satisfy the frozen
pool--donor--carrier--destination interpretation and its claim boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULTS = {
    "donor": ROOT / "data/mtbls13729/sialic_donor_decoupling_v1/report.json",
    "nxpe1": ROOT
    / "data/external/TCGA_COADREAD_Xena_20260830/"
    "nxpe1_free_donor_v3_secretory/report.json",
    "sensitivity": ROOT
    / "data/external/TCGA_COADREAD_Xena_20260830/"
    "nxpe1_secretory_sensitivity_v1/report.json",
    "gse": ROOT / "data/external/GSE236696/nxpe1_secretory_epithelial_v2/report.json",
    "oacetyl": ROOT / "data/mtbls13729/oacetyl_neu5ac_like_v2/report.json",
    "pxd": ROOT
    / "data/external/PXD055865_2026_MUC2/source_data_audit_v1/report.json",
    "output": ROOT / "data/mtbls13729/nxpe1_pool_carrier_story_v1/report.json",
}


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def require(condition: bool, message: str, checks: dict[str, bool]) -> None:
    checks[message] = bool(condition)
    if not condition:
        raise RuntimeError(f"story consistency check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=Path, default=default)
    args = parser.parse_args()

    paths = {key: getattr(args, key).resolve() for key in DEFAULTS}
    donor = load(paths["donor"])
    nxpe1 = load(paths["nxpe1"])
    sensitivity = load(paths["sensitivity"])
    gse = load(paths["gse"])
    oacetyl = load(paths["oacetyl"])
    pxd = load(paths["pxd"])

    checks: dict[str, bool] = {}

    free = donor["node_summaries"]["free_neu5ac"]
    cmp_node = donor["node_summaries"]["cmp_neu5ac"]
    udp = donor["node_summaries"]["udp_glcnac"]
    contrasts = {
        item["contrast"]: item
        for item in donor["pre_specified_patient_level_contrasts"]
    }
    require(free["identity_level"] == "Level 1", "free Neu5Ac remains Level 1", checks)
    require(free["n_pairs"] == 10, "free Neu5Ac uses 10 Rmu pairs", checks)
    require(free["positive_pairs"] == 10, "free Neu5Ac is positive in 10 of 10 pairs", checks)
    require(close(free["mean_paired_log2_delta"], 2.2486354683399297), "free Neu5Ac effect is frozen", checks)
    require(cmp_node["identity_level"] == "Level 2", "CMP-Neu5Ac remains Level 2", checks)
    require(cmp_node["paired_log_ttest_p"] > 0.05, "CMP-Neu5Ac is not nominally increased", checks)
    require(udp["paired_log_ttest_p"] > 0.05, "UDP-GlcNAc is not nominally increased", checks)
    for name in ("free_neu5ac_minus_cmp_neu5ac", "free_neu5ac_minus_udp_glcnac"):
        item = contrasts[name]
        require(item["bootstrap_mean_95ci"][0] > 0, f"{name} bootstrap lower bound is positive", checks)
        require(item["wilcoxon_holm_p"] <= 0.05, f"{name} Holm-Wilcoxon passes", checks)

    primary = nxpe1["primary_nxpe1"]
    paired = nxpe1["units"]["tpm"]["locked_legacy_371"]["nxpe1_paired_tumour_normal"]
    require(primary["analysis_scope"] == "locked_legacy_371", "NXPE1 primary scope remains locked", checks)
    require(primary["lineage_beta"] > 0 and primary["lineage_p"] < 0.001, "NXPE1 mucinous lineage effect is positive", checks)
    require(primary["msi_lineage_beta"] > 0 and primary["msi_lineage_p"] < 0.01, "NXPE1 survives MSI adjustment", checks)
    require(primary["secretory_lineage_p"] > 0.05, "NXPE1 is absorbed by the secretory program", checks)
    require(primary["secretory_msi_lineage_p"] > 0.05, "NXPE1 remains absorbed after MSI adjustment", checks)
    require(paired["n"] == 50 and paired["tumour_lower"] == 47, "general CRC has 47 of 50 paired NXPE1 decreases", checks)
    require(paired["mean_tumour_minus_normal_log_expression"] < 0, "general CRC NXPE1 paired mean is negative", checks)
    require(nxpe1["nxpe1_direction_consistent_across_units"] is True, "NXPE1 direction agrees across TPM and FPKM-UQ", checks)

    for unit in ("tpm", "fpkm_uq"):
        unit_result = sensitivity["units"][unit]
        require(unit_result["leave_one_out"]["all_p_gt_0_05"] is True, f"{unit} leave-one-out secretory adjustment is nonsignificant", checks)
        require(unit_result["pair_marker"]["significant_after_adjustment"] == [], f"{unit} marker-pair sensitivity has no significant NXPE1 effect", checks)

    broad_nxpe1 = next(
        item
        for item in gse["paired_results"]
        if item["compartment"] == "broad_epithelial" and item["gene"] == "NXPE1"
    )
    require(gse["patients"] == 6, "GSE236696 contains six paired patients", checks)
    require(gse["feature_availability_samples"]["MUC2"] == 0, "GSE236696 deposited feature index lacks MUC2", checks)
    require(broad_nxpe1["n_pairs"] == 6 and broad_nxpe1["positive_pairs"] == 0, "GSE236696 NXPE1 is lower in all six tumour pairs", checks)
    require(broad_nxpe1["mean_delta"] < 0, "GSE236696 NXPE1 mean delta is negative", checks)
    require(broad_nxpe1["exact_sign_flip_p"] >= 0.05, "GSE236696 NXPE1 is retained as low-power direction support", checks)

    oac_rows = oacetyl["primary_rmu_results"]
    require(oacetyl["phenotype_blind_discovery"]["frozen_rt_features"] == 2, "O-acetyl-like audit freezes exactly two RT peaks", checks)
    require([item["samples"] for item in oac_rows] == [50, 54], "O-acetyl-like peaks retain 50 and 54 sample support", checks)
    require([item["ms2_spectra"] for item in oac_rows] == [47, 56], "O-acetyl-like peaks retain 47 and 56 MS2 spectra", checks)
    require(all(item["complete_exact_sign_flip_bh_q"] > 0.05 for item in oac_rows), "neither O-acetyl-like peak is Rmu significant", checks)
    require(all(abs(item["spearman_rho"]) < 0.25 for item in oacetyl["free_neu5ac_correlations"]), "O-acetyl-like peaks do not track free Neu5Ac", checks)

    pxd_audit = pxd["figure_level_audit"]
    require(pxd_audit["independent_colorectal_tumour_patients"] == 2, "PXD055865 has two independent tumour patients", checks)
    require(pxd_audit["healthy_donors"] == 1, "PXD055865 has one healthy donor", checks)
    require("not an independent abundance replication" in pxd_audit["interpretation"], "PXD055865 remains structural context rather than abundance replication", checks)

    interpretation = {
        "supported": [
            "same-patient free Neu5Ac pool expansion",
            "free-pool to activated-donor decoupling",
            "mucinous-relative NXPE1 enrichment linked to a distributed secretory-mucin carrier state",
            "general tumour-versus-normal NXPE1 suppression",
            "carrier/destination heterogeneity rather than a single global sialylation program",
        ],
        "not_supported": [
            "NXPE1 as an independent mucinous driver",
            "causal mediation by the secretory program",
            "free Neu5Ac as the uniquely established in-vivo NXPE1 substrate",
            "bulk mono-O-acetyl-Neu5Ac pool expansion",
            "O-acetylation flux, enzyme activity, or positional-isomer identity",
        ],
        "minimal_model": "free pool expansion + activated donor non-expansion + secretory carrier-linked NXPE1 capacity + heterogeneous glycan destination remodelling",
    }

    output = {
        "status": "mtbls13729_nxpe1_pool_carrier_story_validation_passed",
        "formal": True,
        "checks": checks,
        "n_checks": len(checks),
        "interpretation": interpretation,
        "claim_limit": "Cross-platform consistency audit only. It does not create an independent abundance replication, identify an O-acetyl positional isomer, or establish enzyme activity, flux, mediation, or causality.",
        "provenance": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for key, path in paths.items()
            if key != "output"
        },
    }

    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
