"""Add independent raw-UMI and patient-proteomics evidence to the biology ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v9_final"
RAW_PRIMARY = (
    ROOT / "data/external/GSE178341_mucinous_secretory_audit/"
    "nxpe1_mucinous_patient_pseudobulk_v1/report.json"
)
RAW_SOURCE = (
    ROOT / "data/external/GSE178341_mucinous_secretory_audit/"
    "sialic_cell_source_patient_pseudobulk_v1/report.json"
)
PROTEOMICS = (
    ROOT / "data/external/GSE178341_mucinous_secretory_audit/"
    "independent_proteomics_fixed_panel_v1/result.json"
)
RAW_DOC = ROOT / "docs/GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md"
PROTEOMICS_DOC = ROOT / "docs/MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_AUDIT_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v10_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    required = [
        V9 / "mechanism_paper_completion_audit_v9.csv",
        V9 / "report.json",
        RAW_PRIMARY,
        RAW_SOURCE,
        PROTEOMICS,
        RAW_DOC,
        PROTEOMICS_DOC,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = json.loads(RAW_PRIMARY.read_text(encoding="utf-8"))
    source = json.loads(RAW_SOURCE.read_text(encoding="utf-8"))
    protein = json.loads(PROTEOMICS.read_text(encoding="utf-8"))
    if raw.get("status") != "gse178341_nxpe1_mucinous_patient_pseudobulk_complete":
        raise RuntimeError("raw single-cell primary audit is incomplete")
    if source.get("status") != "gse178341_sialic_cell_source_patient_pseudobulk_complete":
        raise RuntimeError("raw single-cell source audit is incomplete")
    if protein.get("status") != "independent_mucinous_crc_proteomics_fixed_panel_complete":
        raise RuntimeError("independent proteomics audit is incomplete")
    if raw["patients"] != {"pure_mucinous": 6, "pure_adenocarcinoma": 53}:
        raise RuntimeError("raw single-cell patient count changed")
    expected_support = {"Epi|secretory_carrier", "Epi|cmp_neu5ac_capacity"}
    if set(source["supporting_endpoints"]) != expected_support:
        raise RuntimeError("cell-source support set changed")
    if raw["gates"]["nxpe1_primary_support"] is not False:
        raise RuntimeError("NXPE1 was silently upgraded")
    if any(row["protein_specific_support"] for row in protein["proteins"]):
        raise RuntimeError("proteomics fixed-panel result was silently upgraded")
    if any(row["orthogonal_support"] for row in protein["modules"]):
        raise RuntimeError("proteomics module result was silently upgraded")

    ledger = pd.read_csv(V9 / "mechanism_paper_completion_audit_v9.csv")
    if len(ledger) != 23 or ledger.gate_id.tolist() != [f"G{i:02d}" for i in range(1, 24)]:
        raise RuntimeError("unexpected v9 ledger")
    additions = pd.DataFrame(
        [
            {
                "gate_id": "G24",
                "domain": "independent_patient_raw_transcript_cell_source",
                "status": "PASS_CONTEXT",
                "evidence": (
                    "GSE178341 raw UMI patient pseudobulks (6 mucinous, 53 conventional) "
                    "support epithelial secretory-carrier and CMP-Neu5Ac-capacity modules; "
                    "AGR2 and SLC35A1 pass fixed-panel correction. NXPE1 does not pass its "
                    "independent primary gate and attenuates after secretory adjustment."
                ),
                "claim_enabled": (
                    "Independent host-cell context supports selective epithelial secretory "
                    "folding and Golgi donor-transport capacity rather than uniform pathway activation."
                ),
                "claim_forbidden": (
                    "RNA does not identify the biochemical source of free Neu5Ac, enzyme activity, "
                    "microbial contribution, glycan destination, flux, or causal mediation."
                ),
                "next_action": (
                    "Prioritise same-sample linkage-aware glycan readout and targeted Neu5Ac/CMP-Neu5Ac "
                    "measurement; do not add more bulk RNA cohorts."
                ),
                "priority": "P0",
            },
            {
                "gate_id": "G25",
                "domain": "independent_patient_proteomics_fixed_panel",
                "status": "NEGATIVE_RESULT",
                "evidence": (
                    "In 15 mucinous versus 15 conventional CRCs, neither frozen protein module "
                    "passes multiplicity control. AGR2, GNE and NANS have stable positive leave-one-"
                    "patient-out directions, but all bootstrap intervals cross zero; AGR2 significance "
                    "is analysis-scale sensitive."
                ),
                "claim_enabled": (
                    "The negative fixed-panel result constrains the model to selective, underpowered "
                    "protein directions and argues against uniform pathway activation."
                ),
                "claim_forbidden": (
                    "The cohort is not an independent Neu5Ac abundance replication and does not confirm "
                    "the secretory or sialic-handling protein modules."
                ),
                "next_action": (
                    "Use targeted proteins only if a future same-sample glycan/Neu5Ac study is available; "
                    "do not reselect a favourable protein panel."
                ),
                "priority": "P1",
            },
        ]
    )
    ledger = pd.concat([ledger, additions], ignore_index=True)
    if len(ledger) != 25 or ledger.gate_id.tolist() != [f"G{i:02d}" for i in range(1, 26)]:
        raise RuntimeError("v10 gate construction failed")

    OUT.mkdir(parents=True, exist_ok=False)
    ledger_path = OUT / "mechanism_paper_completion_audit_v10.csv"
    ledger.to_csv(ledger_path, index=False)

    raw_genes = {
        row["gene"]: row
        for row in raw["all_fixed_gene_results"]
        if row["compartment"] == "broad_epithelial" and row["cohort"] == "all_pure_tumours"
    }
    source_endpoints = {
        f"{row['compartment']}|{row['module']}": row for row in source["fixed_endpoints"]
    }
    protein_rows = {row["name"]: row for row in protein["proteins"]}
    report = json.loads((V9 / "report.json").read_text(encoding="utf-8"))
    report.update(
        {
            "status": "mtbls13729_mechanism_paper_completion_audit_v10_complete",
            "gates": 25,
            "status_counts": {
                str(key): int(value) for key, value in ledger.status.value_counts().items()
            },
            "independent_patient_raw_transcript_context": {
                "status": "PASS_CONTEXT",
                "patients": raw["patients"],
                "supporting_endpoints": source["supporting_endpoints"],
                "epithelial_secretory_delta": source_endpoints[
                    "Epi|secretory_carrier"
                ]["primary_mean_difference"],
                "epithelial_secretory_q": source_endpoints[
                    "Epi|secretory_carrier"
                ]["primary_BH_q_across_7"],
                "epithelial_cmp_capacity_delta": source_endpoints[
                    "Epi|cmp_neu5ac_capacity"
                ]["primary_mean_difference"],
                "epithelial_cmp_capacity_q": source_endpoints[
                    "Epi|cmp_neu5ac_capacity"
                ]["primary_BH_q_across_7"],
                "AGR2_delta": raw_genes["AGR2"]["mean_difference"],
                "AGR2_q": raw_genes["AGR2"]["BH_q_within_fixed_panel"],
                "SLC35A1_delta": raw_genes["SLC35A1"]["mean_difference"],
                "SLC35A1_q": raw_genes["SLC35A1"]["BH_q_within_fixed_panel"],
                "NXPE1_primary_support": raw["gates"]["nxpe1_primary_support"],
                "host_release_mechanism_supported": False,
                "biochemical_source_established": False,
            },
            "independent_patient_proteomics_context": {
                "status": "NEGATIVE_RESULT",
                "patients": protein["cohort"],
                "module_support": False,
                "AGR2_delta": protein_rows["AGR2"]["mc_minus_ac"],
                "AGR2_q": protein_rows["AGR2"]["permutation_bh_q_across_8"],
                "AGR2_original_scale_p": protein_rows["AGR2"][
                    "published_raw_scale_context"
                ]["welch_t_p_on_untransformed_abundance"],
                "independent_neu5ac_abundance_replication": False,
            },
            "biology_package_A_position": (
                "algorithm-enabled, evidence-calibrated clinical discovery with independent patient-level "
                "raw-transcript cell-source context and an explicitly negative fixed-panel proteomics audit"
            ),
            "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
            "claim_limit": (
                "The combined evidence supports a discovery-level free-pool-to-donor/destination "
                "decoupling with selective epithelial secretory and Golgi donor-transport capacity. "
                "It does not provide independent Neu5Ac abundance replication, same-sample glycan "
                "destination, biochemical source, flux, enzyme causality, or therapeutic validation."
            ),
        }
    )
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# MTBLS13729 mechanism-paper completion audit v10",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        "| Gate | Domain | Status | Evidence | Next action |",
        "|---|---|---|---|---|",
    ]
    for row in ledger.to_dict("records"):
        lines.append(
            f"| {row['gate_id']} | {row['domain']} | {row['status']} | "
            f"{row['evidence']} | {row['next_action']} |"
        )
    lines.extend(["", "## Claim boundary", "", report["claim_limit"], ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
