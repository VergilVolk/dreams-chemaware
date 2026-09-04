#!/usr/bin/env python3
"""Freeze the no-new-wet-lab MTBLS13729 biology release candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/biology_package_a_release_v1"
FILES = {
    "donor_report": ROOT / "data/mtbls13729/sialic_donor_decoupling_v1/report.json",
    "raw_primary_report": ROOT / "data/external/GSE178341_mucinous_secretory_audit/nxpe1_mucinous_patient_pseudobulk_v1/report.json",
    "raw_source_report": ROOT / "data/external/GSE178341_mucinous_secretory_audit/sialic_cell_source_patient_pseudobulk_v1/report.json",
    "composition_report": ROOT / "data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/report.json",
    "proteomics_report": ROOT / "data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/result.json",
    "completion_report": ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v10_final/report.json",
    "completion_ledger": ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v10_final/mechanism_paper_completion_audit_v10.csv",
    "main_figure_png": ROOT / "data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final/neu5ac_evidence_convergence_figure_v3.png",
    "main_figure_pdf": ROOT / "data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final/neu5ac_evidence_convergence_figure_v3.pdf",
    "manuscript_results": ROOT / "docs/MTBLS13729_BIOLOGY_MANUSCRIPT_RESULTS_V2_20260830.md",
    "publication_decision": ROOT / "docs/MTBLS13729_BIOLOGY_PUBLICATION_DECISION_20260831.md",
    "raw_umi_audit": ROOT / "docs/GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md",
    "composition_audit": ROOT / "docs/GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_RESULT_20260831.md",
    "proteomics_audit": ROOT / "docs/MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_AUDIT_20260831.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str) -> dict:
    return json.loads(FILES[name].read_text(encoding="utf-8"))


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    for name, path in FILES.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{name}: {path}")
    donor = load("donor_report")
    raw = load("raw_primary_report")
    source = load("raw_source_report")
    composition = load("composition_report")
    protein = load("proteomics_report")
    completion = load("completion_report")
    if completion.get("biology_package_A_ready") is not True:
        raise RuntimeError("completion ledger does not authorize Package A")
    if completion.get("package_B_ready") is not False or completion.get("package_C_ready") is not False:
        raise RuntimeError("higher packages were silently upgraded")
    if set(source.get("supporting_endpoints", [])) != {"Epi|secretory_carrier", "Epi|cmp_neu5ac_capacity"}:
        raise RuntimeError("raw source support set changed")
    if raw["gates"]["nxpe1_primary_support"] is not False:
        raise RuntimeError("NXPE1 was silently upgraded")
    if composition["formal"] is not False or composition["analysis_type"] != "post-result patient-level diagnostic":
        raise RuntimeError("composition diagnostic status changed")
    if any(item["protein_specific_support"] for item in protein["proteins"]):
        raise RuntimeError("proteomics was silently upgraded")

    claims = pd.DataFrame([
        {
            "claim_id": "B01", "status": "PASS_DISCOVERY", "claim": "Free Neu5Ac is increased in Rmu versus matched normal.",
            "evidence": "10/10 patient pairs positive; mean +2.249 log2.",
            "boundary": "Same cohort; static pool size; not independent replication or flux.",
        },
        {
            "claim_id": "B02", "status": "PASS_DISCOVERY", "claim": "Free-pool increase exceeds measured donor/precursor changes.",
            "evidence": "Free-minus-CMP +1.693 and free-minus-UDP +1.922 log2; Holm p=0.0273 each.",
            "boundary": "CMP-Neu5Ac is Level 2; source and destination remain unresolved.",
        },
        {
            "claim_id": "B03", "status": "PASS_CONTEXT", "claim": "Independent mucinous tumour epithelium shows selective secretory-folding and Golgi donor-transport capacity.",
            "evidence": "AGR2 and SLC35A1 pass the frozen raw-UMI panel and retain positive HC3 intervals after goblet-fraction adjustment.",
            "boundary": "RNA capacity is not transport flux, protein activity, source or destination.",
        },
        {
            "claim_id": "B04", "status": "PASS_CONTEXT", "claim": "Goblet-lineage composition contributes to the mucinous transcript state.",
            "evidence": "Mean goblet-lineage fraction 0.284 versus 0.152; broad secretory/MUC2 signals attenuate after adjustment.",
            "boundary": "Post-result and dissociation-sensitive; not causal mediation.",
        },
        {
            "claim_id": "B05", "status": "NEGATIVE_RESULT", "claim": "NXPE1 is not supported as an independent mucinous driver.",
            "evidence": "Raw-UMI fixed-panel q=0.229; matched 4/6; attenuates with secretory/goblet state.",
            "boundary": "Does not exclude carrier-specific O-acetylation in unmeasured contexts.",
        },
        {
            "claim_id": "B06", "status": "NEGATIVE_RESULT", "claim": "Host NEU1/NEU3 upregulation is not supported as the free-Neu5Ac release mechanism.",
            "evidence": "Independent epithelial and myeloid release modules do not pass and are directionally negative.",
            "boundary": "Does not test enzyme activity, microbial sialidases or glycan-bound release directly.",
        },
        {
            "claim_id": "B07", "status": "NEGATIVE_RESULT", "claim": "Independent fixed-panel proteomics does not confirm uniform secretory or sialic-pathway activation.",
            "evidence": "Both modules fail multiplicity control; AGR2/GNE/NANS directions are stable but intervals cross zero.",
            "boundary": "Supporting context only; not Neu5Ac abundance replication.",
        },
        {
            "claim_id": "B08", "status": "MISSING", "claim": "Independent subtype-resolved Neu5Ac abundance replication.",
            "evidence": "No eligible public independent Rmu tissue metabolomics cohort was found.",
            "boundary": "Cannot be replaced by RNA, proteins, glycomics or general-CRC spatial gradients.",
        },
        {
            "claim_id": "B09", "status": "MISSING", "claim": "Full 13,155-feature discovery-space confirmation.",
            "evidence": "No feature passes the full-space exact FDR10 audit.",
            "boundary": "Candidate-panel FDR and targeted follow-up remain selective discovery evidence.",
        },
        {
            "claim_id": "B10", "status": "MISSING", "claim": "Causal metabolic mechanism or therapeutic target.",
            "evidence": "No independent standard/spike-in, isotope tracing, perturbation, rescue or functional model.",
            "boundary": "Use remodeling, abundance program and mechanistic context; do not use drives or flux reprogramming.",
        },
    ])
    if claims["claim_id"].tolist() != [f"B{i:02d}" for i in range(1, 11)]:
        raise RuntimeError("claim ledger ordering failed")

    OUT.mkdir(parents=True, exist_ok=False)
    claims.to_csv(OUT / "biology_claim_ledger_v1.csv", index=False)
    manifest = pd.DataFrame([
        {
            "artifact": name,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in FILES.items()
    ])
    manifest.to_csv(OUT / "artifact_manifest_v1.csv", index=False)
    report = {
        "status": "mtbls13729_biology_package_a_release_v1_complete",
        "formal": True,
        "package": "A",
        "position": "algorithm-enabled, evidence-calibrated clinical discovery",
        "primary_phenomenon": "mucinous-relative free-Neu5Ac pool expansion with selective epithelial secretory/transport capacity and pool-to-donor/destination decoupling",
        "claim_counts": {str(k): int(v) for k, v in claims["status"].value_counts().items()},
        "package_A_ready": True,
        "package_B_ready": False,
        "package_C_ready": False,
        "next_cross_grade_action": "same-method Neu5Ac standard/spike-in plus same-sample linkage-aware glycan readout; independent Rmu tissue metabolomics remains the essential external replication",
        "claim_limit": completion["claim_limit"],
        "manifest_sha256": sha256(OUT / "artifact_manifest_v1.csv"),
        "claim_ledger_sha256": sha256(OUT / "biology_claim_ledger_v1.csv"),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MTBLS13729 biology Package A release v1",
        "",
        f"**Position:** {report['position']}",
        "",
        f"**Primary phenomenon:** {report['primary_phenomenon']}",
        "",
        "This package is ready for a no-new-wet-lab, evidence-calibrated application paper. It is not a causal-mechanism package.",
        "",
        "## Claim ledger",
        "",
        claims.to_markdown(index=False),
        "",
        "## Claim boundary",
        "",
        report["claim_limit"],
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
