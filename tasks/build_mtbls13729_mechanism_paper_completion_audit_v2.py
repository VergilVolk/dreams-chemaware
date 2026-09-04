"""Extend the frozen mechanism-readiness ledger with the hybrid-glycome audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v1"
GLYCAN = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/report.json"
OGLY = ROOT / "data/external/CRC_Oglycomics_PMC9254241_20260830/mucinous_structural_audit_values.csv"
FIGURE = ROOT / "data/mtbls13729/neu5ac_glycan_publication_figure_v2_final/report.json"
DECISION = ROOT / "docs/MTBLS13729_BIOLOGY_PUBLICATION_DECISION_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v2_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (V1 / "mechanism_paper_completion_audit_v1.csv", V1 / "report.json", GLYCAN, OGLY, FIGURE, DECISION):
        if not path.is_file():
            raise FileNotFoundError(path)

    ledger = pd.read_csv(V1 / "mechanism_paper_completion_audit_v1.csv")
    if len(ledger) != 12 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 13)}:
        raise RuntimeError("unexpected v1 completion ledger")

    # Update only claims altered by the branch-resolved audit.
    ledger.loc[ledger.gate_id.eq("G05"), "claim_enabled"] = (
        "Mucinous-relative hybrid mucin glycome is the primary discovery axis."
    )
    ledger.loc[ledger.gate_id.eq("G05"), "claim_forbidden"] = (
        "Do not call this global hypersialylation, carrier-specific incorporation, linkage flux, or enzyme causality."
    )
    ledger.loc[ledger.gate_id.eq("G07"), "evidence"] = (
        "TCGA 42 mucinous versus 329 conventional tumours: donor/transport, secretory mucin and core-3/Sda lineage are positive, while ST6GAL1 and alpha2-3 are negative after lineage adjustment."
    )
    ledger.loc[ledger.gate_id.eq("G07"), "claim_enabled"] = (
        "Branch-resolved transcriptomics supports donor-carrier-core-linkage decoupling."
    )
    ledger.loc[ledger.gate_id.eq("G08"), "evidence"] = (
        "External MUC2 glycopeptide mapping shows spatially distinct, largely low-sialylated/unsialylated tumour glycoforms in three tumours from two patients."
    )

    additions = pd.DataFrame([
        {
            "gate_id": "G13",
            "domain": "abundance_protocol_reconciliation",
            "status": "PASS",
            "evidence": "Locked detection-masked targeted EIC reproduces Rmu 10/10 positive and +1.935 log2; cross-panel log2(EIC+1) is 10/10 and +1.975; discovery matrix is 9/9 and +1.881 because P24 is missing.",
            "claim_enabled": "The locked targeted-EIC estimate is the primary abundance result; the other protocols are sensitivity analyses.",
            "claim_forbidden": "Do not combine n, means or p-values across the three quantification protocols.",
            "next_action": "Use the v2 figure source table and keep the protocol names in every caption.",
            "priority": "P0",
        },
        {
            "gate_id": "G14",
            "domain": "external_structural_glycomics",
            "status": "PASS_WITH_LIMITATION",
            "evidence": "Two authoritative Table-S2 MUC cases have tumour-paired core-2/sLeX gains and alpha2-6 loss; core-3 ranks high among tumours but decreases versus matched normal.",
            "claim_enabled": "Independent structural evidence supports a hybrid mucin glycome and resolves relative-retention versus absolute-loss reference frames.",
            "claim_forbidden": "Do not call n=2 structural data an independent free-Neu5Ac abundance replication or population confirmation.",
            "next_action": "Retain exact patient values/ranks and seek a larger linkage-aware cohort or local assay.",
            "priority": "P0",
        },
        {
            "gate_id": "G15",
            "domain": "glycan_branch_decoupling",
            "status": "PASS_CONTEXT",
            "evidence": "Donor supply/transport beta +0.480 q=3.30e-8, secretory mucin +0.922 q=5.34e-11, core-3/Sda +0.879 q=1.76e-8, core-2/sLeX transcript composite q=0.915, ST6GAL1 beta -0.742 q=8.50e-5.",
            "claim_enabled": "Donor, carrier, core and linkage layers are transcriptionally decoupled in mucinous-relative comparisons.",
            "claim_forbidden": "Transcript branches do not measure glycan structure, incorporation, enzyme activity or flux.",
            "next_action": "Use the branch forest as mechanistic context and prioritize same-sample glycomics over additional RNA cohorts.",
            "priority": "P0",
        },
        {
            "gate_id": "G16",
            "domain": "same_sample_destination_and_causality",
            "status": "FAIL_MISSING",
            "evidence": "No same-sample CMP-Neu5Ac, linkage-aware O-glycan, MUC2 glycopeptide, isotope incorporation, perturbation or rescue is available for MTBLS13729.",
            "claim_enabled": "A testable hybrid-glycome model can be proposed.",
            "claim_forbidden": "Do not claim that free Neu5Ac enters a specific glycan/carrier or drives a phenotype.",
            "next_action": "Package B: Neu5Ac standard/spike-in plus linkage-aware O-glycans; Package C: add isotope, perturbation and rescue.",
            "priority": "P0",
        },
    ])
    ledger = pd.concat([ledger, additions], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 16:
        raise RuntimeError("v2 gate construction failed")
    ledger.to_csv(OUT / "mechanism_paper_completion_audit_v2.csv", index=False)

    counts = {str(k): int(v) for k, v in ledger.status.value_counts().items()}
    report = {
        "status": "mtbls13729_mechanism_paper_completion_audit_v2_complete",
        "formal": False,
        "gates": int(len(ledger)),
        "status_counts": counts,
        "primary_publishable_phenomenon": "mucinous-relative hybrid mucin glycome with donor-carrier-core-linkage decoupling",
        "biology_package_A_ready": True,
        "biology_package_A_position": "algorithm-enabled, evidence-calibrated clinical discovery",
        "package_B_ready": False,
        "package_B_missing": [
            "same-method Neu5Ac authentic standard, RT, MS2 and spike-in",
            "linkage-aware O-glycan readout in the same or a replacement tissue set",
        ],
        "package_C_ready": False,
        "package_C_missing": [
            "same-sample donor and carrier destination",
            "isotope incorporation or flux",
            "node perturbation, phenotype and rescue",
        ],
        "negative_results_with_value": [
            "full 13,155-target exact FDR10 has zero hits",
            "BioAware v1 corrected zero and introduced one",
            "patient-level module coordination rejects a single upstream chain",
            "ST6GAL1/alpha2-6 directions reject a global-hypersialylation shortcut",
            "MUC2 spatial glycopeptides reject free-pool-to-carrier equivalence",
        ],
        "provenance": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (V1 / "report.json", V1 / "mechanism_paper_completion_audit_v1.csv", GLYCAN, OGLY, FIGURE, DECISION)
        },
        "claim_limit": "Package A readiness means the biology discovery package is internally auditable; it does not imply full-space confirmation, assay-level MSI Level 1, independent free-Neu5Ac replication, flux or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MTBLS13729 mechanism-paper completion audit v2",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        f"**Current position:** {report['biology_package_A_position']}",
        "",
        "| Gate | Domain | Status | Evidence | Next action |",
        "|---|---|---|---|---|",
    ]
    for row in ledger.to_dict("records"):
        lines.append(f"| {row['gate_id']} | {row['domain']} | {row['status']} | {row['evidence']} | {row['next_action']} |")
    lines.extend(["", "## Claim boundary", "", report["claim_limit"], ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
