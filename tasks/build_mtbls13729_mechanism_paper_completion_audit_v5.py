"""Add external pooled sialyltransferase/histology context to the readiness ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v4_final"
EXTERNAL = ROOT / "data/external/CRC_sialylome_mucinous_Biology2026_20260831/report.json"
AUDIT_DOC = ROOT / "docs/MTBLS13729_EXTERNAL_SIALYLOME_MUCINOUS_AUDIT_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v5_final"


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
        V4 / "mechanism_paper_completion_audit_v4.csv",
        V4 / "report.json",
        EXTERNAL,
        AUDIT_DOC,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    external = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    if external.get("status") != "external_crc_sialylome_mucinous_audit_complete":
        raise RuntimeError("external sialylome audit is not complete")
    table = external["histology_table"]
    if table["histology_total"] != 980 or external["score_definition"]["gene_count"] != 20:
        raise RuntimeError("unexpected external sialylome contract")
    if external["overlap_audit"]["independent_of_local_tcga_branch_analysis"]:
        raise RuntimeError("external paper unexpectedly classified as TCGA-independent")

    ledger = pd.read_csv(V4 / "mechanism_paper_completion_audit_v4.csv")
    if len(ledger) != 18 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 19)}:
        raise RuntimeError("unexpected v4 completion ledger")
    addition = pd.DataFrame([{
        "gate_id": "G19",
        "domain": "external_sialyltransferase_mucinous_context",
        "status": "PASS_CONTEXT",
        "evidence": (
            "Integrated TCGA/Sidra-LUMC/CPTAC-2 supplement: Sialyl-High in 85/154 mucinous "
            "versus 238/826 non-mucinous cases; reconstructed OR 3.04 (95% CI 2.14-4.33)."
        ),
        "claim_enabled": (
            "Mucinous CRC is externally associated with a high sialyltransferase-expression programme."
        ),
        "claim_forbidden": (
            "The score is 20 sialyltransferase transcripts, not Neu5Ac, glycan structure, flux or enzyme "
            "activity; TCGA overlap prevents calling it fully independent of the local RNA analysis."
        ),
        "next_action": (
            "Use as contextual support only; independent mucinous metabolite abundance and same-sample "
            "linkage-aware glycomics remain the decisive missing evidence."
        ),
        "priority": "P1",
    }])
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 19:
        raise RuntimeError("v5 gate construction failed")

    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v5.csv"
    ledger.to_csv(ledger_path, index=False)
    v4_report = json.loads((V4 / "report.json").read_text(encoding="utf-8"))
    report = {
        "status": "mtbls13729_mechanism_paper_completion_audit_v5_complete",
        "formal": False,
        "gates": 19,
        "status_counts": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "primary_publishable_phenomenon": v4_report["primary_publishable_phenomenon"],
        "biology_package_A_ready": v4_report["biology_package_A_ready"],
        "biology_package_A_position": v4_report["biology_package_A_position"],
        "external_neu5ac_context": v4_report["external_neu5ac_context"],
        "external_public_plot_audit": v4_report["external_public_plot_audit"],
        "external_sialyltransferase_mucinous_context": {
            "status": "PASS_CONTEXT",
            "histology_n": table["histology_total"],
            "mucinous_high_fraction": table["mucinous_high_fraction"],
            "non_mucinous_high_fraction": table["non_mucinous_high_fraction"],
            "odds_ratio": table["reconstructed_odds_ratio"],
            "odds_ratio_ci95": table["reconstructed_odds_ratio_ci95"],
            "partial_tcga_overlap": True,
            "measures_neu5ac": False,
        },
        "independent_mucinous_abundance_replication": v4_report["independent_mucinous_abundance_replication"],
        "package_B_ready": v4_report["package_B_ready"],
        "package_B_missing": v4_report["package_B_missing"],
        "package_C_ready": v4_report["package_C_ready"],
        "package_C_missing": v4_report["package_C_missing"],
        "negative_results_with_value": v4_report["negative_results_with_value"],
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_limit": (
            "The external 980-case histology table supports a pooled sialyltransferase-expression association "
            "with mucinous CRC. It partially overlaps TCGA and does not measure Neu5Ac or glycans, so the "
            "independent mucinous abundance-replication gate remains missing."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MTBLS13729 mechanism-paper completion audit v5",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        "**New context:** external pooled sialyltransferase-expression association with mucinous histology; partial TCGA overlap and no Neu5Ac measurement.",
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
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

