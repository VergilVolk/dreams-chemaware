"""Add the independent Neu5Ac biogeography context to the readiness ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v2_final"
EXTERNAL = ROOT / "data/external/CRC_metabolic_biogeography_PMC11438248_20260831/neu5ac_biogeography_audit_v1/report.json"
AUDIT_DOC = ROOT / "docs/MTBLS13729_EXTERNAL_NEU5AC_BIOGEOGRAPHY_AUDIT_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v3_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    required = [V2 / "mechanism_paper_completion_audit_v2.csv", V2 / "report.json", EXTERNAL, AUDIT_DOC]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    external = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    if external.get("status") != "external_crc_neu5ac_biogeography_audit_complete":
        raise RuntimeError("external Neu5Ac audit is not complete")
    if external["level1_neu5ac"]["id_level"] != "1":
        raise RuntimeError("external Neu5Ac identity is not Level 1")
    if external["histology_audit"]["mucinous_subgroup_available"]:
        raise RuntimeError("unexpected mucinous subgroup in external audit")

    ledger = pd.read_csv(V2 / "mechanism_paper_completion_audit_v2.csv")
    if len(ledger) != 16 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 17)}:
        raise RuntimeError("unexpected v2 completion ledger")
    addition = pd.DataFrame([{
        "gate_id": "G17",
        "domain": "independent_neu5ac_spatial_context",
        "status": "PASS_CONTEXT",
        "evidence": (
            "Independent 372-pair CRC cohort: standard-supported HILIC(-) Level-1 Neu5Ac; "
            "normal cecum-to-rectum slope +0.349 (p<0.001), tumour slope +0.088 (p=0.091)."
        ),
        "claim_enabled": (
            "Neu5Ac is independently measurable in CRC tissue and its anatomical gradient is disease dependent; "
            "paired and location-aware analysis is biologically necessary."
        ),
        "claim_forbidden": (
            "No mucinous subgroup is available; do not call this an independent Rmu abundance replication, "
            "same-method validation, flux, or causality."
        ),
        "next_action": (
            "Retain as external spatial context and continue seeking mucinous-labelled patient-level abundance "
            "or perform same-method standard/spike-in plus linkage-aware glycomics."
        ),
        "priority": "P0",
    }])
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 17:
        raise RuntimeError("v3 gate construction failed")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v3.csv"
    ledger.to_csv(ledger_path, index=False)

    v2_report = json.loads((V2 / "report.json").read_text(encoding="utf-8"))
    report = {
        "status": "mtbls13729_mechanism_paper_completion_audit_v3_complete",
        "formal": False,
        "gates": 17,
        "status_counts": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "primary_publishable_phenomenon": v2_report["primary_publishable_phenomenon"],
        "biology_package_A_ready": True,
        "biology_package_A_position": v2_report["biology_package_A_position"],
        "external_neu5ac_context": {
            "status": "PASS_CONTEXT",
            "cohort_pairs": 372,
            "identity_level": 1,
            "normal_slope": external["spatial_gradient"]["control_slope"],
            "normal_p": external["spatial_gradient"]["control_p_text"],
            "tumour_slope": external["spatial_gradient"]["tumour_slope"],
            "tumour_p": external["spatial_gradient"]["tumour_p_text"],
            "mucinous_subgroup": False,
        },
        "independent_mucinous_abundance_replication": "FAIL_MISSING",
        "package_B_ready": False,
        "package_B_missing": v2_report["package_B_missing"],
        "package_C_ready": False,
        "package_C_missing": v2_report["package_C_missing"],
        "negative_results_with_value": v2_report["negative_results_with_value"],
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_limit": (
            "The 372-pair study adds independent Level-1 CRC-tissue identity and disease-dependent spatial "
            "context. It does not provide a mucinous subgroup, same-method MTBLS13729 validation, independent "
            "Rmu abundance replication, flux, or causality."
        ),
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MTBLS13729 mechanism-paper completion audit v3",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        "**New external context:** independent Level-1 Neu5Ac and disease-dependent spatial gradient in 372 CRC pairs; no mucinous subgroup.",
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
