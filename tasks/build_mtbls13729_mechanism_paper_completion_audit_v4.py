"""Add the public Neu5Ac Dash reproducibility boundary to the readiness ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v3_final"
DASH = (
    ROOT
    / "data/external/CRC_metabolic_biogeography_PMC11438248_20260831"
    / "neu5ac_dash_patient_level_v1/report.json"
)
AUDIT_DOC = ROOT / "docs/MTBLS13729_EXTERNAL_NEU5AC_DASH_REPRODUCIBILITY_AUDIT_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v4_final"


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
        V3 / "mechanism_paper_completion_audit_v3.csv",
        V3 / "report.json",
        DASH,
        AUDIT_DOC,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    dash = json.loads(DASH.read_text(encoding="utf-8"))
    if dash.get("status") != "external_crc_neu5ac_dash_patient_level_audit_complete":
        raise RuntimeError("external Neu5Ac Dash audit is not complete")
    if dash["supplement_regression_reproduced_from_public_plot_values"]:
        raise RuntimeError("unexpected exact reproduction of supplement from public plot")
    if any(dash["displayed_count_matches_paper_pairs"].values()):
        raise RuntimeError("unexpected agreement between public plot and paper sample count")
    for tissue in ("normal", "tumour"):
        audit = dash["figure_audit"][tissue]
        if audit["total_values"] != 371 or audit["patient_identifiers_available"]:
            raise RuntimeError(f"unexpected public plot contract for {tissue}: {audit}")

    ledger = pd.read_csv(V3 / "mechanism_paper_completion_audit_v3.csv")
    if len(ledger) != 17 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 18)}:
        raise RuntimeError("unexpected v3 completion ledger")
    addition = pd.DataFrame([{
        "gate_id": "G18",
        "domain": "external_neu5ac_public_data_reproducibility",
        "status": "PASS_WITH_LIMITATION",
        "evidence": (
            "Dash callback frozen: 371 values per tissue, exactly 53 per subsite, no patient IDs; "
            "direct regression does not reproduce supplement slopes/p-values."
        ),
        "claim_enabled": (
            "The public distributions directionally show a stronger normal than tumour anatomical gradient."
        ),
        "claim_forbidden": (
            "Do not call the Dash values complete patient-level source data, pair tumour/normal arrays, "
            "or replace the supplement's authoritative regression."
        ),
        "next_action": (
            "Cite the formal supplement for external statistics; request analysis-ready patient-level data "
            "only if exact independent reanalysis becomes necessary."
        ),
        "priority": "P1",
    }])
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 18:
        raise RuntimeError("v4 gate construction failed")

    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v4.csv"
    ledger.to_csv(ledger_path, index=False)

    v3_report = json.loads((V3 / "report.json").read_text(encoding="utf-8"))
    report = {
        "status": "mtbls13729_mechanism_paper_completion_audit_v4_complete",
        "formal": False,
        "gates": 18,
        "status_counts": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "primary_publishable_phenomenon": v3_report["primary_publishable_phenomenon"],
        "biology_package_A_ready": v3_report["biology_package_A_ready"],
        "biology_package_A_position": v3_report["biology_package_A_position"],
        "external_neu5ac_context": v3_report["external_neu5ac_context"],
        "external_public_plot_audit": {
            "status": "PASS_WITH_LIMITATION",
            "displayed_values_per_tissue": 371,
            "displayed_values_per_subsite": 53,
            "patient_identifiers": False,
            "supplement_regression_reproduced": False,
            "normal_public_r": dash["statistics_by_tissue"]["normal"]["public_plot_ordinal_regression"]["pearson_r_or_standardized_beta"],
            "tumour_public_r": dash["statistics_by_tissue"]["tumour"]["public_plot_ordinal_regression"]["pearson_r_or_standardized_beta"],
        },
        "independent_mucinous_abundance_replication": v3_report["independent_mucinous_abundance_replication"],
        "package_B_ready": v3_report["package_B_ready"],
        "package_B_missing": v3_report["package_B_missing"],
        "package_C_ready": v3_report["package_C_ready"],
        "package_C_missing": v3_report["package_C_missing"],
        "negative_results_with_value": v3_report["negative_results_with_value"] + [
            "the public Neu5Ac Dash values do not reproduce the supplement's sample counts or regressions"
        ],
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_limit": (
            "External Neu5Ac identity and spatial statistics are supported by the formal paper supplement. "
            "The public Dash values provide direction-only visual context because patient identifiers are "
            "absent and their sample counts/regressions do not reproduce the supplement."
        ),
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MTBLS13729 mechanism-paper completion audit v4",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        "**New audit boundary:** the public Neu5Ac Dash distributions are direction-only context, not analysis-ready patient-level source data.",
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

