"""Fail-closed validation for the MTBLS13729 v9 biology evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v9_final"


def main() -> None:
    report_path = OUT / "report.json"
    ledger_path = OUT / "mechanism_paper_completion_audit_v9.csv"
    readme_path = OUT / "README.md"
    for path in (report_path, ledger_path, readme_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    ledger = pd.read_csv(ledger_path)
    if report.get("status") != "mtbls13729_mechanism_paper_completion_audit_v9_complete":
        raise RuntimeError("unexpected v9 report status")
    if report.get("formal") is not False:
        raise RuntimeError("discovery package must not be labelled formal confirmation")
    if len(ledger) != 23 or ledger.gate_id.tolist() != [f"G{i:02d}" for i in range(1, 24)]:
        raise RuntimeError("v9 gate sequence mismatch")
    g23 = ledger.set_index("gate_id").loc["G23"]
    if g23["status"] != "PASS_WITH_LIMITATION":
        raise RuntimeError("external MUC2 carrier gate must remain limited")

    external = report["external_muc2_carrier_destination"]
    invariants = {
        "two independent mucinous patients": external["independent_mucinous_patients"] == 2,
        "three specimens": external["mucinous_specimens"] == 3,
        "one healthy colon": external["healthy_colons"] == 1,
        "no free Neu5Ac measurement": external["free_neu5ac_measured"] is False,
        "no abundance interpretation": external["abundance_comparison_permitted"] is False,
        "no independent free-pool replication": external["independent_free_pool_replication"] is False,
        "package A remains ready": report["biology_package_A_ready"] is True,
        "package B remains incomplete": report["package_B_ready"] is False,
        "package C remains incomplete": report["package_C_ready"] is False,
    }
    failed = [label for label, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"v9 invariant failures: {failed}")

    resource = report["oacetyl_reference_resource_boundary"]
    if resource["status"] != "FAIL_MISSING_EXPERIMENTAL_REFERENCE":
        raise RuntimeError("O-acetyl reference gap was silently upgraded")
    if resource["positional_isomer_resolved"] is not False:
        raise RuntimeError("O-acetyl positional-isomer identity was overclaimed")

    required_docs = {
        ROOT / "docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md": [
            "独立患者数：**2**",
            "**不是无偏丰度**",
            "不能称为 MTBLS13729 游离 Neu5Ac 的独立复制",
        ],
        ROOT / "docs/MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md": [
            "位置异构体未解析",
            "普通 exact mass + 常规 MS2 仍不足",
        ],
        ROOT / "docs/MTBLS13729_MINIMAL_VALIDATION_AND_REVIEWER_ATTACK_PLAN_20260831.md": [
            "algorithm-enabled, evidence-calibrated clinical discovery",
            "同法标准支持",
        ],
    }
    for path, phrases in required_docs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in phrases if phrase not in text]
        if missing:
            raise RuntimeError(f"{path.name} missing required boundaries: {missing}")

    figure_report_path = (
        ROOT / "data/mtbls13729/pool_carrier_boundary_figure_v1/report.json"
    )
    figure_png = (
        ROOT
        / "data/mtbls13729/pool_carrier_boundary_figure_v1/"
        "pool_carrier_boundary_figure_v1.png"
    )
    if not figure_report_path.is_file() or not figure_png.is_file():
        raise FileNotFoundError("pool-carrier boundary figure package is incomplete")
    figure_report = json.loads(figure_report_path.read_text(encoding="utf-8"))
    if figure_report["pdx_independent_mucinous_patients"] != 2:
        raise RuntimeError("figure patient-independence boundary mismatch")
    if figure_report["pdx_abundance_interpretation_permitted"] is not False:
        raise RuntimeError("figure silently treats identification presence as abundance")

    print(
        "[validate_mtbls13729_mechanism_paper_completion_audit_v9] PASS "
        f"gates={len(ledger)} patients={external['independent_mucinous_patients']} figure=PASS"
    )


if __name__ == "__main__":
    main()
