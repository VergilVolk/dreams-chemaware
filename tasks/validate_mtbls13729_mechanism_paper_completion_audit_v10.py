"""Fail-closed validation for the MTBLS13729 v10 biology evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v10_final"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(OUT / "mechanism_paper_completion_audit_v10.csv")
    if report.get("status") != "mtbls13729_mechanism_paper_completion_audit_v10_complete":
        raise RuntimeError("unexpected v10 report status")
    if report.get("formal") is not False:
        raise RuntimeError("discovery package must not be labelled formal confirmation")
    if len(ledger) != 25 or ledger.gate_id.tolist() != [f"G{i:02d}" for i in range(1, 26)]:
        raise RuntimeError("v10 gate sequence mismatch")
    indexed = ledger.set_index("gate_id")
    if indexed.loc["G24", "status"] != "PASS_CONTEXT":
        raise RuntimeError("raw transcript gate changed")
    if indexed.loc["G25", "status"] != "NEGATIVE_RESULT":
        raise RuntimeError("negative proteomics gate changed")
    transcript = report["independent_patient_raw_transcript_context"]
    if set(transcript["supporting_endpoints"]) != {
        "Epi|secretory_carrier",
        "Epi|cmp_neu5ac_capacity",
    }:
        raise RuntimeError("raw transcript support set changed")
    invariants = {
        "NXPE1 not independently supported": transcript["NXPE1_primary_support"] is False,
        "no host release mechanism": transcript["host_release_mechanism_supported"] is False,
        "no biochemical source": transcript["biochemical_source_established"] is False,
        "proteomics remains negative": report["independent_patient_proteomics_context"]["module_support"] is False,
        "no abundance replication": report["independent_patient_proteomics_context"]["independent_neu5ac_abundance_replication"] is False,
        "package A ready": report["biology_package_A_ready"] is True,
        "package B incomplete": report["package_B_ready"] is False,
        "package C incomplete": report["package_C_ready"] is False,
        "independent metabolite replication missing": report["independent_mucinous_abundance_replication"] == "FAIL_MISSING",
    }
    failed = [label for label, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"v10 invariant failures: {failed}")
    if not (OUT / "README.md").is_file():
        raise FileNotFoundError(OUT / "README.md")
    print(
        "[validate_mtbls13729_mechanism_paper_completion_audit_v10] PASS "
        f"gates={len(ledger)} transcript=PASS_CONTEXT proteomics=NEGATIVE_RESULT"
    )


if __name__ == "__main__":
    main()
