"""Add same-patient free-Neu5Ac to activated-donor decoupling evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v5_final"
DONOR = ROOT / "data/mtbls13729/sialic_donor_decoupling_v1/report.json"
AUDIT_DOC = ROOT / "docs/MTBLS13729_SIALIC_DONOR_DECOUPLING_AUDIT_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v6_final"


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
        V5 / "mechanism_paper_completion_audit_v5.csv",
        V5 / "report.json",
        DONOR,
        AUDIT_DOC,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    donor = json.loads(DONOR.read_text(encoding="utf-8"))
    if donor.get("status") != "mtbls13729_sialic_donor_decoupling_audit_complete":
        raise RuntimeError("donor-decoupling audit is incomplete")
    if not donor.get("formal") or not all(donor["gates"].values()):
        raise RuntimeError("donor-decoupling gates did not all pass")
    free = donor["node_summaries"]["free_neu5ac"]
    cmp_node = donor["node_summaries"]["cmp_neu5ac"]
    udp = donor["node_summaries"]["udp_glcnac"]
    if free["positive_pairs"] != 10 or cmp_node["identity_level"] != "Level 2":
        raise RuntimeError("unexpected node identity or paired-direction contract")

    ledger = pd.read_csv(V5 / "mechanism_paper_completion_audit_v5.csv")
    if len(ledger) != 19 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 20)}:
        raise RuntimeError("unexpected v5 completion ledger")
    contrasts = {row["contrast"]: row for row in donor["pre_specified_patient_level_contrasts"]}
    cmp_contrast = contrasts["free_neu5ac_minus_cmp_neu5ac"]
    udp_contrast = contrasts["free_neu5ac_minus_udp_glcnac"]
    addition = pd.DataFrame([{
        "gate_id": "G20",
        "domain": "same_patient_free_pool_to_donor_decoupling",
        "status": "PASS_DISCOVERY",
        "evidence": (
            "In 10 Rmu pairs, Level-1 free Neu5Ac increased in 10/10 (mean +2.249 log2), while "
            "Level-2 CMP-Neu5Ac and Level-1 UDP-GlcNAc were not nominally increased. Within-patient "
            "free-minus-donor/precursor contrasts were +1.693/+1.922 log2 with Holm-Wilcoxon p=0.0273."
        ),
        "claim_enabled": (
            "The Rmu free Neu5Ac pool expands more strongly than the measured activated-donor and "
            "upstream nucleotide-sugar nodes, supporting same-patient pool-to-donor decoupling."
        ),
        "claim_forbidden": (
            "Same-cohort static abundance does not establish flux, localisation, enzyme activity or causal "
            "direction; CMP-Neu5Ac is Level 2 and the glycan destination was not measured."
        ),
        "next_action": (
            "Confirm CMP-Neu5Ac with an authentic standard and measure free Neu5Ac, CMP-Neu5Ac and "
            "linkage-aware glycans in the same independent mucinous tissue set."
        ),
        "priority": "P0",
    }])
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 20:
        raise RuntimeError("v6 gate construction failed")

    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v6.csv"
    ledger.to_csv(ledger_path, index=False)
    v5_report = json.loads((V5 / "report.json").read_text(encoding="utf-8"))
    report = {
        **v5_report,
        "status": "mtbls13729_mechanism_paper_completion_audit_v6_complete",
        "gates": 20,
        "status_counts": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "same_patient_free_pool_to_donor_decoupling": {
            "status": "PASS_DISCOVERY",
            "rmu_pairs": free["n_pairs"],
            "free_neu5ac_positive_pairs": free["positive_pairs"],
            "free_neu5ac_mean_log2": free["mean_paired_log2_delta"],
            "cmp_neu5ac_mean_log2": cmp_node["mean_paired_log2_delta"],
            "udp_glcnac_mean_log2": udp["mean_paired_log2_delta"],
            "free_minus_cmp_mean_log2": cmp_contrast["mean_log2_delta_difference"],
            "free_minus_cmp_holm_p": cmp_contrast["wilcoxon_holm_p"],
            "free_minus_udp_mean_log2": udp_contrast["mean_log2_delta_difference"],
            "free_minus_udp_holm_p": udp_contrast["wilcoxon_holm_p"],
            "cmp_neu5ac_identity_level": cmp_node["identity_level"],
            "independent_replication": False,
            "flux_evidence": False,
        },
        "package_C_missing": [
            "independent same-sample confirmation of free Neu5Ac and authentic-standard CMP-Neu5Ac",
            "same-sample linkage-aware glycan carrier/destination",
            "isotope incorporation or flux",
            "node perturbation, phenotype and rescue",
        ],
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_limit": (
            "The same-patient decomposition strengthens the free-pool-to-donor decoupling model, but it "
            "remains a same-cohort static-abundance discovery. CMP-Neu5Ac is Level 2, and independent "
            "abundance replication, glycan destination, flux and causality remain missing."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# MTBLS13729 mechanism-paper completion audit v6",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        (
            "**New same-patient evidence:** free Neu5Ac rises more strongly than CMP-Neu5Ac and "
            "UDP-GlcNAc in the 10 Rmu tumour-normal pairs."
        ),
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
