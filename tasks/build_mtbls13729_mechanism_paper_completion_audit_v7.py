"""Add a prespecified transcript mechanism-discrimination gate to v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v6_final"
MECHANISM = ROOT / (
    "data/external/TCGA_COADREAD_Xena_20260830/sialic_pool_mechanisms_v1/report.json"
)
AUDIT_DOC = ROOT / "docs/MTBLS13729_SIALIC_POOL_MECHANISM_DISCRIMINATION_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v7_final"


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
        V6 / "mechanism_paper_completion_audit_v6.csv",
        V6 / "report.json",
        MECHANISM,
        AUDIT_DOC,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    mechanism = json.loads(MECHANISM.read_text(encoding="utf-8"))
    if mechanism.get("status") != "tcga_sialic_pool_mechanism_audit_complete":
        raise RuntimeError("sialic mechanism audit is incomplete")
    axes = {
        row["outcome"]: row
        for row in mechanism["results"]
        if row["outcome_type"] == "axis"
    }
    release = axes["selected_sialidase_release"]
    activation = axes["cmp_activation_transport"]
    if not (
        release["paired_mean_tumour_minus_normal_z"] > 0
        and release["lineage_beta"] < 0
        and release["lineage_bh_q"] < 0.05
        and activation["lineage_beta"] > 0
        and activation["lineage_bh_q"] < 0.05
    ):
        raise RuntimeError("expected discriminating mechanism directions did not reproduce")

    ledger = pd.read_csv(V6 / "mechanism_paper_completion_audit_v6.csv")
    if len(ledger) != 20 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 21)}:
        raise RuntimeError("unexpected v6 completion ledger")
    addition = pd.DataFrame([{
        "gate_id": "G21",
        "domain": "prespecified_free_pool_mechanism_discrimination",
        "status": "PASS_WITH_LIMITATION",
        "evidence": (
            "In 32 paired CRCs NEU1/NEU3 RNA increased (mean +0.854 z; BH q 9.02e-7), but in "
            "42 mucinous versus 329 conventional tumours it decreased after lineage/MSI adjustment "
            "(beta -0.691/-0.654; BH q 5.58e-6/1.53e-5). CMP activation/transport showed the opposite "
            "mucinous-relative direction (beta +0.449; BH q 1.61e-4)."
        ),
        "claim_enabled": (
            "Bulk RNA does not support NEU1/NEU3 transcriptional upregulation as the simple explanation "
            "for the mucinous-relative free-Neu5Ac pool; transcriptional capacity and measured donor pool "
            "are decoupled."
        ),
        "claim_forbidden": (
            "RNA cannot exclude sialidase protein activity, other enzymes, microbiota, secretion/turnover, "
            "O-acetylation, subcellular transport or glycan incorporation mechanisms."
        ),
        "next_action": (
            "Prioritise same-sample Neu5,9Ac2/free Neu5Ac/CMP-Neu5Ac plus linkage-aware glycans; test "
            "NEU1/3, CMAS/SLC35A1 and NXPE1/CASD1 only if protein/activity material becomes available."
        ),
        "priority": "P1",
    }])
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 21:
        raise RuntimeError("v7 gate construction failed")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v7.csv"
    ledger.to_csv(ledger_path, index=False)
    report = json.loads((V6 / "report.json").read_text(encoding="utf-8"))
    report.update({
        "status": "mtbls13729_mechanism_paper_completion_audit_v7_complete",
        "gates": 21,
        "status_counts": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "free_pool_mechanism_discrimination": {
            "status": "PASS_WITH_LIMITATION",
            "general_crc_neu1_neu3_mean_z": release["paired_mean_tumour_minus_normal_z"],
            "general_crc_neu1_neu3_bh_q": release["paired_wilcoxon_bh_q"],
            "mucinous_neu1_neu3_lineage_beta": release["lineage_beta"],
            "mucinous_neu1_neu3_lineage_bh_q": release["lineage_bh_q"],
            "mucinous_cmp_activation_lineage_beta": activation["lineage_beta"],
            "mucinous_cmp_activation_lineage_bh_q": activation["lineage_bh_q"],
            "protein_or_flux_evidence": False,
        },
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_limit": (
            "The prespecified RNA branches reject a simple NEU1/NEU3-transcription explanation for the "
            "mucinous-relative free pool, but do not identify the biochemical source. Same-sample protein, "
            "O-acetylated sialic acid, glycan destination and flux remain missing."
        ),
    })
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# MTBLS13729 mechanism-paper completion audit v7",
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
