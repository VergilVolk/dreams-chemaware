"""Add carrier-resolved PXD055865 MUC2 evidence and the reference-spectrum boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v8_final"
PXD = ROOT / "data/external/PXD055865_2026_MUC2/audit_v1/report.json"
PXD_DOC = ROOT / "docs/MTBLS13729_PXD055865_MUC2_GLYCOPEPTIDE_AUDIT_20260831.md"
RESOURCE_DOC = ROOT / "docs/MTBLS13729_OACETYL_NEU5AC_STANDARD_RESOURCE_AUDIT_20260831.md"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v9_final"


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
        V8 / "mechanism_paper_completion_audit_v8.csv",
        V8 / "report.json",
        PXD,
        PXD_DOC,
        RESOURCE_DOC,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    pxd = json.loads(PXD.read_text(encoding="utf-8"))
    if pxd.get("status") != "pxd055865_muc2_glycoform_audit_complete":
        raise RuntimeError("PXD055865 audit is incomplete")
    design = pxd.get("dataset", {}).get("design", "")
    if "two patients" not in design:
        raise RuntimeError("PXD055865 patient-independence boundary is missing")
    specimens = {row["specimen"]: row for row in pxd.get("specimens", [])}
    if set(specimens) != {"Colon1a", "Colon1b", "Colon2", "HealthyColon"}:
        raise RuntimeError("unexpected PXD055865 specimen set")
    if specimens["Colon1a"]["patient"] != specimens["Colon1b"]["patient"]:
        raise RuntimeError("Colon1a and Colon1b must remain grouped as one patient")
    if pxd.get("claim_limit", "").find("Identification counts") < 0:
        raise RuntimeError("identification-count abundance boundary is missing")

    ledger = pd.read_csv(V8 / "mechanism_paper_completion_audit_v8.csv")
    if len(ledger) != 22 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 23)}:
        raise RuntimeError("unexpected v8 completion ledger")
    addition = pd.DataFrame(
        [
            {
                "gate_id": "G23",
                "domain": "external_muc2_carrier_destination",
                "status": "PASS_WITH_LIMITATION",
                "evidence": (
                    "PXD055865 supplies manually reviewed MUC2 glycopeptide lists and source "
                    "spectra from three mucinous CRC specimens belonging to two independent "
                    "patients plus one healthy colon. The public data contain sialylated, "
                    "O-acetyl-Neu5Ac and putative O-acetyl-GalNAc MUC2 glycopeptides and support "
                    "strong spatial/carrier heterogeneity."
                ),
                "claim_enabled": (
                    "An independent carrier-resolved dataset is directionally compatible with "
                    "destination-level MUC2 glycan remodeling and with a free-pool/carrier "
                    "decoupling interpretation."
                ),
                "claim_forbidden": (
                    "The dataset does not measure free Neu5Ac, contains only two independent "
                    "mucinous patients, and has unequal discovery depth; identification counts "
                    "cannot be interpreted as tumour-versus-normal abundance."
                ),
                "next_action": (
                    "Obtain linkage-aware O-glycan or MUC2 glycopeptide readout in the same or a "
                    "replacement cohort; use authentic standards or IM-MS/CCS for O-acetyl "
                    "positional-isomer claims."
                ),
                "priority": "P1",
            }
        ]
    )
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 23:
        raise RuntimeError("v9 gate construction failed")

    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v9.csv"
    ledger.to_csv(ledger_path, index=False)

    report = json.loads((V8 / "report.json").read_text(encoding="utf-8"))
    report.update(
        {
            "status": "mtbls13729_mechanism_paper_completion_audit_v9_complete",
            "gates": 23,
            "status_counts": {
                str(key): int(value) for key, value in ledger.status.value_counts().items()
            },
            "external_muc2_carrier_destination": {
                "status": "PASS_WITH_LIMITATION",
                "dataset": "PXD055865",
                "mucinous_specimens": 3,
                "independent_mucinous_patients": 2,
                "healthy_colons": 1,
                "unique_muc2_glycopeptides_by_specimen": {
                    name: int(row["unique_muc2_glycopeptides"])
                    for name, row in specimens.items()
                },
                "source_spectra_support_oacetyl_neu5ac": bool(
                    pxd["source_spectrum_support"]["di_o_acetyl_neu5ac_sheets"]
                ),
                "source_spectra_support_oacetyl_galnac": bool(
                    pxd["source_spectrum_support"]["o_acetyl_galnac_sheet_present"]
                ),
                "free_neu5ac_measured": False,
                "abundance_comparison_permitted": False,
                "independent_free_pool_replication": False,
            },
            "oacetyl_reference_resource_boundary": {
                "status": "FAIL_MISSING_EXPERIMENTAL_REFERENCE",
                "hmdb_evidence": "predicted spectra only for HMDB0000794",
                "massbank_exact_name_or_formula_record": False,
                "mona_exact_name_or_formula_record": False,
                "positional_isomer_resolved": False,
                "minimum_resolution": (
                    "paired 4-O- and 9-O-acetyl-Neu5Ac standards under the study LC-MS/MS "
                    "method; IM-MS/CCS if chromatographic and MS2 separation remains insufficient"
                ),
            },
            "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
            "claim_limit": (
                "The combined evidence supports free-pool/donor decoupling and provides "
                "independent carrier-resolved context consistent with MUC2 destination "
                "remodeling. PXD055865 is not a free-Neu5Ac abundance replication, contains only "
                "two independent mucinous patients, and does not establish O-acetyl positional "
                "isomers, biochemical source, flux, or causality."
            ),
        }
    )
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# MTBLS13729 mechanism-paper completion audit v9",
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
