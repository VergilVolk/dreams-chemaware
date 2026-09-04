#!/usr/bin/env python
"""Assemble the auditable MTBLS13729 ion-family and acylcarnitine closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-report",
        type=Path,
        default=Path("data/mtbls13729/frozen_ion_family_audit_20260829/report.json"),
    )
    parser.add_argument(
        "--anchor-report",
        type=Path,
        default=Path("data/mtbls13729/c20_4_anchor_ms2_audit_20260829/report.json"),
    )
    parser.add_argument(
        "--class-report",
        type=Path,
        default=Path("data/mtbls13729/acylcarnitine_panel/class_score_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mtbls13729/biology_closure_20260829/report.json"),
    )
    args = parser.parse_args()

    family = read_json(args.family_report)
    anchor = read_json(args.anchor_report)
    class_report = read_json(args.class_report)
    if family.get("status") != "mtbls13729_frozen_ion_family_audit_complete":
        raise RuntimeError("unexpected frozen ion-family report")
    if anchor.get("status") != "mtbls13729_c20_4_ms2_audit_complete":
        raise RuntimeError("unexpected C20:4 anchor report")
    if class_report.get("status") != "complete":
        raise RuntimeError("unexpected acylcarnitine class report")
    if not class_report.get("selection_is_phenotype_blind", False):
        raise RuntimeError("acylcarnitine class was not selected phenotype-blind")

    pqn = class_report["variants"]["pqn"]
    collapsed_pqn = class_report["chain_collapsed_sensitivity"]["variants"]["pqn"]
    primary_features = int(family["panels"]["pos_rp"]["primary_fdr10_features"])
    primary_families = int(
        family["panels"]["pos_rp"]["primary_fdr10_descriptive_ion_families"]
    )
    payload = {
        "status": "mtbls13729_biology_closure_complete",
        "formal": False,
        "feature_to_ion_family_reconciliation": {
            "pos_rp_primary_fdr10_features": primary_features,
            "pos_rp_descriptive_ion_families": primary_families,
            "features_removed_as_redundant_ion_forms": primary_features - primary_families,
            "formal_outcome_blind_global_peak_graph": bool(
                family.get("formal_global_peak_graph", False)
            ),
        },
        "long_chain_acylcarnitine_class": {
            "selection_is_phenotype_blind": True,
            "pqn_rmu_vs_rn": pqn["Rmu_vs_RN"],
            "pqn_rtu_vs_rn": pqn["Rtu_vs_RN"],
            "pqn_secondary_subtype_interaction": pqn["interaction"],
            "chain_collapsed_pqn_rmu_vs_rn": collapsed_pqn["Rmu_vs_RN"],
            "chain_collapsed_pqn_rtu_vs_rn": collapsed_pqn["Rtu_vs_RN"],
            "chain_collapsed_pqn_secondary_subtype_interaction": collapsed_pqn["interaction"],
        },
        "c20_4_anchor": {
            "feature_id": 3222,
            "samples_with_matching_ms2": anchor["samples_with_matching_ms2"],
            "samples_with_strong_carnitine_motif": anchor["samples_with_strong_motif"],
            "matching_ms2_spectra": anchor["matching_ms2_spectra"],
            "strong_motif_spectra": anchor["strong_motif_spectra"],
            "identity_level": "MSI Level 2 candidate",
        },
        "gates": {
            "formal_outcome_blind_ion_family_graph": bool(
                family.get("formal_global_peak_graph", False)
            ),
            "feature_redundancy_not_inflating_family_count": primary_families <= primary_features,
            "phenotype_blind_acylcarnitine_class": bool(
                class_report.get("selection_is_phenotype_blind", False)
            ),
            "c20_4_has_matching_ms2": int(anchor["samples_with_matching_ms2"]) > 0,
            "c20_4_has_diagnostic_class_motif": int(anchor["samples_with_strong_motif"]) > 0,
        },
        "provenance": {
            "family_report_sha256": sha256(args.family_report),
            "anchor_report_sha256": sha256(args.anchor_report),
            "class_report_sha256": sha256(args.class_report),
        },
        "claim_limit": (
            "This is a discovery-level static-abundance closure. Ion-family grouping prevents duplicate "
            "counting and diagnostic fragments support the acylcarnitine class, but neither establishes "
            "MSI Level 1 identity, mucinous-specific interaction, metabolic flux, or enzyme activity."
        ),
    }
    payload["gates"]["pass"] = bool(all(payload["gates"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
