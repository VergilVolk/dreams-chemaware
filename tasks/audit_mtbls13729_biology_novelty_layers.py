"""Separate MTBLS13729 rediscovery, annotation gain, and biological novelty.

The source publication, DreaMS votes, re-extracted abundance, and orthogonal
chromatography answer different questions.  This audit prevents an identity
already present in the source supplement from being presented as a newly
discovered metabolite, while retaining genuine algorithmic additions and
corrections as explicit, reviewable categories.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v1/integrated_candidate_ledger.csv"
SOURCE = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
OUT = ROOT / "data/mtbls13729/biology_novelty_audit_v1"


def normalize_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    # Library names often retain racemate/vendor qualifiers that do not imply
    # a different metabolite identity.  Strip only the frozen aliases observed
    # in this ledger; do not apply fuzzy matching.
    aliases = {
        "carnitinedl": "carnitine",
        "dlcarnitine": "carnitine",
    }
    return aliases.get(text, text)


def truthy_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    source = pd.read_excel(SOURCE, sheet_name="metabolites", header=1)
    source_names = {
        normalize_name(value): str(value)
        for value in source["metabolites"].dropna().astype(str)
        if normalize_name(value)
    }

    rows: list[dict] = []
    for record in ledger.to_dict("records"):
        source_exact_row = isinstance(record.get("source_name"), str) and bool(record["source_name"].strip())
        label_key = normalize_name(record["label"])
        source_name_elsewhere = source_names.get(label_key)

        # The RP feature is not the published HILIC row, but the same chemical
        # name is present in the source supplement.  Keep this distinct from a
        # truly absent source-table identity.
        if int(record["feature_id"]) == 1717:
            source_name_elsewhere = "N1,N8-Diacetylspermidine"

        if source_exact_row:
            novelty_layer = "source_identity_recovered_or_remapped"
        elif source_name_elsewhere:
            novelty_layer = "source_listed_identity_new_peak_or_orthogonal_support"
        else:
            novelty_layer = "algorithm_added_candidate_absent_from_source_identity_table"

        if record["manuscript_evidence_tier"] == "C_downgraded_or_control":
            novelty_layer = "source_identity_but_biology_downgraded"

        dreams_name = record.get("dreams_name")
        dreams_present = isinstance(dreams_name, str) and bool(dreams_name.strip())
        source_name = record.get("source_name")
        name_conflict = bool(
            dreams_present
            and isinstance(source_name, str)
            and normalize_name(dreams_name) != normalize_name(source_name)
        )

        source_rmu_p = record.get("source_rmu_vs_normal_p")
        source_global_fdr = record.get("source_all_cancer_vs_normal_fdr")
        rows.append(
            {
                "feature_id": int(record["feature_id"]),
                "label": record["label"],
                "module": record["module"],
                "manuscript_evidence_tier": record["manuscript_evidence_tier"],
                "novelty_layer": novelty_layer,
                "source_exact_row_linked": source_exact_row,
                "source_same_name_elsewhere": source_name_elsewhere or "",
                "source_rmu_nominal_p_lt_0_05": bool(truthy_finite(source_rmu_p) and float(source_rmu_p) < 0.05),
                "source_global_fdr_lt_0_05": bool(
                    truthy_finite(source_global_fdr) and float(source_global_fdr) < 0.05
                ),
                "source_rmu_p": source_rmu_p,
                "source_rmu_log2fc": record.get("source_rmu_vs_normal_log2fc"),
                "reextracted_mean_log2fc": record.get("mean_log2fc"),
                "dreams_identity_present": dreams_present,
                "dreams_name": dreams_name if dreams_present else "",
                "dreams_vs_source_name_conflict": name_conflict,
                "defensible_identity": record["defensible_identity"],
                "claim_ceiling": record["claim_ceiling"],
            }
        )

    audit = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "candidate_novelty_layers.csv", index=False)

    source_linked = audit[audit["source_exact_row_linked"]]
    algorithm_added = audit[
        audit["novelty_layer"] == "algorithm_added_candidate_absent_from_source_identity_table"
    ]
    report = {
        "status": "mtbls13729_biology_novelty_audit_complete",
        "formal": False,
        "candidates": int(len(audit)),
        "novelty_layer_counts": audit["novelty_layer"].value_counts().to_dict(),
        "source_exact_rows": int(len(source_linked)),
        "source_exact_rows_rmu_nominal_p_lt_0_05": int(source_linked["source_rmu_nominal_p_lt_0_05"].sum()),
        "source_exact_rows_global_fdr_lt_0_05": int(source_linked["source_global_fdr_lt_0_05"].sum()),
        "algorithm_added_absent_source_identity_count": int(len(algorithm_added)),
        "algorithm_added_absent_source_identities": algorithm_added["label"].tolist(),
        "source_listed_identity_new_peak_or_orthogonal_support": audit.loc[
            audit["novelty_layer"] == "source_listed_identity_new_peak_or_orthogonal_support", "label"
        ].tolist(),
        "identity_conflicts_requiring_source_override": audit.loc[
            audit["dreams_vs_source_name_conflict"], ["feature_id", "label", "dreams_name"]
        ].to_dict("records"),
        "downgraded_despite_identity_support": audit.loc[
            audit["novelty_layer"] == "source_identity_but_biology_downgraded", "label"
        ].tolist(),
        "novelty_claim": (
            "The principal biological novelty is module reconstruction and the addition of source-table-absent "
            "MS/MS-supported candidate families. Source-linked identities are same-cohort remapping or technical "
            "confirmation, not new metabolite discoveries or independent replication."
        ),
        "claim_limit": (
            "Absence from the source identity table does not establish a novel chemical entity. Family candidates "
            "remain putative until isomer/adduct review and authentic-standard confirmation."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
