"""Quantify what the expanded biology ledger adds beyond the source paper.

This audit never divides selected candidates by all detected features and never
calls a same-cohort orthogonal recovery a new metabolite.  It reports distinct
forms of value: identity remapping, orthogonal recovery, source-table-absent
family candidates, and deliberate downgrades.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
SUPPLEMENT = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
OUT = ROOT / "data/mtbls13729/original_paper_delta_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    return " ".join(str(value).strip().casefold().replace("-", " ").split())


def main() -> None:
    if not LEDGER.is_file() or not SUPPLEMENT.is_file():
        raise FileNotFoundError("expanded ledger or source supplement is missing")
    OUT.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_csv(LEDGER)
    source = pd.read_excel(SUPPLEMENT, sheet_name="metabolites", header=1)
    source = source.loc[pd.to_numeric(source["m/z"], errors="coerce").notna()].copy()
    source["normalized_name"] = source["metabolites"].map(normalize_name)
    source_names = set(source.normalized_name)

    rows: list[dict[str, object]] = []
    for item in ledger.itertuples(index=False):
        source_name = "" if pd.isna(item.source_name) else str(item.source_name).strip()
        source_linked = bool(source_name)
        exact_name_present = source_linked and normalize_name(source_name) in source_names
        tier = str(item.manuscript_evidence_tier)
        if tier == "A_source_identity_remap":
            delta_type = "source_identity_remap"
            novelty = "new assay/panel evidence; not new chemistry"
        elif tier == "A_source_identity_orthogonal_recovery":
            delta_type = "orthogonal_level1_recovery"
            novelty = "same-cohort orthogonal recovery of a source Level-1 identity"
        elif tier == "B_strong_family_candidate":
            delta_type = "source_table_absent_family_candidate"
            novelty = "new candidate ion family; exact chemical entity unresolved"
        else:
            delta_type = "downgraded_or_control"
            novelty = "negative/control result retained to prevent selective reporting"

        rows.append(
            {
                "feature_id": int(item.feature_id),
                "label": item.label,
                "module": item.module,
                "manuscript_evidence_tier": tier,
                "delta_type": delta_type,
                "source_linked": source_linked,
                "source_name": source_name,
                "published_source_msi": "" if pd.isna(item.published_source_msi) else item.published_source_msi,
                "source_exact_name_found_in_345_row_table": exact_name_present,
                "discovery_panel": item.discovery_panel,
                "pairs": item.pairs,
                "mean_log2fc": item.mean_log2fc,
                "positive_pairs": item.positive_pairs,
                "peak_resolved_ms2_spectra": item.peak_resolved_ms2_spectra,
                "classical_median_cosine": item.classical_median_cosine,
                "novelty_interpretation": novelty,
                "maximum_claim": item.claim_ceiling,
            }
        )

    audit = pd.DataFrame(rows)
    if len(audit) != 18 or audit.feature_id.nunique() != 18:
        raise RuntimeError("expanded ledger must contain exactly 18 unique candidate features")
    audit.to_csv(OUT / "candidate_original_paper_delta_v2.csv", index=False)

    counts = audit.delta_type.value_counts().to_dict()
    report = {
        "status": "mtbls13729_original_paper_delta_v2_complete",
        "formal": False,
        "source_paper_annotation_rows": int(len(source)),
        "source_paper_level_counts": {
            str(key): int(value)
            for key, value in source["MSI(Metabolomics Standards Initiative)"].value_counts(dropna=False).items()
        },
        "expanded_selected_candidates": int(len(audit)),
        "delta_types": {str(key): int(value) for key, value in counts.items()},
        "source_linked_candidates": int(audit.source_linked.sum()),
        "source_exact_names_recovered": int(audit.source_exact_name_found_in_345_row_table.sum()),
        "source_table_absent_family_candidates": int((audit.delta_type == "source_table_absent_family_candidate").sum()),
        "orthogonal_level1_recoveries": int((audit.delta_type == "orthogonal_level1_recovery").sum()),
        "downgraded_or_controls": int((audit.delta_type == "downgraded_or_control").sum()),
        "annotation_rate_statement": (
            "No global annotation-rate increase is estimable from this selected 18-candidate ledger. "
            "The source paper's 345-row annotation table and the selected evidence panel have different denominators."
        ),
        "algorithm_value_statement": (
            "The auditable increment is candidate-level: cross-panel recovery and raw-MS2 evidence for known "
            "identities, five source-table-absent ion-family hypotheses, and explicit downgrading of a false/weak node."
        ),
        "claim_limit": (
            "Source-table absence is not proof of a new metabolite. Orthogonal recoveries use the same patient "
            "cohort and are not independent biological replication. Exact family candidates require standards."
        ),
        "provenance": {
            "ledger_sha256": sha256(LEDGER),
            "supplement_sha256": sha256(SUPPLEMENT),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
