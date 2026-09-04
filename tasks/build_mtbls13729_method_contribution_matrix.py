"""Attribute each frozen MTBLS13729 biology candidate to evidence modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
OUT = ROOT / "data/mtbls13729/annotation_biology_benchmark_v1"


def text_present(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    if len(ledger) != 18 or ledger.feature_id.nunique() != 18:
        raise RuntimeError("expected the frozen 18-candidate biology ledger")

    source = text_present(ledger.source_name)
    dreams = text_present(ledger.dreams_tier)
    classical = text_present(ledger.classical_library_name)
    tier = ledger.manuscript_evidence_tier.astype(str)
    family = ledger.feature_id.isin([1597, 3019])
    downgrade = tier.eq("C_downgraded_or_control")

    # These agreements are frozen candidate-level facts from the integrated
    # ledger.  A non-empty DreaMS assignment is not automatically agreement:
    # feature 722, for example, carries a synephrine vote while the source
    # Level-1 identity is phenylalanine.
    dreams_identity_agreement = ledger.feature_id.isin([150, 398, 73, 732, 428])

    matrix = ledger.loc[:, [
        "feature_id", "label", "module", "defensible_identity", "manuscript_evidence_tier",
        "pairs", "mean_log2fc", "positive_pairs", "peak_resolved_ms2_spectra",
        "dreams_name", "dreams_tier", "classical_library_name", "source_name",
        "published_source_msi", "claim_ceiling",
    ]].copy()
    matrix["author_identity_row_present"] = source
    matrix["author_identity_positive_anchor"] = source & ~downgrade
    matrix["official_dreams_assignment_present"] = dreams
    matrix["official_dreams_identity_agreement"] = dreams_identity_agreement
    matrix["official_dreams_positive_usable_agreement"] = dreams_identity_agreement & ~downgrade
    matrix["classical_library_orthogonal_support"] = classical
    matrix["bioaware_ion_family_consolidation"] = family
    matrix["source_table_absent_family_candidate"] = tier.eq("B_strong_family_candidate")
    matrix["deliberate_downgrade_control"] = downgrade
    matrix["paired_ms1_requantification"] = pd.to_numeric(ledger.pairs, errors="coerce").fillna(0).gt(0)
    matrix["peak_resolved_raw_ms2"] = pd.to_numeric(ledger.peak_resolved_ms2_spectra, errors="coerce").fillna(0).gt(0)
    matrix["annotation_increment_type"] = np.select(
        [
            tier.eq("A_source_identity_orthogonal_recovery"),
            tier.eq("A_source_identity_remap"),
            tier.eq("B_strong_family_candidate"),
            tier.eq("C_downgraded_or_control"),
        ],
        [
            "same-cohort orthogonal recovery",
            "source identity remap",
            "source-table-absent candidate family",
            "negative/control downgrade",
        ],
        default="unclassified",
    )
    matrix["new_exact_metabolite_claim_permitted"] = False
    matrix.to_csv(OUT / "candidate_method_contribution_matrix.csv", index=False)

    module_rows = []
    for module, frame in matrix.groupby("module", sort=False):
        module_rows.append(
            {
                "module": module,
                "candidates": int(len(frame)),
                "author_identity_rows": int(frame.author_identity_row_present.sum()),
                "author_positive_anchors": int(frame.author_identity_positive_anchor.sum()),
                "official_dreams_assignments": int(frame.official_dreams_assignment_present.sum()),
                "official_dreams_identity_agreements": int(frame.official_dreams_identity_agreement.sum()),
                "official_dreams_positive_usable_agreements": int(
                    frame.official_dreams_positive_usable_agreement.sum()
                ),
                "classical_orthogonal": int(frame.classical_library_orthogonal_support.sum()),
                "bioaware_family_anchors": int(frame.bioaware_ion_family_consolidation.sum()),
                "source_table_absent_family_candidates": int(frame.source_table_absent_family_candidate.sum()),
                "mean_feature_log2fc_descriptive": float(pd.to_numeric(frame.mean_log2fc, errors="coerce").mean()),
                "minimum_positive_pair_fraction": float(
                    (pd.to_numeric(frame.positive_pairs, errors="coerce") / pd.to_numeric(frame.pairs, errors="coerce")).min()
                ),
                "claim_boundary": "module summary is descriptive; features are not independent and abundance does not upgrade identity",
            }
        )
    modules = pd.DataFrame(module_rows)
    modules.to_csv(OUT / "biology_module_method_summary.csv", index=False)

    report = {
        "status": "mtbls13729_method_contribution_matrix_complete",
        "formal": True,
        "candidates": int(len(matrix)),
        "modules": int(matrix.module.nunique()),
        "source_identity_rows": int(matrix.author_identity_row_present.sum()),
        "source_identity_positive_anchors": int(matrix.author_identity_positive_anchor.sum()),
        "source_identity_downgraded_controls": int(
            (matrix.author_identity_row_present & matrix.deliberate_downgrade_control).sum()
        ),
        "official_dreams_assignments_present": int(matrix.official_dreams_assignment_present.sum()),
        "official_dreams_identity_agreements": int(matrix.official_dreams_identity_agreement.sum()),
        "official_dreams_positive_usable_agreements": int(
            matrix.official_dreams_positive_usable_agreement.sum()
        ),
        "classical_library_orthogonal_candidates": int(matrix.classical_library_orthogonal_support.sum()),
        "bioaware_family_anchors": int(matrix.bioaware_ion_family_consolidation.sum()),
        "source_table_absent_family_candidates": int(matrix.source_table_absent_family_candidate.sum()),
        "deliberate_downgrades": int(matrix.deliberate_downgrade_control.sum()),
        "paired_ms1_requantified": int(matrix.paired_ms1_requantification.sum()),
        "with_peak_resolved_raw_ms2": int(matrix.peak_resolved_raw_ms2.sum()),
        "new_exact_metabolite_claims": 0,
        "claim_limit": (
            "Module attribution describes evidence provenance. Source-table absence, DreaMS consensus, "
            "BioAware consolidation and abundance effects do not establish an exact new metabolite identity."
        ),
    }
    (OUT / "method_contribution_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
