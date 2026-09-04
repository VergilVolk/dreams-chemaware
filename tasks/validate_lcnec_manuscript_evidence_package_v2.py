"""Validate the packaged LCNEC manuscript evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_manuscript_evidence_package_v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    assert report["formal"] is True
    assert report["headline_numbers"]["priority_hypotheses"] == 4
    assert report["headline_numbers"]["new_exact_metabolite_claims"] == 0
    assert report["headline_numbers"]["source_annotation_rate_available"] is False
    assert sum(bool(row["supported"]) for row in report["claim_matrix"]) == 7
    assert len(report["claim_matrix"]) == 12
    assert report["headline_numbers"]["independent_proteomic_pairs"] == 103
    assert report["headline_numbers"]["independent_fixed_proteins_passing"] == 13
    assert report["headline_numbers"]["same_universe_dreams_candidates"] == 158
    assert report["headline_numbers"]["same_universe_full_evidence_retained"] == 66
    assert report["headline_numbers"]["biology_claims_audited"] == 19
    assert report["headline_numbers"]["biology_claims_passing"] == 19
    assert report["headline_numbers"]["source_positive_control_structure_resolvable"] == 19
    assert report["headline_numbers"]["source_positive_control_concordant"] == 17
    assert report["headline_numbers"]["source_positive_control_same_formula_errors_retained"] == 2
    assert report["headline_numbers"]["priority_covariation_pairs_passing"] == 0
    assert report["headline_numbers"]["biology_readiness_components"] == 14
    assert report["headline_numbers"]["public_data_level2_application_ready"] is True
    assert report["headline_numbers"]["new_exact_metabolite_claim_ready"] is False
    assert report["headline_numbers"]["priorities_with_same_formula_top5_rivals"] == 3
    assert report["headline_numbers"]["exploratory_protein_covariation_pairs_passing"] == 12
    assert report["headline_numbers"]["manuscript_numeric_checks_passing"] == 38
    assert report["headline_numbers"]["recorded_technical_confounding_tests_passing"] == 0
    assert report["headline_numbers"]["objective_smoking_sensitive_priorities"] == 0
    assert report["headline_numbers"]["priorities_absent_global_source_resolver"] == 4
    assert report["headline_numbers"]["source_rows_neutral_mass_reconstructable"] == 1050
    assert report["headline_numbers"]["external_transcript_lcnec_tumors"] == 66
    assert report["headline_numbers"]["external_clean_genomic_tumors"] == 39
    assert report["headline_numbers"]["external_genomic_axes_passing"] == 3
    assert report["headline_numbers"]["external_genomic_genes_passing"] == 4
    assert report["headline_numbers"]["external_genomic_axes_fully_leave_one_gene_robust"] == 1
    for item in report["files"].values():
        path = OUT / item["package"]
        assert path.is_file() and path.stat().st_size > 0
        assert sha256(path) == item["sha256"]
    assert (OUT / "tables/claim_evidence_matrix.csv").stat().st_size > 300
    print(f"[validate_lcnec_manuscript_evidence_package_v2] PASS files={len(report['files'])}")


if __name__ == "__main__":
    main()
