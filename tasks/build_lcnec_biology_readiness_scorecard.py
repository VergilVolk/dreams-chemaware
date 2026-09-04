"""Build a machine-readable LCNEC biology manuscript readiness scorecard."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_biology_readiness_scorecard_v1"
SOURCES = {
    "same_universe": ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/report.json",
    "positive_control": ROOT / "data/validation/lcnec_hsst3n_source_positive_control_identity_v1/report.json",
    "triangulation": ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/report.json",
    "covariation": ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/report.json",
    "claim_audit": ROOT / "data/validation/lcnec_hsst3n_biology_claim_audit_v1/report.json",
    "external_transcript_validation": ROOT / "data/external/LCNEC_George2018_transcriptome/external_axis_validation_v1.json",
    "smoking_confounding": ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/report.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in SOURCES.items()}
    if not reports["claim_audit"]["pass"]:
        raise RuntimeError("claim audit must pass before readiness scoring")

    rows = [
        ("analytical_universe", "ready", "263 QC/blank/dilution-qualified families", "supports a fixed denominator"),
        ("source_biology_positive_control", "ready", "12/12 direction concordance; rho=0.902", "supports pipeline recovery"),
        ("same_denominator_algorithm_comparison", "ready", "42 source-overlap; 158 DreaMS candidates; 66 full evidence", "coverage and yield, not accuracy"),
        ("source_identity_positive_control", "limited", "17/19 structure concordance; Wilson 95% CI 68.6-97.1%", "small Level-2-biased subset"),
        ("author_unreported_hypotheses", "ready_level2", "4 frozen priorities absent from all 1,054 source identity rows under exact-name/structure and 5-ppm neutral-mass audit", "not exact identities or chemical novelty"),
        ("independent_pathway_context", "ready_context", "13/22 fixed proteins pass in 80 pure-LCNEC pairs", "not metabolite replication"),
        ("external_genomic_stratum_context", "ready_context", "3/3 frozen axes pass in 39 clean external LCNEC tumors", "tumor-only expression heterogeneity, not tumor-normal metabolite replication"),
        ("bioaware_context", "ready_abstention", "3 nonhub anchors; ADP hub abstained", "context cannot override identity"),
        ("patient_level_shared_module", "not_supported", "0/6 covariation pairs pass frozen gate", "group directions only"),
        ("objective_smoking_sensitivity", "ready_sensitivity", "0/4 priorities pass cotinine smoking-sensitivity gate in 34 pairs", "null sensitivity does not prove absence of smoking confounding"),
        ("exact_identity_level1", "missing", "0 new exact metabolite claims", "requires same-method authentic standards"),
        ("independent_metabolite_replication", "missing", "no independent LCNEC metabolite cohort", "protein and tumor-only transcript context are insufficient"),
        ("flux_or_enzyme_mechanism", "missing", "static abundance only", "requires tracing or perturbation"),
        ("clinical_biomarker", "not_tested", "no locked predictive endpoint", "requires independent clinical validation"),
    ]
    fieldnames = ["component", "status", "evidence", "claim_boundary"]
    with (OUT / "biology_readiness_scorecard.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    report = {
        "status": "lcnec_hsst3n_biology_readiness_scorecard_complete",
        "formal": True,
        "components": len(rows),
        "public_data_level2_application_ready": True,
        "new_exact_metabolite_claim_ready": False,
        "causal_mechanism_claim_ready": False,
        "highest_value_next_validation": ["quinolinic acid authentic standard", "ascorbic acid authentic standard", "independent LCNEC metabolomics"],
        "source_provenance": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for key, path in SOURCES.items()},
        "claim_limit": "Ready means manuscript evidence for bounded Level-2 hypothesis generation, not exact identity, flux, causality or clinical utility.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
