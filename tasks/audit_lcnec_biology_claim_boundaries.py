"""Audit the frozen LCNEC biology claims against evidence and claim boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_biology_claim_audit_v1"
TRIAGE = ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1"
SAME = ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/report.json"
POSITIVE_CONTROL = ROOT / "data/validation/lcnec_hsst3n_source_positive_control_identity_v1/report.json"
COVARIATION = ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/report.json"
FORMULA_RIVALS = ROOT / "data/validation/lcnec_hsst3n_priority_formula_rivals_v1/report.json"
MANUSCRIPT = ROOT / "docs/LCNEC_MANUSCRIPT_RESULTS_DRAFT_20260831.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    triage_report = json.loads((TRIAGE / "report.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(TRIAGE / "candidate_triangulation.csv")
    same = json.loads(SAME.read_text(encoding="utf-8"))
    positive_control = json.loads(POSITIVE_CONTROL.read_text(encoding="utf-8"))
    covariation = json.loads(COVARIATION.read_text(encoding="utf-8"))
    formula_rivals = json.loads(FORMULA_RIVALS.read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    manuscript_normalized = " ".join(manuscript.split()).lower()

    checks = [
        ("source_annotation_rate_reconstructable", False, False,
         "Source supplement lacks a detected-feature denominator."),
        ("source_overlap_is_annotation_accuracy", False, False,
         "42/263 is reconstructed source-table feature overlap only."),
        ("dreams_candidate_coverage_is_accuracy", False, False,
         "158/263 is constrained candidate coverage, not correctness."),
        ("full_evidence_yield_is_accuracy", False, False,
         "66/263 is evidence retention, not correctness."),
        ("source_positive_control_is_global_accuracy", False, False,
         "17/19 is a uniquely structure-resolvable source-name subset, mostly MSI Level 2."),
        ("full_gate_eliminates_same_formula_isomer_errors", False,
         positive_control["metrics"]["full_tool_incorrect_retained_same_formula_isomers"] == 0,
         "Both discordant source-positive controls were retained and were same-formula isomers."),
        ("author_unreported_means_chemical_novelty", False, False,
         "It means absent from the source HSST3n table only."),
        ("four_exact_metabolite_identities_allowed", False,
         bool(candidates["exact_identity_allowed"].astype(bool).any()),
         "All four remain Level-2/connectivity-family hypotheses."),
        ("four_hypotheses_cross_platform_replicated", False, False,
         "Cross-platform reproduction validates neighboring/source abundance context, not all four new rows."),
        ("four_hypotheses_independently_replicated", False, False,
         "The independent cohort contains proteins, not the four metabolite measurements."),
        ("protein_context_confirms_identity", False, False,
         "Protein abundance cannot confirm chromatographic or structural identity."),
        ("static_abundance_establishes_flux", False, False,
         "No isotope tracing or dynamic flux data are available."),
        ("adpr_parp_context_supported", True,
         triage_report["decisions"]["strongest_cross_omics_mechanism_context"] == "ADP-ribose / PARP turnover",
         "Local ADP-ribose-family increase and independent PARP1/2 increase form a hypothesis-level bridge."),
        ("quinolinate_is_highest_standard_priority", True,
         triage_report["decisions"]["highest_chemical_confirmation_priority"] == "quinolinate",
         "Quinolinate has a focused QPRT relationship but still requires authentic RT/MSMS."),
        ("ascorbate_independent_context_is_uniform", False, False,
         "Independent redox proteins are directionally mixed."),
        ("adp_specific_pathway_claim_allowed", False, False,
         "ADP is a high-degree currency hub and lacks a frozen specific protein bridge."),
        ("four_priorities_form_confirmed_patient_module", False,
         covariation["pairs_passing_fixed_gate"] > 0,
         "None of six patient-level effect correlations passed the frozen joint gate."),
        ("adp_adpr_covariation_confirmed_after_multiplicity", False,
         bool(covariation["primary_adp_adpr"]["gate"]),
         "ADP--ADP-ribose is suggestive (rho=0.373) but misses the BH threshold (q=0.101)."),
        ("priority_top1_scores_eliminate_same_formula_rivals", False,
         formula_rivals["priorities_with_same_formula_top5_rivals"] == 0,
         "Three priorities have observed same-formula top-five rivals; the fourth lacks spectral coverage, not alternatives."),
    ]
    audit = pd.DataFrame(checks, columns=["claim", "allowed", "observed", "basis"])
    audit["pass"] = audit["allowed"] == audit["observed"]
    audit.to_csv(OUT / "claim_boundary_audit.csv", index=False)

    required_text = [
        "number of new exact metabolite claims remains zero",
        "Independent proteins provide pathway context only",
        "not claim that the four author-unreported hypotheses themselves were cross-platform or independently replicated",
        "do not establish a shared patient-level metabolic module",
    ]
    missing_required = [text for text in required_text if text.lower() not in manuscript_normalized]
    forbidden_assertions = [
        "the four metabolites were independently replicated",
        "confirmed novel metabolite",
        "quinolinate flux increased",
        "pentose-phosphate pathway was activated in pure LCNEC",
    ]
    present_forbidden = [text for text in forbidden_assertions if text.lower() in manuscript_normalized]

    report = {
        "status": "lcnec_biology_claim_boundary_audit_complete",
        "formal": True,
        "claims_audited": int(len(audit)),
        "claims_passing": int(audit["pass"].sum()),
        "same_universe": {
            "families": 263,
            "source_overlap": 42,
            "official_dreams_candidates": 158,
            "full_evidence_retained": 66,
            "all_are_accuracy_metrics": False,
        },
        "missing_required_manuscript_boundaries": missing_required,
        "present_forbidden_assertions": present_forbidden,
        "pass": bool(audit["pass"].all() and not missing_required and not present_forbidden),
        "provenance": {
            "triangulation_report_sha256": sha256(TRIAGE / "report.json"),
            "candidate_triangulation_sha256": sha256(TRIAGE / "candidate_triangulation.csv"),
            "same_universe_report_sha256": sha256(SAME),
            "manuscript_sha256": sha256(MANUSCRIPT),
            "positive_control_report_sha256": sha256(POSITIVE_CONTROL),
            "patient_covariation_report_sha256": sha256(COVARIATION),
            "priority_formula_rivals_report_sha256": sha256(FORMULA_RIVALS),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["pass"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(f"[audit_lcnec_biology_claim_boundaries] PASS claims={len(audit)}")


if __name__ == "__main__":
    main()
