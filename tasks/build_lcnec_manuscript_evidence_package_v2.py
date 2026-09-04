"""Assemble the frozen LCNEC biology manuscript evidence package.

The package is intentionally pointer-rich: source artifacts retain their original
locations and hashes, while only reviewer-facing tables, figures and prose are copied.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_manuscript_evidence_package_v2"
SOURCES = {
    "readiness": ROOT / "data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json",
    "supplement_manifest": ROOT / "data/validation/lcnec_hsst3n_manuscript_supplement/supplement_manifest.json",
    "figure_report": ROOT / "data/validation/lcnec_hsst3n_manuscript_figures/figure_report.json",
    "annotation_benchmark": ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1/report.json",
    "identity_defense": ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1/report.json",
    "mechanism_coherence": ROOT / "data/validation/lcnec_hsst3n_mechanism_coherence_v1/report.json",
    "independent_proteomic_context": ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/report.json",
    "external_transcript_subtype": ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_subtype_audit_v1/report.json",
    "external_transcript_genomic": ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/report.json",
    "external_transcript_genomic_loo": ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_leave_one_gene_v1/report.json",
    "external_transcript_validation": ROOT / "data/external/LCNEC_George2018_transcriptome/external_axis_validation_v1.json",
    "same_universe_comparison": ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/report.json",
    "multicohort_triangulation": ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/report.json",
    "biology_claim_audit": ROOT / "data/validation/lcnec_hsst3n_biology_claim_audit_v1/report.json",
    "source_positive_control_identity": ROOT / "data/validation/lcnec_hsst3n_source_positive_control_identity_v1/report.json",
    "priority_patient_covariation": ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/report.json",
    "priority_technical_confounding": ROOT / "data/validation/lcnec_hsst3n_priority_technical_confounding_v1/report.json",
    "priority_smoking_confounding": ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/report.json",
    "priority_global_source_novelty": ROOT / "data/validation/lcnec_hsst3n_priority_global_source_novelty_v1/report.json",
    "results_draft": ROOT / "docs/LCNEC_MANUSCRIPT_RESULTS_DRAFT_20260831.md",
    "independent_proteomic_result": ROOT / "docs/LCNEC_INDEPENDENT_PROTEOGENOMIC_RESULT_20260901.md",
    "biology_novelty_audit": ROOT / "docs/LCNEC_BIOLOGY_NOVELTY_AUDIT_20260901.md",
    "external_novelty_update": ROOT / "docs/LCNEC_EXTERNAL_NOVELTY_UPDATE_20260901.md",
    "discussion_draft": ROOT / "docs/LCNEC_MANUSCRIPT_DISCUSSION_DRAFT_20260901.md",
    "abstract_draft": ROOT / "docs/LCNEC_MANUSCRIPT_ABSTRACT_DRAFT_20260901.md",
    "methods_outline": ROOT / "docs/LCNEC_MANUSCRIPT_METHODS_OUTLINE_20260901.md",
    "biology_readiness_scorecard": ROOT / "data/validation/lcnec_hsst3n_biology_readiness_scorecard_v1/report.json",
    "priority_formula_rivals": ROOT / "data/validation/lcnec_hsst3n_priority_formula_rivals_v1/report.json",
    "protein_axis_covariation": ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1/report.json",
    "manuscript_numeric_audit": ROOT / "data/validation/lcnec_hsst3n_manuscript_numeric_audit_v1/report.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_checked(src: Path, dst: Path) -> dict:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise RuntimeError(f"copy hash mismatch: {src}")
    return {"source": str(src.relative_to(ROOT)), "package": str(dst.relative_to(OUT)), "sha256": sha256(src)}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SOURCES.items() if path.suffix == ".json"}
    if not loaded["readiness"]["formal"] or not loaded["identity_defense"]["formal"]:
        raise RuntimeError("upstream formal evidence is not ready")
    if loaded["identity_defense"]["new_exact_metabolite_claims"] != 0:
        raise RuntimeError("exact metabolite claims are forbidden in this package")
    if loaded["external_transcript_validation"]["genomic_axis_gates_passing"] != 3:
        raise RuntimeError("external genomic-axis validation is not frozen and passing")

    files = {}
    files["results_draft"] = copy_checked(SOURCES["results_draft"], OUT / "MANUSCRIPT_RESULTS.md")
    files["biology_novelty_audit"] = copy_checked(
        SOURCES["biology_novelty_audit"], OUT / "BIOLOGY_NOVELTY_AUDIT.md"
    )
    files["external_novelty_update"] = copy_checked(
        SOURCES["external_novelty_update"], OUT / "EXTERNAL_NOVELTY_UPDATE.md"
    )
    files["discussion_draft"] = copy_checked(
        SOURCES["discussion_draft"], OUT / "MANUSCRIPT_DISCUSSION.md"
    )
    files["abstract_draft"] = copy_checked(
        SOURCES["abstract_draft"], OUT / "MANUSCRIPT_ABSTRACT.md"
    )
    files["methods_outline"] = copy_checked(
        SOURCES["methods_outline"], OUT / "MANUSCRIPT_METHODS.md"
    )
    files["biology_readiness_scorecard.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_biology_readiness_scorecard_v1/biology_readiness_scorecard.csv",
        OUT / "tables/biology_readiness_scorecard.csv",
    )
    files["annotation_figure_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1/lcnec_annotation_biology_benchmark.png",
        OUT / "figures/figure_annotation_recovery.png",
    )
    files["annotation_figure_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1/lcnec_annotation_biology_benchmark.pdf",
        OUT / "figures/figure_annotation_recovery.pdf",
    )
    for name in [
        "cross_platform_reproduction.pdf",
        "priority_pair_effects.pdf",
        "abundance_evidence_map.pdf",
        "priority_full_mirror_spectra.pdf",
        "priority_matched_fragment_mirrors.pdf",
    ]:
        files[name] = copy_checked(
            ROOT / "data/validation/lcnec_hsst3n_manuscript_figures" / name,
            OUT / "figures" / name,
        )
    for name in [
        "table_s1_81_dark_module_membership.csv",
        "table_s2_21_identity_hypotheses.csv",
        "table_s3_12_cross_platform_reproductions.csv",
        "table_s4_9_author_unreported_hypotheses.csv",
        "table_s5_4_priority_evidence_ledger.csv",
        "table_s6_priority_per_patient_effects.csv",
    ]:
        files[name] = copy_checked(
            ROOT / "data/validation/lcnec_hsst3n_manuscript_supplement" / name,
            OUT / "tables" / name,
        )
    files["annotation_benchmark_ledger.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1/annotation_benchmark_ledger.csv",
        OUT / "tables/annotation_benchmark_ledger.csv",
    )
    files["identity_claim_ledger.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1/priority_identity_claim_ledger.csv",
        OUT / "tables/identity_claim_ledger.csv",
    )
    files["reviewer_identity_defense.md"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1/REVIEWER_DEFENSE.md",
        OUT / "REVIEWER_IDENTITY_DEFENSE.md",
    )
    files["mechanism_axis_evidence_ledger.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_mechanism_coherence_v1/axis_evidence_ledger.csv",
        OUT / "tables/mechanism_axis_evidence_ledger.csv",
    )
    files["mechanism_axis_summary.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_mechanism_coherence_v1/axis_summary.csv",
        OUT / "tables/mechanism_axis_summary.csv",
    )
    files["mechanism_coherence_boundary.md"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_mechanism_coherence_v1/README.md",
        OUT / "MECHANISM_COHERENCE_BOUNDARY.md",
    )
    files["independent_proteomic_result.md"] = copy_checked(
        SOURCES["independent_proteomic_result"],
        OUT / "INDEPENDENT_PROTEOGENOMIC_CONTEXT.md",
    )
    files["independent_proteomic_figure_png"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/figures/independent_fixed_panel.png",
        OUT / "figures/figure_independent_proteomic_context.png",
    )
    files["independent_proteomic_figure_pdf"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/figures/independent_fixed_panel.pdf",
        OUT / "figures/figure_independent_proteomic_context.pdf",
    )
    files["independent_proteomic_protein_results.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/protein_results.csv",
        OUT / "tables/independent_proteomic_protein_results.csv",
    )
    files["independent_proteomic_axis_results.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/axis_results.csv",
        OUT / "tables/independent_proteomic_axis_results.csv",
    )
    files["external_transcript_subtype_figure_png"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_subtype_audit_v1/frozen_axis_subtype_audit.png",
        OUT / "figures/extended_external_transcript_subtype_context.png",
    )
    files["external_transcript_subtype_figure_pdf"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_subtype_audit_v1/frozen_axis_subtype_audit.pdf",
        OUT / "figures/extended_external_transcript_subtype_context.pdf",
    )
    files["external_transcript_subtype_axes.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_subtype_audit_v1/axis_subtype_results.csv",
        OUT / "tables/external_transcript_subtype_axis_results.csv",
    )
    files["external_transcript_genomic_figure_png"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/frozen_axis_genomic_audit.png",
        OUT / "figures/figure_external_genomic_axis_context.png",
    )
    files["external_transcript_genomic_figure_pdf"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/frozen_axis_genomic_audit.pdf",
        OUT / "figures/figure_external_genomic_axis_context.pdf",
    )
    files["external_transcript_genomic_axes.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/axis_genomic_results.csv",
        OUT / "tables/external_transcript_genomic_axis_results.csv",
    )
    files["external_transcript_genomic_genes.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/gene_genomic_results.csv",
        OUT / "tables/external_transcript_genomic_gene_results.csv",
    )
    files["external_transcript_genomic_loo.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_leave_one_gene_v1/leave_one_gene_results.csv",
        OUT / "tables/external_transcript_genomic_leave_one_gene_results.csv",
    )
    files["external_transcript_genomic_loo_summary.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_leave_one_gene_v1/axis_leave_one_gene_summary.csv",
        OUT / "tables/external_transcript_genomic_leave_one_gene_summary.csv",
    )
    files["three_cohort_mechanism_figure_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_three_cohort_mechanism_v1/lcnec_three_cohort_mechanism.png",
        OUT / "figures/figure_three_cohort_mechanism.png",
    )
    files["three_cohort_mechanism_figure_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_three_cohort_mechanism_v1/lcnec_three_cohort_mechanism.pdf",
        OUT / "figures/figure_three_cohort_mechanism.pdf",
    )
    files["same_universe_comparison_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/same_universe_annotation_comparison.png",
        OUT / "figures/extended_same_universe_annotation_comparison.png",
    )
    files["same_universe_comparison_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/same_universe_annotation_comparison.pdf",
        OUT / "figures/extended_same_universe_annotation_comparison.pdf",
    )
    files["same_universe_comparison.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/same_universe_comparison.csv",
        OUT / "tables/same_universe_annotation_comparison.csv",
    )
    files["same_universe_strata.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_same_universe_comparison_v1/stratified_summary.csv",
        OUT / "tables/same_universe_annotation_strata.csv",
    )
    files["multicohort_triangulation_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/multicohort_triangulation.png",
        OUT / "figures/figure_multicohort_triangulation.png",
    )
    files["multicohort_triangulation_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/multicohort_triangulation.pdf",
        OUT / "figures/figure_multicohort_triangulation.pdf",
    )
    files["candidate_triangulation.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/candidate_triangulation.csv",
        OUT / "tables/candidate_multicohort_triangulation.csv",
    )
    files["mechanism_triangulation.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/mechanism_triangulation.csv",
        OUT / "tables/mechanism_multicohort_triangulation.csv",
    )
    files["triangulation_readme.md"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/README.md",
        OUT / "MULTICOHORT_TRIANGULATION_BOUNDARY.md",
    )
    files["biology_claim_boundary_audit.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_biology_claim_audit_v1/claim_boundary_audit.csv",
        OUT / "tables/biology_claim_boundary_audit.csv",
    )
    files["source_positive_control_identity_ledger.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_source_positive_control_identity_v1/positive_control_identity_ledger.csv",
        OUT / "tables/source_positive_control_identity_ledger.csv",
    )
    files["priority_patient_covariation_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/priority_patient_covariation.png",
        OUT / "figures/extended_priority_patient_covariation.png",
    )
    files["priority_patient_covariation_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/priority_patient_covariation.pdf",
        OUT / "figures/extended_priority_patient_covariation.pdf",
    )
    files["priority_patient_covariation.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1/pairwise_covariation.csv",
        OUT / "tables/priority_patient_covariation.csv",
    )
    files["priority_technical_confounding_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_technical_confounding_v1/priority_technical_confounding.png",
        OUT / "figures/extended_priority_technical_confounding.png",
    )
    files["priority_technical_confounding_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_technical_confounding_v1/priority_technical_confounding.pdf",
        OUT / "figures/extended_priority_technical_confounding.pdf",
    )
    files["priority_technical_confounding.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_technical_confounding_v1/technical_confounding_tests.csv",
        OUT / "tables/priority_technical_confounding.csv",
    )
    files["priority_smoking_confounding_png"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/priority_smoking_confounding.png",
        OUT / "figures/extended_priority_smoking_confounding.png",
    )
    files["priority_smoking_confounding_pdf"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/priority_smoking_confounding.pdf",
        OUT / "figures/extended_priority_smoking_confounding.pdf",
    )
    files["priority_smoking_confounding.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_v1/priority_smoking_confounding_tests.csv",
        OUT / "tables/priority_smoking_confounding.csv",
    )
    files["priority_smoking_preregistration.json"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_smoking_confounding_preregistration_v1.json",
        OUT / "contracts/priority_smoking_confounding_preregistration.json",
    )
    files["priority_global_source_novelty_audit.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_global_source_novelty_v1/priority_global_source_novelty_audit.csv",
        OUT / "tables/priority_global_source_novelty_audit.csv",
    )
    files["priority_global_source_mass_matches.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_global_source_novelty_v1/priority_global_source_neutral_mass_matches.csv",
        OUT / "tables/priority_global_source_neutral_mass_matches.csv",
    )
    files["source_identity_resolution_ledger.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_global_source_novelty_v1/source_identity_resolution_ledger.csv",
        OUT / "tables/source_identity_resolution_ledger.csv",
    )
    files["priority_formula_rival_summary.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_formula_rivals_v1/priority_formula_rival_summary.csv",
        OUT / "tables/priority_formula_rival_summary.csv",
    )
    files["priority_top5_formula_rivals.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_priority_formula_rivals_v1/priority_top5_formula_rivals.csv",
        OUT / "tables/priority_top5_formula_rivals.csv",
    )
    files["protein_axis_covariation_png"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1/protein_axis_covariation.png",
        OUT / "figures/extended_independent_protein_axis_covariation.png",
    )
    files["protein_axis_covariation_pdf"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1/protein_axis_covariation.pdf",
        OUT / "figures/extended_independent_protein_axis_covariation.pdf",
    )
    files["protein_axis_covariation.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1/all_within_axis_pairwise_covariation.csv",
        OUT / "tables/independent_protein_axis_covariation.csv",
    )
    files["protein_axis_covariation_summary.csv"] = copy_checked(
        ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1/axis_covariation_summary.csv",
        OUT / "tables/independent_protein_axis_covariation_summary.csv",
    )
    files["manuscript_numeric_consistency_audit.csv"] = copy_checked(
        ROOT / "data/validation/lcnec_hsst3n_manuscript_numeric_audit_v1/numeric_consistency_audit.csv",
        OUT / "tables/manuscript_numeric_consistency_audit.csv",
    )

    claims = [
        {
            "claim_id": "C1",
            "claim": "QC/blank/dilution-qualified raw HSST3n contains source-table-absent analytical headroom",
            "supported": True,
            "evidence": "263 qualified families; 42 source-table matches; 221 source-table-absent",
            "boundary": "absence from source table is not metabolite novelty",
        },
        {
            "claim_id": "C2",
            "claim": "The algorithmic workflow recovers cross-platform biological direction",
            "supported": True,
            "evidence": "12/12 direction concordance; Spearman rho=0.902; 10 author FDR<0.05",
            "boundary": "positive control, not independent cohort replication",
        },
        {
            "claim_id": "C3",
            "claim": "Four author-unreported Level-2/connectivity-family hypotheses show strong paired abundance effects",
            "supported": True,
            "evidence": "absent across all 1,054 source identity rows under exact-name/structure and 5-ppm neutral-mass audit; formula <=1.51 ppm; 5-20 fragments; 28-33/34 concordant pairs",
            "boundary": "no Level-1 or chemical-novelty claim",
        },
        {
            "claim_id": "C4",
            "claim": "Measured pools are consistent with nucleotide/NAD-related and antioxidant remodeling",
            "supported": True,
            "evidence": "coherent measured abundance families plus three non-hub BioAware anchors",
            "boundary": "no flux, enzyme-activity or causal-dependency claim",
        },
        {
            "claim_id": "C5",
            "claim": "The four metabolites are independently replicated in LCNEC",
            "supported": False,
            "evidence": "not available",
            "boundary": "external proteogenomics is context only",
        },
        {
            "claim_id": "C6",
            "claim": "A frozen independent protein panel supports ADP-ribose/PARP, NAD-redistribution and mixed-redox contexts",
            "supported": True,
            "evidence": "103 protein pairs; 80 pure-LCNEC paired primary; 13/22 proteins pass fixed BH and stability gates",
            "boundary": "protein abundance is context, not metabolite replication, flux or causality",
        },
        {
            "claim_id": "C7",
            "claim": "Frozen multi-cohort triangulation assigns distinct validation roles to the four priorities",
            "supported": True,
            "evidence": "ADP-ribose/PARP strongest context; quinolinate highest standard priority; ascorbate largest effect; ADP family-level sentinel",
            "boundary": "role prioritization does not upgrade identity or establish flux",
        },
        {
            "claim_id": "C8",
            "claim": "The four priority effects form a confirmed patient-level metabolic module",
            "supported": False,
            "evidence": "0/6 correlations passed the frozen joint gate; ADP--ADP-ribose rho=0.373, q=0.101",
            "boundary": "group-level directional coherence is not patient-level module confirmation",
        },
        {
            "claim_id": "C9",
            "claim": "Top-ranked spectral candidates eliminate same-formula alternatives for the four priorities",
            "supported": False,
            "evidence": "3/4 have same-formula top-five rivals; ADP-ribose has incomplete rival-spectrum coverage",
            "boundary": "ranking supports prioritization, not chemical uniqueness",
        },
        {
            "claim_id": "C10",
            "claim": "Recorded tissue amount or injection order explains a frozen priority effect",
            "supported": False,
            "evidence": "0/16 fixed tests passed; minimum BH q=0.378; tumor-after-normal pairs=17/34",
            "boundary": "the public workbook lacks stage, smoking, sex and tumor-purity covariates",
        },
        {
            "claim_id": "C11",
            "claim": "The three frozen pathway-context axes differ across expression-independent genomic strata in an external LCNEC cohort",
            "supported": True,
            "evidence": "39 clean tumors (22 STK11/KEAP1-altered, 17 RB1-altered); all three axes pass fixed gates; leave-one-gene: redox 8/8, NAD 8/9, ADP-ribose 4/5 omissions pass",
            "boundary": "tumor-only expression context; not tumor-normal direction replication, metabolite validation, flux, prognosis or causality",
        },
        {
            "claim_id": "C12",
            "claim": "An objective smoking-exposure proxy explains a frozen priority effect",
            "supported": False,
            "evidence": "0/4 priorities pass the preregistered joint cotinine gate; 11 cotinine-classified smokers and 23 non-smokers",
            "boundary": "a null sensitivity audit does not prove absence of smoking confounding; tumor cotinine is not identity or mechanism validation",
        },
    ]
    with (OUT / "tables/claim_evidence_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(claims[0]))
        writer.writeheader()
        writer.writerows(claims)

    report = {
        "status": "lcnec_hsst3n_manuscript_evidence_package_v2_complete",
        "formal": True,
        "primary_biology_dataset": "LCNEC 34 paired tumor-adjacent tissues, HSST3n raw arm",
        "backup_transfer_dataset": "MTBLS13729 mucinous colorectal tissue",
        "headline_numbers": {
            "source_all_platform_declared_metabolites": 1052,
            "source_annotation_rate_available": False,
            "qc_qualified_hsst3n_families": 263,
            "source_table_absent_families": 221,
            "frozen_dark_modules": 81,
            "official_dreams_candidate_coverage": 51,
            "dreams_p2b_agreement": 45,
            "multi_evidence_retained": 22,
            "cross_platform_reproduced": 12,
            "author_unreported": 9,
            "priority_hypotheses": 4,
            "new_exact_metabolite_claims": 0,
            "same_universe_dreams_candidates": 158,
            "same_universe_full_evidence_retained": 66,
            "independent_proteomic_pairs": 103,
            "independent_pure_lcnec_pairs": 80,
            "independent_fixed_proteins_passing": 13,
            "biology_claims_audited": 19,
            "biology_claims_passing": 19,
            "source_positive_control_structure_resolvable": 19,
            "source_positive_control_concordant": 17,
            "source_positive_control_same_formula_errors_retained": 2,
            "priority_covariation_pairs_passing": 0,
            "biology_readiness_components": 14,
            "public_data_level2_application_ready": True,
            "new_exact_metabolite_claim_ready": False,
            "priorities_with_same_formula_top5_rivals": 3,
            "exploratory_protein_covariation_pairs_passing": 12,
            "recorded_technical_confounding_tests_passing": 0,
            "objective_smoking_sensitive_priorities": 0,
            "priorities_absent_global_source_resolver": 4,
            "source_rows_neutral_mass_reconstructable": 1050,
            "external_transcript_lcnec_tumors": 66,
            "external_clean_genomic_tumors": 39,
            "external_genomic_axes_passing": 3,
            "external_genomic_genes_passing": 4,
            "external_genomic_axes_fully_leave_one_gene_robust": 1,
            "manuscript_numeric_checks_passing": 38,
        },
        "primary_claim": "In 34 paired LCNEC tissues, an algorithm-enabled workflow recovered a cross-platform-reproduced abundance context and generated four author-unreported Level-2/connectivity-family hypotheses with distinct, independently contextualized follow-up roles.",
        "identity_boundary": loaded["identity_defense"]["identity_rule"],
        "claim_matrix": claims,
        "files": files,
        "source_provenance": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in SOURCES.items()
        },
        "open_high_value_gap": "Independent metabolite-level replication and same-method authentic-standard RT remain unavailable; the completed independent protein and tumor-only transcriptomic audits supply pathway context only.",
        "claim_limit": loaded["identity_defense"]["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
