"""Build a fail-closed manuscript-readiness ledger for the LCNEC result.

This audit does not discover or refit any result. It only verifies the frozen
outputs that support the algorithm-enabled Level-2 biology manuscript and
separates them from evidence that is still absent (Level-1 identity, external
metabolite replication, and causal/flux validation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required frozen report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-root", type=Path, default=Path("data/validation")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_manuscript_readiness"),
    )
    args = parser.parse_args()

    root = args.validation_root
    paths = {
        "acquisition": root / "lcnec_hsst3n_acquisition_gate/acquisition_gate.json",
        "qc_headroom": root / "lcnec_hsst3n_qc_headroom_gate/qc_headroom_gate.json",
        "dark_eic": root / "lcnec_hsst3n_dark_eic_gate/dark_eic_gate.json",
        "normalization_robustness": root / "lcnec_hsst3n_dark_robustness_gate/robustness_gate.json",
        "known_feature_positive_control": root / "lcnec_hsst3n_known_eic_validation/known_eic_validation.json",
        "annotation_biology": root / "lcnec_hsst3n_annotation_biology/biology_report.json",
        "priority_structure": root / "lcnec_hsst3n_priority_structure/structure_report.json",
        "priority_pair_consistency": root / "lcnec_hsst3n_priority_pair_consistency/pair_consistency_report.json",
        "bioaware_context": root / "lcnec_hsst3n_bioaware_context/bioaware_context_report.json",
        "priority_adduct_audit": root / "lcnec_hsst3n_priority_adduct_audit/priority_adduct_report.json",
        "manuscript_figures": root / "lcnec_hsst3n_manuscript_figures/figure_report.json",
    }
    frozen = {name: load_json(path) for name, path in paths.items()}

    require(frozen["acquisition"].get("pass_to_feature_headroom") is True, "acquisition gate failed")
    require(frozen["qc_headroom"].get("pass_to_author_overlap_audit") is True, "QC/headroom gate failed")
    require(frozen["dark_eic"].get("pass_to_annotation_and_mechanism_module") is True, "dark-EIC gate failed")
    require(frozen["normalization_robustness"].get("pass_to_identity_annotation") is True, "normalization gate failed")
    require(frozen["known_feature_positive_control"].get("pass") is True, "known-feature positive control failed")
    require(frozen["annotation_biology"].get("pass_to_structure_confirmation") is True, "annotation/biology gate failed")
    require(frozen["priority_structure"].get("pass") is True, "priority structure gate failed")
    require(frozen["priority_pair_consistency"].get("pass") is True, "paired abundance gate failed")
    require(frozen["bioaware_context"].get("pass") is True, "BioAware context/abstention gate failed")
    require(
        frozen["priority_adduct_audit"].get("hypotheses_with_common_spacing_flags") == 0,
        "common isotope/adduct spacing audit raised a flag",
    )
    require(
        frozen["manuscript_figures"].get("status") == "lcnec_hsst3n_manuscript_figures_complete",
        "manuscript figure bundle is incomplete",
    )

    biology = frozen["annotation_biology"]
    structure = frozen["priority_structure"]
    pairs = frozen["priority_pair_consistency"]
    figures = frozen["manuscript_figures"]
    require(biology.get("published_atlas_overlaps", 0) >= 8, "too few cross-platform overlaps")
    require(
        biology.get("cross_platform_reproduction", {}).get("direction_concordance", 0) >= 0.8,
        "cross-platform direction concordance failed",
    )
    require(len(biology.get("priority_author_unreported_hypotheses", [])) >= 3, "too few priority hypotheses")
    require(structure.get("formula_mass_pass_5ppm") == 4, "not all priority formulas pass 5 ppm")
    require(structure.get("fragment_support_pass") == 4, "not all priorities have direct fragment support")
    require(pairs.get("pairs") == 34, "paired-tissue panel is not the frozen 34-pair cohort")
    require(pairs.get("all_leave_one_pair_out_sign_stable") is True, "LOPO direction is unstable")
    require(figures.get("cross_platform", {}).get("n") == 12, "figure/report overlap count mismatch")

    passed_evidence = {
        "raw_acquisition_and_controls_audited": True,
        "pooled_qc_blank_and_dilution_headroom_passed": True,
        "targeted_dark_feature_requantification_passed": True,
        "cross_normalization_robustness_passed": True,
        "known_feature_positive_control_passed": True,
        "full_81_module_annotation_audited": True,
        "cross_platform_reproduction_passed": True,
        "four_priority_formula_and_fragment_checks_passed": True,
        "four_priority_34_pair_consistency_passed": True,
        "bioaware_context_and_hub_abstention_passed": True,
        "common_adduct_spacing_screen_clear": True,
        "manuscript_figure_bundle_complete": True,
    }
    hard_evidence_available = {
        "authentic_standard_retention_time_level1_identity": False,
        "independent_lcnec_metabolite_abundance_replication": False,
        "causal_perturbation_or_isotope_tracing": False,
        "external_proteogenomic_small_panel_validation": False,
    }

    report = {
        "status": "lcnec_hsst3n_manuscript_readiness_complete",
        "formal": True,
        "decision": "primary_algorithm_enabled_level2_biology_manuscript_candidate",
        "cohort": {"paired_tumor_adjacent": 34, "platform": "HSST3n"},
        "frozen_result": {
            "nonredundant_dark_modules": 81,
            "consistency_filtered_features": biology["consistency_filtered_features"],
            "connectivity_hypotheses": biology["unique_connectivity_hypotheses"],
            "cross_platform_overlaps": biology["published_atlas_overlaps"],
            "cross_platform_direction_concordance": biology["cross_platform_reproduction"]["direction_concordance"],
            "cross_platform_spearman_rho": biology["cross_platform_reproduction"]["spearman_effect_rho"],
            "author_unreported_hypotheses": biology["author_unreported_hypotheses"],
            "priority_hypotheses": biology["priority_author_unreported_hypotheses"],
        },
        "passed_evidence": passed_evidence,
        "hard_evidence_available": hard_evidence_available,
        "unresolved_hard_gaps": [
            name for name, available in hard_evidence_available.items() if not available
        ],
        "readiness": {
            "ready_for_algorithm_enabled_level2_biology_manuscript": all(passed_evidence.values()),
            "ready_for_level1_identity": hard_evidence_available["authentic_standard_retention_time_level1_identity"],
            "ready_for_independent_metabolite_replication_claim": hard_evidence_available["independent_lcnec_metabolite_abundance_replication"],
            "ready_for_causal_metabolism_claim": hard_evidence_available["causal_perturbation_or_isotope_tracing"],
        },
        "allowed_primary_claim": (
            "In 34 paired LCNEC tissues, an algorithm-enabled untargeted workflow recovered a cross-platform-reproducible "
            "abundance program and generated four high-consistency, author-unreported Level-2/connectivity-family hypotheses "
            "in nucleotide/NAD-related and antioxidant metabolite pools."
        ),
        "forbidden_claims": [
            "authentic-standard-confirmed identity",
            "ATP energy charge",
            "metabolic flux or enzyme activity",
            "causal tumor dependency or adaptation",
            "clinical biomarker performance",
            "exact stereoisomer for connectivity-family assignments",
        ],
        "provenance": {
            name: {"path": path.as_posix(), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "claim_limit": (
            "Static paired-tissue abundance with library-spectrum/exact-formula support. External pathway and proteogenomic "
            "information may be used only as independent biological context until metabolite-level replication or wet validation."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "readiness_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
