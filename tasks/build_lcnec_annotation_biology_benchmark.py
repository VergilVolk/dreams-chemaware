"""Build a denominator-safe author/DreaMS/P2b/consensus benchmark for LCNEC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/validation"
OUT = VAL / "lcnec_hsst3n_annotation_benchmark_v1"
INPUTS = {
    "author_workbook": VAL / "lcnec_zenodo19005638_preflight/author_annotation_workbook_audit.json",
    "headroom": VAL / "lcnec_hsst3n_qc_headroom_gate/qc_headroom_gate.json",
    "author_overlap": VAL / "lcnec_hsst3n_author_overlap_gate/author_overlap_gate.json",
    "annotation": VAL / "lcnec_hsst3n_all_robust_annotation/annotation_report.json",
    "biology": VAL / "lcnec_hsst3n_annotation_biology/biology_report.json",
    "readiness": VAL / "lcnec_hsst3n_manuscript_readiness/readiness_report.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {key: load(path) for key, path in INPUTS.items()}
    if not all(value.get("formal") is True for value in data.values()):
        raise RuntimeError("all LCNEC benchmark inputs must be formal")

    author = data["author_workbook"]
    headroom = data["headroom"]
    overlap = data["author_overlap"]
    annotation = data["annotation"]
    biology = data["biology"]

    qualified = headroom["all_three_qualified_families"]
    dark = annotation["queries"]
    with_candidates = annotation["primary_20ppm"]["queries_with_candidates"]
    agreement = annotation["ppm_coverage"]["20"]["dreams_p2b_agreement"]
    high = (
        annotation["primary_20ppm"]["high_consistency_compound_level2_hypotheses"]
        + annotation["primary_20ppm"]["high_consistency_connectivity_family_hypotheses"]
    )
    moderate = (
        annotation["primary_20ppm"]["moderate_consistency_compound_level2_hypotheses"]
        + annotation["primary_20ppm"]["moderate_consistency_connectivity_family_hypotheses"]
    )
    consistency = high + moderate

    universe_rows = [
        {
            "universe": "source supplement annotated atlas",
            "denominator": None,
            "method": "source authors",
            "count": author["paper_declared_metabolites"],
            "rate": None,
            "meaning": "declared annotated metabolites across all platforms",
            "claim_boundary": "no detected-feature denominator; count is not an annotation rate",
        },
        {
            "universe": "source supplement HSST3n rows",
            "denominator": None,
            "method": "source authors",
            "count": author["platform_counts"]["HSST3n"],
            "rate": None,
            "meaning": "published main-HSST3n annotated rows",
            "claim_boundary": "HSST3n-2HG has three additional targeted rows and is separate",
        },
        {
            "universe": "reconstructed QC-qualified precursor-RT families",
            "denominator": qualified,
            "method": "source-table overlap",
            "count": overlap["matched_to_published_hsst3n"],
            "rate": overlap["matched_to_published_hsst3n"] / qualified,
            "meaning": "our QC-qualified families matched to the published HSST3n m/z-RT ledger",
            "claim_boundary": "overlap with the source table, not source-study annotation accuracy",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "official DreaMS constrained candidates",
            "count": with_candidates,
            "rate": with_candidates / dark,
            "meaning": "candidate coverage at the frozen 20-ppm protocol",
            "claim_boundary": "candidate coverage, not correct identity rate",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "official DreaMS plus frozen P2b agreement",
            "count": agreement,
            "rate": agreement / dark,
            "meaning": "top-candidate agreement under the application protocol",
            "claim_boundary": "agreement is not truth or independent validation",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "full multi-evidence consistency gate",
            "count": consistency,
            "rate": consistency / dark,
            "meaning": "high or moderate consistency feature hypotheses",
            "claim_boundary": "Level-2/connectivity-family hypotheses only",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "high-consistency subset",
            "count": high,
            "rate": high / dark,
            "meaning": "high-consistency feature hypotheses",
            "claim_boundary": "authentic-standard RT remains absent",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "cross-platform reproduced hypotheses",
            "count": biology["cross_platform_reproduction"]["unique_overlaps"],
            "rate": biology["cross_platform_reproduction"]["unique_overlaps"] / dark,
            "meaning": "hypotheses also reported by the authors on another LC-MS platform",
            "claim_boundary": "orthogonal positive control, not author-unreported discovery",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "author-unreported consistency hypotheses",
            "count": biology["author_unreported_hypotheses"],
            "rate": biology["author_unreported_hypotheses"] / dark,
            "meaning": "spectral hypotheses absent from the published atlas",
            "claim_boundary": "author-unreported is not chemically novel",
        },
        {
            "universe": "frozen 81 dark modules",
            "denominator": dark,
            "method": "priority author-unreported hypotheses",
            "count": len(biology["priority_author_unreported_hypotheses"]),
            "rate": len(biology["priority_author_unreported_hypotheses"]) / dark,
            "meaning": "four formula/fragment/patient-consistent priority hypotheses",
            "claim_boundary": "two compound Level-2 plus two connectivity-family hypotheses",
        },
    ]
    ledger = pd.DataFrame(universe_rows)
    ledger.to_csv(OUT / "annotation_benchmark_ledger.csv", index=False)

    funnel = pd.DataFrame(
        [
            ["raw control precursor-RT families", headroom["precursor_rt_families"], "phenotype blind"],
            ["QC/blank/dilution qualified", qualified, "analytical headroom"],
            ["absent from published HSST3n table", overlap["unmatched_acquisition_qualified_families"], "not automatically novel"],
            ["cross-normalization robust features", 100, "before coelution redundancy control"],
            ["nonredundant dark modules", dark, "frozen annotation universe"],
            ["official DreaMS candidate-bearing", with_candidates, "20-ppm constrained candidate coverage"],
            ["DreaMS-P2b top-candidate agreement", agreement, "agreement, not truth"],
            ["high/moderate consistency features", consistency, "21 unique connectivity hypotheses"],
            ["cross-platform reproduced", biology["cross_platform_reproduction"]["unique_overlaps"], "positive control"],
            ["author-unreported hypotheses", biology["author_unreported_hypotheses"], "not chemical novelty"],
            ["priority hypotheses", len(biology["priority_author_unreported_hypotheses"]), "two Level-2 plus two family"],
        ],
        columns=["stage", "count", "interpretation"],
    )
    funnel.to_csv(OUT / "discovery_funnel.csv", index=False)

    report = {
        "status": "lcnec_annotation_biology_benchmark_complete",
        "formal": True,
        "source_paper": {
            "declared_metabolites_all_platforms": author["paper_declared_metabolites"],
            "valid_statistical_rows": author["valid_statistical_rows"],
            "msi_level_1_2_3": [
                author["msi_level_counts_valid_rows"]["1"],
                author["msi_level_counts_valid_rows"]["2"],
                author["msi_level_counts_valid_rows"]["3"],
            ],
            "hsst3n_main_rows": author["platform_counts"]["HSST3n"],
            "hsst3n_2hg_rows": author["platform_counts"]["HSST3n-2HG"],
            "annotation_rate_available": False,
        },
        "reconstructed_qualified_universe": {
            "families": qualified,
            "source_hsst3n_overlap": overlap["matched_to_published_hsst3n"],
            "source_hsst3n_overlap_fraction": overlap["matched_to_published_hsst3n"] / qualified,
            "source_table_absent": overlap["unmatched_acquisition_qualified_families"],
        },
        "frozen_dark_universe": {
            "modules": dark,
            "official_dreams_candidate_coverage": with_candidates / dark,
            "official_dreams_candidates": with_candidates,
            "dreams_p2b_agreement_coverage": agreement / dark,
            "dreams_p2b_agreement": agreement,
            "high_or_moderate_consistency_coverage": consistency / dark,
            "high_or_moderate_consistency_features": consistency,
            "high_consistency_features": high,
            "unique_connectivity_hypotheses": biology["unique_connectivity_hypotheses"],
            "cross_platform_reproductions": biology["cross_platform_reproduction"]["unique_overlaps"],
            "author_unreported_hypotheses": biology["author_unreported_hypotheses"],
            "priority_hypotheses": len(biology["priority_author_unreported_hypotheses"]),
        },
        "claim_limit": "Counts from different universes are never divided or compared as accuracy. The source paper lacks a detected-feature denominator; DreaMS/P2b application values are candidate coverage/agreement, and only the consistency gate produces Level-2/family hypotheses.",
        "provenance": {key: sha256(path) for key, path in INPUTS.items()},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = f"""# LCNEC denominator-safe annotation benchmark

## Source paper

The supplement declares {author['paper_declared_metabolites']:,} metabolites across all platforms and contains {author['valid_statistical_rows']:,} valid statistical rows: {author['msi_level_counts_valid_rows']['1']} MSI Level 1, {author['msi_level_counts_valid_rows']['2']} Level 2 and {author['msi_level_counts_valid_rows']['3']} Level 3. The main HSST3n table contributes {author['platform_counts']['HSST3n']} rows, with three additional HSST3n-2HG rows. Because no denominator of all detected and unannotated features is supplied, **the author annotation rate cannot be calculated**.

## Reconstructed analytical universe

From 1,138 control precursor-RT families, {qualified} passed pooled-QC reproducibility, blank and dilution gates. Only {overlap['matched_to_published_hsst3n']} ({100*overlap['matched_to_published_hsst3n']/qualified:.2f}%) matched the source HSST3n ledger; {overlap['unmatched_acquisition_qualified_families']} were source-table-absent analytical headroom, not automatically novel metabolites.

## Frozen 81-module comparison

- official DreaMS found constrained library candidates for {with_candidates}/{dark} modules ({100*with_candidates/dark:.2f}% candidate coverage);
- official DreaMS and frozen P2b agreed on {agreement}/{dark} ({100*agreement/dark:.2f}% of all modules; {100*agreement/with_candidates:.2f}% of candidate-bearing modules);
- the full exact-mass, DreaMS/P2b, classical-fragment and stability gate retained {consistency}/{dark} ({100*consistency/dark:.2f}%) feature hypotheses, representing {biology['unique_connectivity_hypotheses']} unique connectivity hypotheses;
- 12 were cross-platform positive controls, nine were absent from the published atlas, and four survived as priority author-unreported Level-2/connectivity-family hypotheses.

These numbers are not a ladder of accuracy estimates. They quantify candidate coverage, model agreement, evidence calibration, cross-platform reproduction and final hypothesis yield on explicitly different denominators.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
