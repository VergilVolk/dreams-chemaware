"""Build an auditable claim-level scorecard for the frozen MTBLS13729 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/mtbls13729/manuscript_evidence_matrix_v2/candidate_manuscript_evidence_matrix_v2.csv"
SUBTYPE = ROOT / "data/mtbls13729/candidate_subtype_interactions_v1/cross_normalization_candidate_decisions.csv"
FULL = ROOT / "data/mtbls13729/full_requantifiable_space_audit_v1/pos_rp__full_feature_audit.csv.gz"
BACKGROUND = ROOT / "data/mtbls13729/module_matched_background_sensitivity_v2/module_background_sensitivity.csv"
OUT = ROOT / "data/mtbls13729/candidate_claim_scorecard_v3"

MODULE_MAP = {
    "acetylated_polyamine_mta_turnover": "acetylated_polyamine_mta",
    "purine_modified_nucleoside_pool": "purine_modified_guanosine",
    "long_chain_acylcarnitine_accumulation": "long_chain_acylcarnitine",
    "expanded_amino_acid_pool": "expanded_amino_acid",
    "large_neutral_amino_acid_pool": "expanded_amino_acid",
    "sialic_acid_pool": "neu5ac",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_bool(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(EVIDENCE)
    evidence = evidence[evidence.feature_id.astype(str).str.fullmatch(r"\d+")].copy()
    evidence["feature_id"] = evidence.feature_id.astype(int)
    subtype = pd.read_csv(SUBTYPE)
    full = pd.read_csv(FULL)
    background = pd.read_csv(BACKGROUND)

    background_summary = (
        background.groupby("module")
        .agg(
            background_specs=("specification", "nunique"),
            background_tail_max=("mean_effect_empirical_upper_tail", "max"),
            background_tail_min=("mean_effect_empirical_upper_tail", "min"),
        )
        .reset_index()
    )

    keep_columns = [
        "feature_id",
        "label",
        "module",
        "defensible_identity",
        "manuscript_evidence_tier",
        "published_source_msi",
        "pairs",
        "mean_log2fc",
        "positive_pairs",
        "peak_resolved_ms2_spectra",
        "dreams_median_similarity",
        "dreams_agreement",
        "delta_type",
        "novelty_interpretation",
        "external_evidence_level",
        "validation_priority",
        "identity_claim_in_manuscript",
        "forbidden_interpretation",
    ]
    scorecard = evidence[keep_columns].merge(subtype, on=["feature_id", "label", "module"], how="left")
    full_keep = full[
        [
            "feature_id",
            "min_pairs",
            "exact_q",
            "ttest_q",
            "nominal_exact_gate",
            "fdr10_exact_gate",
            "fdr05_exact_gate",
            "ion_family_size",
        ]
    ].rename(
        columns={
            "exact_q": "full_13155_exact_q",
            "ttest_q": "full_13155_ttest_q",
            "nominal_exact_gate": "full_13155_nominal_exact_gate",
            "fdr10_exact_gate": "full_13155_fdr10_exact_gate",
            "fdr05_exact_gate": "full_13155_fdr05_exact_gate",
        }
    )
    scorecard = scorecard.merge(full_keep, on="feature_id", how="left")
    scorecard["background_module"] = scorecard.module.map(MODULE_MAP)
    scorecard = scorecard.merge(background_summary, left_on="background_module", right_on="module", how="left", suffixes=("", "_background"))
    if "module_background" in scorecard:
        scorecard = scorecard.drop(columns=["module_background"])

    scorecard["identity_anchor"] = scorecard.manuscript_evidence_tier.str.startswith("A_")
    scorecard["family_candidate"] = scorecard.manuscript_evidence_tier.eq("B_strong_family_candidate")
    scorecard["adequate_rmu_coverage"] = scorecard.rmu_n_min.fillna(0).ge(8)
    scorecard["candidate_panel_primary_fdr10"] = scorecard.rmu_primary_bh_q_max.lt(0.10)
    scorecard["candidate_panel_subtype_fdr10"] = scorecard.rmu_vs_rtu_bh_q_max.lt(0.10)
    scorecard["background_robust_three_specs"] = scorecard.background_specs.eq(3) & scorecard.background_tail_max.lt(0.05)
    scorecard["full_untargeted_exact_fdr10"] = scorecard.full_13155_fdr10_exact_gate.map(safe_bool)

    def classify(row: pd.Series) -> str:
        if row["module"] == "downgraded_control":
            return "NEGATIVE_CONTROL"
        if row["candidate_panel_subtype_fdr10"] and row["identity_anchor"] and row["adequate_rmu_coverage"]:
            return "PRIMARY_SUBTYPE_ANCHOR"
        if not row["adequate_rmu_coverage"]:
            return "LOW_COVERAGE_IDENTITY_VALIDATION"
        if row["candidate_panel_primary_fdr10"] and row["identity_anchor"]:
            return "GENERAL_TUMOUR_SUPPORT"
        if row["candidate_panel_primary_fdr10"] and row["family_candidate"]:
            return "FAMILY_VALIDATION_PRIORITY"
        return "CONTEXT_ONLY"

    scorecard["claim_class"] = scorecard.apply(classify, axis=1)
    scorecard["claim_ceiling_v3"] = np.select(
        [
            scorecard.claim_class.eq("PRIMARY_SUBTYPE_ANCHOR"),
            scorecard.claim_class.eq("GENERAL_TUMOUR_SUPPORT"),
            scorecard.claim_class.eq("FAMILY_VALIDATION_PRIORITY"),
            scorecard.claim_class.eq("LOW_COVERAGE_IDENTITY_VALIDATION"),
            scorecard.claim_class.eq("NEGATIVE_CONTROL"),
        ],
        [
            "mucinous-relative abundance/remodeling anchor; not flux or glycan linkage",
            "paired Rmu abundance support; not mucinous-specific",
            "paired abundance of an unresolved chemical family; standard required",
            "hypothesis-generating family signal; coverage and identity insufficient",
            "reported discordant control; no positive biological claim",
        ],
        default="supporting context only",
    )
    scorecard = scorecard.sort_values(
        ["claim_class", "rmu_vs_rtu_bh_q_max", "rmu_primary_bh_q_max", "feature_id"],
        na_position="last",
    )
    scorecard.to_csv(OUT / "candidate_claim_scorecard_v3.csv", index=False)

    class_counts = scorecard.claim_class.value_counts().to_dict()
    report = {
        "status": "mtbls13729_candidate_claim_scorecard_v3_complete",
        "formal": False,
        "candidates": int(len(scorecard)),
        "claim_class_counts": {key: int(value) for key, value in class_counts.items()},
        "primary_subtype_anchors": scorecard.loc[
            scorecard.claim_class.eq("PRIMARY_SUBTYPE_ANCHOR"), ["feature_id", "label"]
        ].to_dict(orient="records"),
        "full_untargeted_exact_fdr10_candidates": int(scorecard.full_untargeted_exact_fdr10.sum()),
        "low_coverage_candidates": scorecard.loc[
            ~scorecard.adequate_rmu_coverage, ["feature_id", "label", "rmu_n_min"]
        ].to_dict(orient="records"),
        "interpretation": (
            "Claim classes combine identity, Rmu coverage, candidate-panel primary and subtype statistics, "
            "full-space exact FDR, and outcome-blind matched-background sensitivity. They are evidence ceilings, "
            "not a post-hoc numerical discovery score."
        ),
        "claim_limit": (
            "No row reaches full untargeted-space exact FDR10. Candidate-panel and matched-background results "
            "are same-cohort, post-selection evidence and require independent or standard-based validation."
        ),
        "provenance": {
            "evidence_sha256": sha256(EVIDENCE),
            "subtype_sha256": sha256(SUBTYPE),
            "full_space_sha256": sha256(FULL),
            "background_sha256": sha256(BACKGROUND),
            "scorecard_sha256": sha256(OUT / "candidate_claim_scorecard_v3.csv"),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
