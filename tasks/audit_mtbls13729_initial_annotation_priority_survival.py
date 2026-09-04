"""Audit which initial DreaMS annotation priorities survive biology-grade review.

The six rows in ``full_annotated_feature_audit_v1/all_priority.csv`` were an
early annotation-plus-statistics screen.  They are not a truth set.  This
script joins the frozen targeted-EIC re-extraction, source-paper overlap and
final claim scorecard so that an attractive library match cannot silently
become a new-metabolite or phenotype claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
OUT = BASE / "annotation_biology_benchmark_v1"


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().eq("true")


def main() -> None:
    initial = pd.read_csv(BASE / "full_annotated_feature_audit_v1/all_priority.csv")
    eic = pd.read_csv(BASE / "broad_candidate_eic_analysis_v1/candidate_targeted_eic_results.csv")
    overlap = pd.read_csv(BASE / "broad_candidate_novelty_audit_v1/candidate_original_paper_overlap.csv")
    scorecard = pd.read_csv(BASE / "candidate_claim_scorecard_v3/candidate_claim_scorecard_v3.csv")

    if len(initial) != 6 or initial.feature_id.nunique() != 6:
        raise RuntimeError("expected the frozen six-row initial annotation priority screen")
    key = ["panel", "feature_id"]
    joined = initial.merge(eic, on=key, how="left", validate="one_to_one", suffixes=("_screen", "_eic"))
    joined = joined.merge(overlap, on=key, how="left", validate="one_to_one", suffixes=("", "_overlap"))
    claims = scorecard[["feature_id", "claim_class", "claim_ceiling_v3"]].copy()
    joined = joined.merge(claims, on="feature_id", how="left", validate="one_to_one")
    if joined.eic_detection_fraction.isna().any():
        raise RuntimeError("every initial priority must have frozen targeted-EIC re-extraction")

    max_targeted_p = joined[["eic_raw_rmu_ttest_p", "eic_full_pqn_rmu_ttest_p"]].max(axis=1)
    min_targeted_effect = joined[["eic_raw_rmu_mean_log2fc", "eic_full_pqn_rmu_mean_log2fc"]].abs().min(axis=1)
    joined["targeted_eic_max_ttest_p"] = max_targeted_p
    joined["targeted_eic_min_abs_log2fc"] = min_targeted_effect
    joined["targeted_eic_both_p_lt_0_05"] = max_targeted_p.lt(0.05)
    joined["source_identity_already_present"] = as_bool(joined.author_identity_present)
    joined["strict_source_coordinate_match"] = pd.to_numeric(
        joined.strict_mz_rt_match_count, errors="coerce"
    ).fillna(0).gt(0)
    joined["survives_as_positive_biology_node"] = joined.claim_class.fillna("").isin(
        ["CONTEXT_ONLY", "GENERAL_TUMOUR_SUPPORT", "FAMILY_VALIDATION_PRIORITY", "PRIMARY_SUBTYPE_ANCHOR"]
    )
    joined["new_exact_metabolite_claim_permitted"] = False

    decisions = {
        41: (
            "known_identity_nonrobust_abundance",
            "same InChIKey already occurs in the source table; targeted EIC is directionally positive but not robust",
        ),
        486: (
            "source_level1_remap_nonrobust_abundance",
            "strict source Level-1 malic-acid coordinate match; targeted EIC does not pass the frozen abundance gate",
        ),
        79: (
            "identity_conflict_and_nonrobust_abundance",
            "DreaMS 3-phenyllactic-acid vote conflicts with a source-table coordinate match to 2-hydroxycinnamic acid",
        ),
        73: ("retained_context_only", "source Level-1 hypoxanthine retained as purine-pool context"),
        732: ("retained_general_tumour_support", "source Level-1 tryptophan retained; no kynurenine-pathway claim"),
        398: ("retained_general_tumour_support", "source Level-1 carnitine retained; no flux claim"),
    }
    if set(joined.feature_id) != set(decisions):
        raise RuntimeError("initial-priority identities changed; update the frozen decision audit explicitly")
    joined["final_survival_decision"] = joined.feature_id.map(lambda value: decisions[int(value)][0])
    joined["decision_reason"] = joined.feature_id.map(lambda value: decisions[int(value)][1])

    keep = [
        "panel", "feature_id", "best_name", "annotation_evidence_tier",
        "n_support_samples", "median_cosine", "raw_rmu_mean_log2fc", "pqn_rmu_mean_log2fc",
        "screen_fdr10", "targeted_eic_min_abs_log2fc", "targeted_eic_max_ttest_p",
        "targeted_eic_both_p_lt_0_05", "source_identity_already_present",
        "strict_source_coordinate_match", "closest_author_name", "claim_class", "claim_ceiling_v3",
        "survives_as_positive_biology_node", "new_exact_metabolite_claim_permitted",
        "final_survival_decision", "decision_reason",
    ]
    audit = joined[keep].copy()
    audit.to_csv(OUT / "initial_annotation_priority_survival.csv", index=False)

    report = {
        "status": "mtbls13729_initial_annotation_priority_survival_complete",
        "formal": True,
        "initial_priorities": int(len(audit)),
        "retained_positive_biology_nodes": int(audit.survives_as_positive_biology_node.sum()),
        "filtered_or_downgraded": int((~audit.survives_as_positive_biology_node).sum()),
        "targeted_eic_both_p_lt_0_05": int(audit.targeted_eic_both_p_lt_0_05.sum()),
        "source_identities_already_present": int(audit.source_identity_already_present.sum()),
        "new_exact_metabolite_claims": 0,
        "retained_feature_ids": sorted(audit.loc[audit.survives_as_positive_biology_node, "feature_id"].astype(int).tolist()),
        "filtered_feature_ids": sorted(audit.loc[~audit.survives_as_positive_biology_node, "feature_id"].astype(int).tolist()),
        "claim_limit": (
            "This audit measures survival of a selected initial screen, not annotation accuracy. "
            "Source identities, targeted-EIC robustness and identity conflicts are reported separately."
        ),
    }
    (OUT / "initial_annotation_priority_survival_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
