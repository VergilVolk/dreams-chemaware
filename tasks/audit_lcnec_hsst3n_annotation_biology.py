"""Reconcile robust LCNEC dark-feature annotations with the published atlas.

The alias map below is explicit, small and auditable.  It is used only to ask
whether a spectral-library hypothesis was already present in Table S2; it does
not alter candidate ranking or use phenotype information to assign identity.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


AUTHOR_NAME_BY_IK14 = {
    "LFTYTUAZOPRMMI": "UDP-GlcNAc",
    "UDMBCSSLTHHNCD": "Adenosine 5'-monophosphate",
    "RQFCJASXJCIDSX": "Guanosine 5'-monophosphate",
    "JFLIEFSWGNOPJJ": "Phenylacetylglutamine",
    "RWSXRVCMGQZWBV": "Glutathione (reduced)",
    "NYHBQMYGNKIUIF": "Guanosine",
    "JCMUOFQHZLPHQP": "Ophthalmic acid",
    "GHOKWGTUZJEAQD": "Pantothenic acid",
    "QIVBCDIJIAJPQS": "Tryptophan",
    "CVSVTCORWBXHQV": "Creatine",
    "YPZRWBKMTBYPTK": "Glutathione (oxidized)",
    "UYTPUPDQBNUYGX": "Guanine",
}

MODULE_BY_IK14 = {
    "LFTYTUAZOPRMMI": "phosphorylated_nucleotide_or_sugar",
    "UDMBCSSLTHHNCD": "phosphorylated_nucleotide_or_sugar",
    "XTWYTFMLZFPYCI": "phosphorylated_nucleotide_or_sugar",
    "RQFCJASXJCIDSX": "phosphorylated_nucleotide_or_sugar",
    "LNQVTSROQXJCDD": "phosphorylated_nucleotide_or_sugar",
    "SRNWOUGRCWSEMX": "nad_adenylate_turnover",
    "GJAWHXHKYYXBSV": "nad_adenylate_turnover",
    "NYHBQMYGNKIUIF": "free_nucleoside_or_base",
    "UYTPUPDQBNUYGX": "free_nucleoside_or_base",
    "RWSXRVCMGQZWBV": "redox_buffering",
    "YPZRWBKMTBYPTK": "redox_buffering",
    "CIWBSHSKHKDKBQ": "redox_buffering",
    "JCMUOFQHZLPHQP": "redox_buffering",
    "JFLIEFSWGNOPJJ": "amino_acid_or_host_microbiome",
    "GHOKWGTUZJEAQD": "amino_acid_or_host_microbiome",
    "VOXXWSYKYCBWHO": "amino_acid_or_host_microbiome",
    "QIVBCDIJIAJPQS": "amino_acid_or_host_microbiome",
    "CVSVTCORWBXHQV": "creatine_energy_buffer",
    "OUIKMDRGUNIXSP": "exposure_or_xenobiotic",
    "CZWCKYRVOZZJNM": "steroid_sulfate",
    "KXKVLQRXCPHEJC": "volatile_or_low_specificity",
}

PRIORITY_NOVEL_IK14 = {
    "XTWYTFMLZFPYCI",  # ADP connectivity family
    "SRNWOUGRCWSEMX",  # ADP-ribose connectivity family
    "GJAWHXHKYYXBSV",  # quinolinate
    "CIWBSHSKHKDKBQ",  # ascorbate
}


def normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("data/validation/lcnec_hsst3n_all_robust_annotation/priority_annotation_primary20.csv"),
    )
    parser.add_argument(
        "--supplement", type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/article_mmc7.xlsx"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/lcnec_hsst3n_annotation_biology"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(args.annotations)
    annotations = annotations[annotations["annotation_confidence"].str.contains("consistency", na=False)].copy()
    if annotations.empty:
        raise RuntimeError("no consistency-filtered annotation hypotheses")
    author = pd.read_excel(args.supplement, sheet_name="Table S2", header=3)
    author = author[author["Metabolite"].notna()].copy()
    author["normalized_name"] = author["Metabolite"].map(normalized_name)

    rows = []
    for record in annotations.itertuples(index=False):
        ik14 = str(record.p2b_top_ik14)
        author_name = AUTHOR_NAME_BY_IK14.get(ik14)
        matched = author.iloc[0:0]
        if author_name:
            matched = author[author["normalized_name"].eq(normalized_name(author_name))]
        author_row = matched.iloc[0] if len(matched) == 1 else None
        author_beta = float(author_row["beta"]) if author_row is not None else np.nan
        author_q = float(author_row["p FDR"]) if author_row is not None else np.nan
        rows.append({
            "family_id": int(record.family_id),
            "target_mz": float(record.target_mz),
            "target_rt_sec": float(record.target_rt_sec),
            "dark_effect_log2fc": float(record.effect_log2fc),
            "dark_effect_q": float(record.effect_q),
            "ik14": ik14,
            "spectral_hypothesis": str(record.p2b_top_name),
            "annotation_confidence": str(record.annotation_confidence),
            "dreams_score": float(record.dreams_top_score),
            "dreams_margin": float(record.dreams_margin),
            "mass_error_ppm": float(record.selected_mass_error_ppm),
            "reference_spectra": int(record.selected_reference_spectra),
            "module": MODULE_BY_IK14.get(ik14, "unclassified"),
            "author_status": "published_atlas_overlap" if author_row is not None else "author_unreported_spectral_hypothesis",
            "author_metabolite": str(author_row["Metabolite"]) if author_row is not None else "",
            "author_platform": str(author_row["Platform"]) if author_row is not None else "",
            "author_msi_level": float(author_row["MSI Level"]) if author_row is not None else np.nan,
            "author_beta_log10fc": author_beta,
            "author_fdr": author_q,
            "cross_platform_direction_concordant": bool(
                author_row is not None and float(record.effect_log2fc) * author_beta > 0
            ),
            "priority_novel_hypothesis": ik14 in PRIORITY_NOVEL_IK14,
        })
    ledger = pd.DataFrame(rows)

    # Avoid treating RT-separated features that map to the same connectivity as
    # independent identity confirmations.  Keep the highest DreaMS score per IK14.
    identity_ledger = (
        ledger.sort_values(["ik14", "dreams_score", "dark_effect_q"], ascending=[True, False, True])
        .drop_duplicates("ik14", keep="first")
        .reset_index(drop=True)
    )
    overlap = identity_ledger[identity_ledger["author_status"].eq("published_atlas_overlap")]
    rho = spearmanr(overlap["dark_effect_log2fc"], overlap["author_beta_log10fc"]).statistic if len(overlap) >= 3 else np.nan
    sign_concordance = float(overlap["cross_platform_direction_concordant"].mean()) if len(overlap) else np.nan

    phosphorylated = identity_ledger[
        identity_ledger["module"].isin(["phosphorylated_nucleotide_or_sugar", "nad_adenylate_turnover"])
    ]
    free_pool = identity_ledger[identity_ledger["module"].eq("free_nucleoside_or_base")]
    redox = identity_ledger[identity_ledger["module"].eq("redox_buffering")]
    report = {
        "status": "lcnec_hsst3n_annotation_biology_audit_complete",
        "formal": True,
        "consistency_filtered_features": len(ledger),
        "unique_connectivity_hypotheses": len(identity_ledger),
        "published_atlas_overlaps": len(overlap),
        "author_unreported_hypotheses": int(identity_ledger["author_status"].eq("author_unreported_spectral_hypothesis").sum()),
        "priority_author_unreported_hypotheses": identity_ledger.loc[
            identity_ledger["priority_novel_hypothesis"],
            ["spectral_hypothesis", "ik14", "dark_effect_log2fc", "dark_effect_q", "annotation_confidence"],
        ].to_dict("records"),
        "cross_platform_reproduction": {
            "unique_overlaps": len(overlap),
            "direction_concordance": sign_concordance,
            "spearman_effect_rho": None if not np.isfinite(rho) else float(rho),
            "same_direction_fdr05": int(
                (overlap["cross_platform_direction_concordant"] & overlap["author_fdr"].lt(0.05)).sum()
            ),
        },
        "candidate_biology": {
            "phosphorylated_or_nad_related_count": len(phosphorylated),
            "phosphorylated_or_nad_related_positive": int(phosphorylated["dark_effect_log2fc"].gt(0).sum()),
            "free_nucleoside_or_base_count": len(free_pool),
            "free_nucleoside_or_base_negative": int(free_pool["dark_effect_log2fc"].lt(0).sum()),
            "redox_count": len(redox),
            "redox_positive": int(redox["dark_effect_log2fc"].gt(0).sum()),
            "redox_negative": int(redox["dark_effect_log2fc"].lt(0).sum()),
            "interpretation": (
                "The abundance pattern supports a phosphorylated-nucleotide/nucleotide-sugar pool shift and expanded antioxidant "
                "pools. It does not establish ATP energy charge, pathway flux, PARP/CD38 activity, or causal redox adaptation."
            ),
        },
        "gates": {
            "at_least_8_cross_platform_overlaps": len(overlap) >= 8,
            "cross_platform_direction_concordance_ge_0_80": bool(sign_concordance >= 0.8),
            "at_least_3_priority_unreported_hypotheses": int(identity_ledger["priority_novel_hypothesis"].sum()) >= 3,
            "nucleotide_nad_pattern_has_4_positive_hypotheses": int(phosphorylated["dark_effect_log2fc"].gt(0).sum()) >= 4,
            "redox_pattern_has_both_positive_and_negative_arms": bool(
                redox["dark_effect_log2fc"].gt(0).any() and redox["dark_effect_log2fc"].lt(0).any()
            ),
        },
        "decision": (
            "LCNEC passes to formula/fragment-level confirmation. It has stronger sample size and cross-platform reproducibility "
            "than MTBLS13729, but it does not yet replace the frozen Neu5Ac result until the unreported ADP/ADP-ribose/quinolinate/"
            "ascorbate hypotheses survive orthogonal structure checks."
        ),
        "claim_limit": (
            "Static paired-tissue abundance and level-2/connectivity-family hypotheses only. No flux, enzyme activity, exact "
            "stereoisomer, causal tumor dependency, or clinical biomarker claim."
        ),
    }
    report["pass_to_structure_confirmation"] = all(report["gates"].values())

    ledger.to_csv(args.output_dir / "feature_evidence_ledger.csv", index=False)
    identity_ledger.to_csv(args.output_dir / "identity_evidence_ledger.csv", index=False)
    (args.output_dir / "biology_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
