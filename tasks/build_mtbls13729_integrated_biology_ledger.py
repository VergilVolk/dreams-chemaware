"""Build the frozen, evidence-calibrated MTBLS13729 biology ledger.

This is deliberately a synthesis step.  It never upgrades an identity merely
because several weak evidence sources agree.  Published standard assignments,
same-cohort orthogonal chromatography, DreaMS consensus, peak-resolved MS2 and
paired abundance are retained as separate columns so that the manuscript can
state exactly what each candidate supports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/integrated_biology_ledger_v1"


FEATURES = [
    # feature_id, label, module, source-table identity, source MSI, final identity, ceiling
    (347, "myristoylcarnitine", "long_chain_acylcarnitine_accumulation", "Myristoylcarnitine", "Level 1", "myristoylcarnitine remapped to a published Level-1 source feature", "same-cohort remapping; no new standard injection"),
    (150, "palmitoylcarnitine", "long_chain_acylcarnitine_accumulation", None, None, "palmitoylcarnitine, strong DreaMS library consensus", "putative library identity; no source-table standard assignment"),
    (3222, "c20_4_acylcarnitine_like", "long_chain_acylcarnitine_accumulation", None, None, "long-chain/C20:4-acylcarnitine-like", "lipid class and nominal chain composition only; double-bond position unresolved"),
    (398, "free_carnitine", "single_node_context", "Carnitine", "Level 1", "carnitine remapped to a published Level-1 source feature", "same-cohort cross-panel orthogonality, not independent replication"),
    (457, "n1_acetylspermine", "acetylated_polyamine_mta_turnover", "N1-Acetylspermine", "Level 2", "N1-acetylspermine remapped to a published Level-2 source feature", "putative source identity; authentic standard still needed for new Level 1 confirmation"),
    (1717, "n1_n8_diacetylspermidine_like", "acetylated_polyamine_mta_turnover", None, None, "N1,N8-diacetylspermidine-like / acetylated-polyamine family", "cross-chromatography family evidence; exact positional identity unresolved"),
    (494, "methylthioadenosine", "acetylated_polyamine_mta_turnover", "Methylthioadenosine", "Level 2", "methylthioadenosine remapped to a published Level-2 source feature", "putative source identity; no new standard injection"),
    (73, "hypoxanthine", "purine_modified_nucleoside_pool", "Hypoxanthine", "Level 1", "hypoxanthine remapped to a published Level-1 source feature", "same-cohort cross-polarity orthogonality, not independent replication"),
    (1597, "methylguanosine_family", "purine_modified_nucleoside_pool", None, None, "methylguanosine positional-isomer family", "m7G/m2G/Gm unresolved"),
    (3019, "dimethylguanosine_family", "purine_modified_nucleoside_pool", None, None, "dimethylguanosine positional-isomer family", "1,7-/N2,N2- and other positional isomers unresolved"),
    (83, "isoleucine", "large_neutral_amino_acid_pool", "Isoleucine", "Level 1", "isoleucine remapped to a published Level-1 source feature", "same-cohort remapping; no new standard injection"),
    (722, "phenylalanine", "large_neutral_amino_acid_pool", "Phenylalanine", "Level 1", "phenylalanine remapped to a published Level-1 source feature", "published chromatographic identity supersedes weak DreaMS synephrine vote"),
    (732, "tryptophan", "large_neutral_amino_acid_pool", "Tryptophan", "Level 1", "tryptophan remapped to a published Level-1 source feature", "same-cohort cross-polarity orthogonality; no kynurenine claim"),
    (9900175, "sphingosine", "single_node_context", "Sphingosine", "Level 1", "sphingosine from the published Level-1 HILIC feature", "selective single-node accumulation; no broad sphingolipid or flux claim"),
    (428, "taurine", "downgraded_control", "Taurine", "Level 1", "taurine-compatible MS2 but biologically discordant cross-panel signal", "strong spectral identity does not rescue failed cross-panel abundance concordance"),
]


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature = pd.read_csv(ROOT / "data/mtbls13729/convergent_metabolic_modules_v1/feature_summary.csv")
    dreams = pd.read_csv(ROOT / "data/mtbls13729/expanded_candidate_dreams_consensus_v1/expanded_candidate_dreams_consensus.csv")
    cross = pd.read_csv(ROOT / "data/mtbls13729/named_candidate_crosspanel_audit_v1/candidate_crosspanel_summary.csv")
    old = pd.read_csv(ROOT / "data/mtbls13729/candidate_evidence_ledger_v1/candidate_evidence_ledger.csv")

    source_path = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
    source = pd.read_excel(source_path, sheet_name="metabolites", header=1)
    source_name_col = "metabolites"
    source_msi_col = "MSI(Metabolomics Standards Initiative)"

    feature_by_id = feature.set_index("feature_id", drop=False)
    dreams_by_id = dreams.drop_duplicates("feature_id").set_index("feature_id", drop=False)
    cross_by_id = cross.drop_duplicates("feature_id").set_index("feature_id", drop=False)
    old_by_id = old.drop_duplicates("feature_id").set_index("feature_id", drop=False)

    rows: list[dict] = []
    for feature_id, label, module, source_name, expected_msi, identity, ceiling in FEATURES:
        row = {
            "feature_id": feature_id,
            "label": label,
            "module": module,
            "defensible_identity": identity,
            "claim_ceiling": ceiling,
            "source_name": source_name,
            "published_source_msi": expected_msi,
        }

        if feature_id in feature_by_id.index:
            f = feature_by_id.loc[feature_id]
            row.update(
                pairs=int(f["pairs"]),
                mean_log2fc=float(f["mean_log2fc"]),
                positive_pairs=int(f["positive_pairs"]),
                abundance_bootstrap_ci_low=float(f["bootstrap_ci_low"]),
                abundance_bootstrap_ci_high=float(f["bootstrap_ci_high"]),
                one_sided_sign_p=float(f["one_sided_sign_p"]),
            )
        elif feature_id in dreams_by_id.index:
            d = dreams_by_id.loc[feature_id]
            row.update(
                pairs=np.nan,
                mean_log2fc=float(d["eic_raw_rmu_log2fc"]),
                positive_pairs=np.nan,
                abundance_bootstrap_ci_low=np.nan,
                abundance_bootstrap_ci_high=np.nan,
                one_sided_sign_p=float(d["eic_max_exact_p"]),
            )

        if feature_id in dreams_by_id.index:
            d = dreams_by_id.loc[feature_id]
            row.update(
                discovery_panel=d["panel"],
                mz=float(d["mz"]),
                rt_sec=float(d["rt_sec"]),
                peak_resolved_ms2_spectra=int(d["n_ms2_spectra"]),
                dreams_name=d.get("ref_name"),
                dreams_tier=d.get("dreams_consensus_evidence_tier"),
                dreams_supporting_spectra=int(d["supporting_spectra"]) if finite(d.get("supporting_spectra")) else 0,
                dreams_median_similarity=float(d["median_dreams_similarity"]) if finite(d.get("median_dreams_similarity")) else np.nan,
                dreams_agreement=float(d["agreement_fraction"]) if finite(d.get("agreement_fraction")) else np.nan,
            )
        elif feature_id in old_by_id.index:
            d = old_by_id.loc[feature_id]
            row.update(
                discovery_panel="pos_rp",
                mz=float(d["mz"]),
                rt_sec=float(d["rt_sec"]),
                peak_resolved_ms2_spectra=int(d["peak_resolved_ms2_spectra"]),
                dreams_name=np.nan,
                dreams_tier=np.nan,
                dreams_supporting_spectra=0,
                dreams_median_similarity=np.nan,
                dreams_agreement=np.nan,
            )
        elif feature_id == 9900175:
            row.update(
                discovery_panel="pos_hilic",
                mz=300.28586,
                rt_sec=0.884 * 60.0,
                peak_resolved_ms2_spectra=np.nan,
                dreams_name=np.nan,
                dreams_tier=np.nan,
                dreams_supporting_spectra=0,
                dreams_median_similarity=np.nan,
                dreams_agreement=np.nan,
            )

        if source_name:
            hit = source[source[source_name_col].astype(str).str.casefold().eq(source_name.casefold())]
            if len(hit) != 1:
                raise RuntimeError(f"expected one source-table row for {source_name}, got {len(hit)}")
            s = hit.iloc[0]
            observed_msi = str(s[source_msi_col])
            if expected_msi and observed_msi != expected_msi:
                raise RuntimeError(f"MSI mismatch for {source_name}: {observed_msi} != {expected_msi}")
            row.update(
                source_mz=float(s["m/z"]),
                source_rt_sec=float(s["RT [min]"]) * 60.0,
                source_hmdb=s["HMDB"],
                source_inchikey=s["InChIKey"],
                source_adduct=s["Adducts"],
                source_type=s["Type"],
                source_rmu_vs_normal_p=float(s["p-value.4"]) if finite(s.get("p-value.4")) else np.nan,
                source_rmu_vs_normal_log2fc=float(s["FC [log2].4"]) if finite(s.get("FC [log2].4")) else np.nan,
                source_all_cancer_vs_normal_p=float(s["p.value(cancer\u00a0vs normal)"]) if finite(s.get("p.value(cancer\u00a0vs normal)")) else np.nan,
                source_all_cancer_vs_normal_fdr=float(s["FDR(cancer\u00a0vs normal)"]) if finite(s.get("FDR(cancer\u00a0vs normal)")) else np.nan,
            )
            discovery_panel = str(row.get("discovery_panel", ""))
            source_adduct = str(s["Adducts"])
            polarity_compatible = (
                (discovery_panel.startswith("pos_") and "+" in source_adduct)
                or (discovery_panel.startswith("neg_") and "-" in source_adduct)
            )
            same_rplc_mode = discovery_panel in {"pos_rp", "neg_rp"} and str(s["Type"]) == "RPLC" and polarity_compatible
            if same_rplc_mode:
                row["source_same_chromatography_and_mode"] = True
                row["source_mass_error_ppm"] = (row["mz"] - row["source_mz"]) / row["source_mz"] * 1e6
                row["source_rt_error_sec"] = row["rt_sec"] - row["source_rt_sec"]
            else:
                row["source_same_chromatography_and_mode"] = False
                row["source_mass_error_ppm"] = np.nan
                row["source_rt_error_sec"] = np.nan
        else:
            row.update(
                source_mz=np.nan,
                source_rt_sec=np.nan,
                source_hmdb=np.nan,
                source_inchikey=np.nan,
                source_adduct=np.nan,
                source_type=np.nan,
                source_rmu_vs_normal_p=np.nan,
                source_rmu_vs_normal_log2fc=np.nan,
                source_all_cancer_vs_normal_p=np.nan,
                source_all_cancer_vs_normal_fdr=np.nan,
                source_same_chromatography_and_mode=False,
                source_mass_error_ppm=np.nan,
                source_rt_error_sec=np.nan,
            )

        if feature_id in cross_by_id.index:
            c = cross_by_id.loc[feature_id]
            row.update(
                crosspanel_common_samples=int(c["common_samples"]),
                crosspanel_within_tissue_spearman=float(c["within_tissue_spearman"]),
                crosspanel_tissue_permutation_p=float(c["tissue_stratified_permutation_p"]),
                crosspanel_paired_spearman=float(c["paired_tumor_normal_spearman"]),
                crosspanel_paired_permutation_p=float(c["paired_permutation_p"]),
                crosspanel_source_rank=int(c["source_target_correlation_rank"]),
            )
        elif feature_id == 1717:
            row.update(
                crosspanel_common_samples=59,
                crosspanel_within_tissue_spearman=0.755855579501777,
                crosspanel_tissue_permutation_p=4.999750012499375e-05,
                crosspanel_paired_spearman=0.7192118226600983,
                crosspanel_paired_permutation_p=4.999750012499375e-05,
                crosspanel_source_rank=1,
            )
        else:
            row.update(
                crosspanel_common_samples=np.nan,
                crosspanel_within_tissue_spearman=np.nan,
                crosspanel_tissue_permutation_p=np.nan,
                crosspanel_paired_spearman=np.nan,
                crosspanel_paired_permutation_p=np.nan,
                crosspanel_source_rank=np.nan,
            )

        # Manuscript evidence tier: identity, technical orthogonality and abundance
        # are combined only for prioritization, never to manufacture an MSI level.
        if feature_id in {347, 398, 457, 494, 73, 83, 722, 732, 9900175}:
            tier = "A_source_identity_remap"
        elif feature_id in {150, 1717, 1597, 3019, 3222}:
            tier = "B_strong_family_candidate"
        else:
            tier = "C_downgraded_or_control"
        row["manuscript_evidence_tier"] = tier
        rows.append(row)

    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "integrated_candidate_ledger.csv", index=False)

    modules = pd.read_csv(ROOT / "data/mtbls13729/convergent_metabolic_modules_v1/module_summary.csv")
    modules.to_csv(OUT / "integrated_module_summary.csv", index=False)

    source_effect = ledger.dropna(subset=["source_rmu_vs_normal_log2fc", "mean_log2fc"])
    source_effect_concordance = {
        "candidates": int(len(source_effect)),
        "all_same_direction": bool(
            np.all(np.sign(source_effect["source_rmu_vs_normal_log2fc"]) == np.sign(source_effect["mean_log2fc"]))
        ),
        "spearman_rho": float(spearmanr(source_effect["mean_log2fc"], source_effect["source_rmu_vs_normal_log2fc"]).statistic),
        "spearman_p": float(spearmanr(source_effect["mean_log2fc"], source_effect["source_rmu_vs_normal_log2fc"]).pvalue),
        "pearson_r": float(pearsonr(source_effect["mean_log2fc"], source_effect["source_rmu_vs_normal_log2fc"]).statistic),
        "pearson_p": float(pearsonr(source_effect["mean_log2fc"], source_effect["source_rmu_vs_normal_log2fc"]).pvalue),
        "interpretation": "same-cohort re-extraction agreement, not independent biological replication",
    }
    report = {
        "status": "mtbls13729_integrated_biology_ledger_complete",
        "formal": False,
        "candidates": int(len(ledger)),
        "tier_counts": ledger["manuscript_evidence_tier"].value_counts().to_dict(),
        "modules": modules.to_dict(orient="records"),
        "source_table_effect_concordance": source_effect_concordance,
        "key_corrections": {
            "feature_722": "Published Level-1 phenylalanine m/z/RT evidence supersedes a weak DreaMS synephrine vote.",
            "feature_428": "Taurine-compatible MS2 is retained, but cross-panel abundance concordance failed and the feature is downgraded.",
            "feature_3222": "Long-chain acylcarnitine class is supported; exact C20:4 isomer and flux direction remain unresolved.",
        },
        "claim_limit": (
            "The ledger separates source-table identity, DreaMS consensus, raw MS2, same-cohort cross-panel "
            "orthogonality and post-selection paired abundance. It does not establish independent cohort replication, "
            "new MSI Level-1 confirmation, subtype specificity, metabolic flux, enzyme activity or causality."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
