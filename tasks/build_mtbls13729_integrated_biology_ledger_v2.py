#!/usr/bin/env python
"""Versioned expansion of the MTBLS13729 evidence-calibrated biology ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/integrated_biology_ledger_v2"
NEW = {
    345: ("proline", "expanded_amino_acid_pool", "Proline", "Proline parent recovered in positive RP and reconciled to published Level-1 negative-HILIC proline"),
    374: ("glutamic_acid", "expanded_amino_acid_pool", "Glutamic acid", "Glutamic acid recovered in positive RP and reconciled to published Level-1 positive-HILIC glutamate"),
    703: ("n_acetylneuraminic_acid", "sialic_acid_pool", "N-Acetylneuraminic acid", "Free N-acetylneuraminic acid recovered in positive RP and reconciled to published Level-1 negative-HILIC Neu5Ac"),
}


def bootstrap(values: np.ndarray, seed: int, repeats: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def patient_deltas(matrix: pd.DataFrame, feature_id: int) -> pd.DataFrame:
    row = matrix.loc[feature_id]
    output = []
    for number in range(21, 31):
        patient = f"P{number:02d}"
        tumour, normal = f"{patient}-Rmu", f"{patient}-RN"
        output.append({
            "patient": patient,
            "feature_id": feature_id,
            "log2_tumor_normal": float(np.log2(float(row[tumour]) + 1.0) - np.log2(float(row[normal]) + 1.0)),
        })
    return pd.DataFrame(output)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(ROOT / "data/mtbls13729/integrated_biology_ledger_v1/integrated_candidate_ledger.csv")
    eic = pd.read_csv(ROOT / "data/mtbls13729/full_space_eic_v1/pos_rp__eic_auc_matrix.csv.gz").set_index("feature_id")
    coverage = pd.read_csv(ROOT / "data/mtbls13729/expanded_ms2_links_v1/candidate_ms2_coverage.csv").set_index("feature_id")
    classical = pd.read_csv(ROOT / "data/mtbls13729/expanded_candidate_classical_v1/pos_rp__classical_library_consensus.csv")
    cross = pd.read_csv(ROOT / "data/mtbls13729/expanded_crosspanel_audit_v1/expanded_crosspanel_summary.csv").set_index("feature_id")
    source = pd.read_excel(
        ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx",
        sheet_name="metabolites", header=1,
    )
    source = source.set_index("metabolites", drop=False)
    new_rows = []
    delta_tables = []
    for fid, (label, module, source_name, identity) in NEW.items():
        deltas = patient_deltas(eic, fid)
        delta_tables.append(deltas.assign(label=label, module=module))
        values = deltas.log2_tumor_normal.to_numpy(float)
        ci_low, ci_high = bootstrap(values, 20260830 + fid)
        c = coverage.loc[fid]
        s = source.loc[source_name]
        x = cross.loc[fid]
        library = classical[
            classical.feature_id.eq(fid)
            & classical.library_name.astype(str).str.contains(source_name, case=False, regex=False, na=False)
        ].sort_values(["n_strong_support_samples", "median_cosine"], ascending=False)
        if library.empty:
            # Aceneuramic acid is a common synonym for Neu5Ac.
            library = classical[
                classical.feature_id.eq(fid)
                & classical.library_name.astype(str).str.contains("Aceneuramic", case=False, na=False)
            ].sort_values(["n_strong_support_samples", "median_cosine"], ascending=False)
        if library.empty:
            raise RuntimeError(f"no identity-matched classical library row for feature {fid}")
        lib = library.iloc[0]
        positive = int(np.sum(values > 0))
        row = {column: np.nan for column in old.columns}
        row.update({
            "feature_id": fid,
            "label": label,
            "module": module,
            "defensible_identity": identity,
            "claim_ceiling": "same-cohort orthogonal source remapping; no new positive-RP authentic-standard injection and no flux claim",
            "source_name": source_name,
            "published_source_msi": str(s["MSI(Metabolomics Standards Initiative)"]),
            "pairs": len(values),
            "mean_log2fc": float(values.mean()),
            "positive_pairs": positive,
            "abundance_bootstrap_ci_low": ci_low,
            "abundance_bootstrap_ci_high": ci_high,
            "one_sided_sign_p": float(binomtest(positive, len(values), 0.5, alternative="greater").pvalue),
            "discovery_panel": "pos_rp",
            "mz": float(c.mz),
            "rt_sec": float(c.rt_sec),
            "peak_resolved_ms2_spectra": int(c.n_ms2_spectra),
            "dreams_name": np.nan,
            "dreams_tier": np.nan,
            "dreams_supporting_spectra": 0,
            "dreams_median_similarity": np.nan,
            "dreams_agreement": np.nan,
            "source_mz": float(s["m/z"]),
            "source_rt_sec": float(s["RT [min]"]) * 60.0,
            "source_hmdb": s["HMDB"],
            "source_inchikey": s["InChIKey"],
            "source_adduct": s["Adducts"],
            "source_type": s["Type"],
            "source_rmu_vs_normal_p": float(s["p-value.4"]),
            "source_rmu_vs_normal_log2fc": float(s["FC [log2].4"]),
            "source_all_cancer_vs_normal_p": float(s["p.value(cancer\u00a0vs normal)"]),
            "source_all_cancer_vs_normal_fdr": float(s["FDR(cancer\u00a0vs normal)"]),
            "source_same_chromatography_and_mode": False,
            "source_mass_error_ppm": np.nan,
            "source_rt_error_sec": np.nan,
            "crosspanel_common_samples": int(x.common_samples),
            "crosspanel_within_tissue_spearman": float(x.within_tissue_spearman),
            "crosspanel_tissue_permutation_p": float(x.tissue_stratified_permutation_p),
            "crosspanel_paired_spearman": float(x.paired_delta_spearman),
            "crosspanel_paired_permutation_p": float(x.paired_delta_permutation_p),
            "crosspanel_source_rank": int(x.source_target_rank),
            "manuscript_evidence_tier": "A_source_identity_orthogonal_recovery",
            "classical_library_name": str(lib.library_name),
            "classical_library_adduct": str(lib.library_adduct),
            "classical_median_cosine": float(lib.median_cosine),
            "classical_strong_support_samples": int(lib.n_strong_support_samples),
        })
        new_rows.append(row)

    for column in ["classical_library_name", "classical_library_adduct", "classical_median_cosine", "classical_strong_support_samples"]:
        if column not in old.columns:
            old[column] = np.nan
    ledger = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    ledger.to_csv(OUT / "integrated_candidate_ledger_v2.csv", index=False)
    deltas_new = pd.concat(delta_tables, ignore_index=True)
    deltas_new.to_csv(OUT / "new_anchor_patient_deltas.csv", index=False)

    # Expanded five-node amino-acid pool: original Ile/Phe/Trp plus Pro/Glu.
    old_delta = pd.read_csv(ROOT / "data/mtbls13729/convergent_metabolic_modules_v1/feature_patient_deltas.csv")
    old_aa = old_delta[old_delta.feature.isin(["isoleucine", "phenylalanine", "tryptophan"])][
        ["patient", "feature", "log2_tumor_normal"]
    ]
    new_aa = deltas_new[deltas_new.feature_id.isin([345, 374])].rename(columns={"label": "feature"})[
        ["patient", "feature", "log2_tumor_normal"]
    ]
    aa = pd.concat([old_aa, new_aa], ignore_index=True)
    pivot = aa.pivot(index="patient", columns="feature", values="log2_tumor_normal")
    module_delta = pivot.mean(axis=1, skipna=True)
    ci_low, ci_high = bootstrap(module_delta.to_numpy(float), 20261515)
    loo = {column: float(pivot.drop(columns=column).mean(axis=1, skipna=True).mean()) for column in pivot.columns}
    expanded_module = {
        "module": "expanded_amino_acid_pool",
        "features": list(pivot.columns),
        "patients": int(len(module_delta)),
        "mean_module_log2fc": float(module_delta.mean()),
        "median_module_log2fc": float(module_delta.median()),
        "positive_patients": int((module_delta > 0).sum()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "one_sided_sign_p": float(binomtest(int((module_delta > 0).sum()), len(module_delta), 0.5, alternative="greater").pvalue),
        "leave_one_feature_out_mean_log2fc": loo,
        "leave_one_feature_out_direction_stable": bool(all(v > 0 for v in loo.values())),
        "postselection": True,
    }
    module_patient = pivot.copy()
    module_patient["module_mean_log2fc"] = module_delta
    module_patient.to_csv(OUT / "expanded_amino_acid_patient_matrix.csv")

    source_effect = ledger.dropna(subset=["source_rmu_vs_normal_log2fc", "mean_log2fc"])
    concordance = {
        "candidates": int(len(source_effect)),
        "all_same_direction": bool(np.all(np.sign(source_effect.mean_log2fc) == np.sign(source_effect.source_rmu_vs_normal_log2fc))),
        "spearman_rho": float(spearmanr(source_effect.mean_log2fc, source_effect.source_rmu_vs_normal_log2fc).statistic),
        "spearman_p": float(spearmanr(source_effect.mean_log2fc, source_effect.source_rmu_vs_normal_log2fc).pvalue),
        "pearson_r": float(pearsonr(source_effect.mean_log2fc, source_effect.source_rmu_vs_normal_log2fc).statistic),
        "pearson_p": float(pearsonr(source_effect.mean_log2fc, source_effect.source_rmu_vs_normal_log2fc).pvalue),
    }
    payload = {
        "status": "mtbls13729_integrated_biology_ledger_v2_complete",
        "formal": False,
        "versioning": "v1 remains frozen; v2 appends three source-anchored orthogonal recoveries",
        "candidates": int(len(ledger)),
        "new_candidates": [345, 374, 703],
        "tier_counts": ledger.manuscript_evidence_tier.value_counts().to_dict(),
        "expanded_amino_acid_module": expanded_module,
        "source_table_effect_concordance": concordance,
        "triage_exclusions": {
            "feature_301": "coeluting with proline but exact-mass and library evidence conflict; not accepted as a sodium adduct",
            "feature_1695": "leucine/isoleucine-like MS2 but source leucine paired-delta concordance failed",
            "feature_725": "5-HIAA versus pyridoxine/isomer ambiguity remains unresolved",
            "feature_458": "multiple isomeric peptide assignments with no source/RT anchor",
        },
        "claim_limit": "The three new v2 nodes are source-identity recoveries in an orthogonal panel, not new metabolites or independent cohort replication. Expanded modules are postselection descriptive and do not establish subtype specificity, flux, enzyme activity, or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
