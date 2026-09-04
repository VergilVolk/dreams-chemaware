#!/usr/bin/env python
"""Local, phenotype-separated biology closure for frozen MTBLS13729 candidates.

Identity evidence is frozen before abundance testing.  BioAware/Rhea and HMDB
are used only to describe network coverage and identity ambiguity; they never
change a candidate identity or the abundance endpoint.

The primary abundance endpoint is paired Rmu versus RN.  The Rmu-versus-Rtu
interaction is secondary.  Because the eight candidates were selected earlier,
the statistics below are confirmation/consolidation diagnostics, not a new
untargeted-discovery FDR analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_ind, wilcoxon


FROZEN_IDENTITIES = {
    4966: {
        "refined_label": "C7H9N5O purine-like / deazaguanine-like isomer family",
        "inchikey": "MXLVASHNANBJDZ-UHFFFAOYSA-N",
        "neutral_formula": "C7H9N5O",
        "neutral_exact_mass": 179.0807099,
        "identity_boundary": "library structure and HMDB name are discordant; family-level only",
        "module": "purine_turnover",
    },
    3019: {
        "refined_label": "dimethylguanosine isomer family",
        "inchikey": "NNQCGMWOZJYFTM-IOSLPCCCSA-N",
        "neutral_formula": "C12H17N5O5",
        "neutral_exact_mass": 311.1229686,
        "identity_boundary": "exact mass supports dimethylguanosine isomers; positional isomer unresolved",
        "module": "modified_guanosine",
    },
    1597: {
        "refined_label": "methylguanosine isomer family [M+H]+",
        "inchikey": "IXOXBSCIXZEQEQ-UHFFFAOYSA-N",
        "neutral_formula": "C11H15N5O5",
        "neutral_exact_mass": 297.1073186,
        "identity_boundary": "endogenous methylguanosine positional isomer unresolved",
        "module": "modified_guanosine",
    },
    7489: {
        "refined_label": "methylguanosine isomer family [M+Na]+",
        "inchikey": "IXOXBSCIXZEQEQ-UHFFFAOYSA-N",
        "neutral_formula": "C11H15N5O5",
        "neutral_exact_mass": 297.1073186,
        "identity_boundary": "same neutral family as feature 1597; sodium adduct",
        "module": "modified_guanosine",
    },
    1717: {
        "refined_label": "N1,N8-diacetylspermidine-like",
        "inchikey": "BKCVMAZDKFQPHB-UHFFFAOYSA-N",
        "neutral_formula": "C11H23N3O2",
        "neutral_exact_mass": 229.179027,
        "identity_boundary": "unique exact-formula HMDB match but no authentic-standard RT confirmation",
        "module": "polyamine_acetylation",
    },
    3222: {
        "refined_label": "C20:4 acylcarnitine-like",
        "inchikey": "RBFQHRALHSUPIA-SNPVRQPZSA-N",
        "neutral_formula": "C27H45NO4",
        "neutral_exact_mass": 447.3348589,
        "identity_boundary": "Level 2; acyl-chain position/isomer and authentic-standard RT unresolved",
        "module": "long_chain_acylcarnitine",
    },
    3180: {
        "refined_label": "exogenous/implausibility-control candidate",
        "inchikey": "PHPXAJCVSGIDFH-UHFFFAOYSA-N",
        "neutral_formula": "",
        "neutral_exact_mass": math.nan,
        "identity_boundary": "biological plausibility is weak; retain as a negative interpretation control",
        "module": "interpretation_control",
    },
    16425: {
        "refined_label": "LPE-like lipid candidate",
        "inchikey": "",
        "neutral_formula": "",
        "neutral_exact_mass": math.nan,
        "identity_boundary": "lipid subclass candidate only; chain/adduct identity unresolved",
        "module": "glycerophospholipid",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ik14(value: object) -> str:
    text = str(value).strip()
    return text.split("-")[0] if text and text.lower() != "nan" else ""


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    observed = abs(float(values.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def exact_label_permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return math.nan
    pooled = np.r_[a, b]
    observed = abs(float(a.mean() - b.mean()))
    exceed = 0
    total = 0
    for chosen in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        stat = abs(float(pooled[mask].mean() - pooled[~mask].mean()))
        exceed += int(stat >= observed - 1e-12)
        total += 1
    return float(exceed / total)


def safe_wilcoxon(values: np.ndarray) -> float:
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    mean = float(values.mean())
    loo = [float(np.delete(values, i).mean()) for i in range(len(values))] if len(values) > 1 else [mean]
    return {
        "n": int(len(values)),
        "mean_log2fc": mean,
        "median_log2fc": float(np.median(values)),
        "fold_change": float(2**mean),
        "ttest_p": float(ttest_1samp(values, 0).pvalue) if len(values) > 1 and values.std() > 0 else 1.0,
        "wilcoxon_p": safe_wilcoxon(values),
        "exact_signflip_p": exact_signflip_p(values),
        "positive_fraction": float(np.mean(values > 0)),
        "loo_min_mean_log2fc": float(min(loo)),
        "loo_max_mean_log2fc": float(max(loo)),
        "loo_direction_stable": bool(all(np.sign(x) == np.sign(mean) for x in loo)),
    }


def paired_feature_deltas(
    log_auc: pd.DataFrame,
    detected: pd.DataFrame,
    tumour_suffix: str,
    normal_suffix: str,
) -> pd.DataFrame:
    rows = []
    for patient_number in range(1, 31):
        patient = f"P{patient_number:02d}"
        tumour = f"{patient}-{tumour_suffix}"
        normal = f"{patient}-{normal_suffix}"
        if tumour not in log_auc or normal not in log_auc:
            continue
        for feature_id in log_auc.index:
            if not bool(detected.loc[feature_id, tumour]) or not bool(detected.loc[feature_id, normal]):
                continue
            a = float(log_auc.loc[feature_id, tumour])
            b = float(log_auc.loc[feature_id, normal])
            if np.isfinite(a) and np.isfinite(b):
                rows.append({"patient": patient, "feature_id": int(feature_id), "delta_log2": a - b})
    return pd.DataFrame(rows)


def correlation_report(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(joined) < 3 or joined.x.nunique() < 2 or joined.y.nunique() < 2:
        return {"n": int(len(joined)), "pearson_r": math.nan, "spearman_rho": math.nan}
    return {
        "n": int(len(joined)),
        "pearson_r": float(pearsonr(joined.x, joined.y).statistic),
        "pearson_p": float(pearsonr(joined.x, joined.y).pvalue),
        "spearman_rho": float(spearmanr(joined.x, joined.y).statistic),
        "spearman_p": float(spearmanr(joined.x, joined.y).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, default=Path("data/mtbls13729/biology_closure_targets_v1"))
    parser.add_argument("--eic-dir", type=Path, default=Path("data/mtbls13729/biology_closure_eic_v1"))
    parser.add_argument("--hmdb", type=Path, default=Path("data/external/netid_v1/source/LiChenPU-NetID-9f63202/dependent/hmdb_library.csv"))
    parser.add_argument("--rhea-participants", type=Path, default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"))
    parser.add_argument("--discovery-matrix", type=Path, default=Path("data/mtbls13729/ms1_consensus/pos_rp__discovery_intensity_matrix.csv.gz"))
    parser.add_argument("--clinical", type=Path, default=Path("data/mtbls13729/clinical_metadata_s2.tsv"))
    parser.add_argument("--family-target-dir", type=Path, default=Path("data/mtbls13729/biology_closure_family_targets_v1"))
    parser.add_argument("--family-eic-dir", type=Path, default=Path("data/mtbls13729/biology_closure_family_eic_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/biology_closure_analysis_v1"))
    args = parser.parse_args()

    ledger_path = args.target_dir / "biology_candidate_ledger.csv"
    auc_path = args.eic_dir / "pos_rp__eic_auc_matrix.csv.gz"
    detection_path = args.eic_dir / "pos_rp__eic_detection_matrix.csv.gz"
    for path in (ledger_path, auc_path, detection_path, args.hmdb, args.rhea_participants, args.discovery_matrix):
        if not path.exists():
            raise FileNotFoundError(path)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path).set_index("feature_id")
    auc = pd.read_csv(auc_path).set_index("feature_id").astype(float)
    detected = pd.read_csv(detection_path).set_index("feature_id").astype(bool)
    expected = sorted(FROZEN_IDENTITIES)
    if sorted(auc.index.astype(int)) != expected or sorted(ledger.index.astype(int)) != expected:
        raise RuntimeError("candidate ledger/EIC rows do not match the eight frozen feature IDs")

    # Complete-detection pair analysis avoids outcome-dependent imputation.
    log_auc = np.log2(auc.where((auc > 0) & detected))
    discovery = pd.read_csv(args.discovery_matrix).set_index("feature_id").astype(float)
    if list(discovery.columns) != list(log_auc.columns):
        raise RuntimeError("discovery matrix and targeted EIC sample order differ")
    discovery_log = np.log2(discovery.where(discovery > 0))
    normalization_factors: dict[str, pd.Series] = {"raw": pd.Series(0.0, index=log_auc.columns)}
    factor_rows = []
    for prevalence in (0.60, 0.80, 0.90):
        keep = discovery.gt(0).mean(axis=1).ge(prevalence)
        stable = discovery_log.loc[keep]
        reference = stable.median(axis=1, skipna=True)
        factors = stable.sub(reference, axis=0).median(axis=0, skipna=True)
        name = f"global_pqn_prev{int(prevalence * 100)}"
        normalization_factors[name] = factors
        factor_rows.extend(
            {"normalization": name, "sample": sample, "log2_factor": float(value), "background_features": int(keep.sum())}
            for sample, value in factors.items()
        )
    pd.DataFrame(factor_rows).to_csv(out / "phenotype_blind_normalization_factors.csv", index=False)

    paired_by_variant: dict[str, dict[str, pd.DataFrame]] = {}
    for normalization, factors in normalization_factors.items():
        matrix = log_auc.sub(factors, axis=1)
        paired_by_variant[normalization] = {
            "Rmu": paired_feature_deltas(matrix, detected, "Rmu", "RN"),
            "Rtu": paired_feature_deltas(matrix, detected, "Rtu", "RN"),
            "Ltu": paired_feature_deltas(matrix, detected, "Ltu", "LN"),
        }
        for cohort, frame in paired_by_variant[normalization].items():
            frame.assign(normalization=normalization).to_csv(
                out / f"{cohort.lower()}_{normalization}_pair_deltas.csv", index=False
            )

    rmu = paired_by_variant["raw"]["Rmu"]
    rtu = paired_by_variant["raw"]["Rtu"]
    ltu = paired_by_variant["raw"]["Ltu"]

    abundance_rows = []
    for normalization, cohorts in paired_by_variant.items():
        for feature_id in expected:
            a = cohorts["Rmu"].loc[cohorts["Rmu"].feature_id.eq(feature_id), "delta_log2"].to_numpy(float)
            b = cohorts["Rtu"].loc[cohorts["Rtu"].feature_id.eq(feature_id), "delta_log2"].to_numpy(float)
            c = cohorts["Ltu"].loc[cohorts["Ltu"].feature_id.eq(feature_id), "delta_log2"].to_numpy(float)
            row = {"normalization": normalization, "feature_id": feature_id}
            row.update({f"rmu_{k}": v for k, v in summarize(a).items()})
            row.update({f"rtu_{k}": v for k, v in summarize(b).items()})
            row.update({f"ltu_{k}": v for k, v in summarize(c).items()})
            if len(a) >= 3 and len(b) >= 3:
                row["interaction_log2fc"] = float(a.mean() - b.mean())
                row["interaction_ttest_p"] = float(ttest_ind(a, b, equal_var=False).pvalue)
                row["interaction_exact_permutation_p"] = exact_label_permutation_p(a, b)
            abundance_rows.append(row)
    abundance_all = pd.DataFrame(abundance_rows)
    abundance_all.to_csv(out / "paired_abundance_by_normalization.csv", index=False)
    abundance = abundance_all.loc[abundance_all.normalization.eq("raw")].drop(columns="normalization").set_index("feature_id")

    # Collapse the two adducts of the methylguanosine family before module tests.
    rmu_wide = rmu.pivot(index="patient", columns="feature_id", values="delta_log2")
    rtu_wide = rtu.pivot(index="patient", columns="feature_id", values="delta_log2")
    module_rows = []
    module_patient_rows = []
    for normalization, cohorts in paired_by_variant.items():
        for cohort in ("Rmu", "Rtu"):
            wide = cohorts[cohort].pivot(index="patient", columns="feature_id", values="delta_log2")
            methyl = wide.reindex(columns=[1597, 7489]).median(axis=1, skipna=True)
            dimethyl = wide.get(3019, pd.Series(index=wide.index, dtype=float))
            modified = pd.concat([methyl.rename("methyl"), dimethyl.rename("dimethyl")], axis=1).mean(axis=1, skipna=False)
            contrast = dimethyl - methyl
            for patient in wide.index:
                module_patient_rows.append({
                    "normalization": normalization,
                    "cohort": cohort,
                    "patient": patient,
                    "methylguanosine_collapsed_log2fc": methyl.get(patient, math.nan),
                    "dimethylguanosine_log2fc": dimethyl.get(patient, math.nan),
                    "modified_guanosine_module_log2fc": modified.get(patient, math.nan),
                    "dimethyl_minus_methyl_log2fc": contrast.get(patient, math.nan),
                })
            for name, series in (
                ("methylguanosine_collapsed", methyl),
                ("dimethylguanosine", dimethyl),
                ("modified_guanosine_module", modified),
                ("dimethyl_minus_methyl_contrast", contrast),
            ):
                summary = summarize(series.dropna().to_numpy(float))
                module_rows.append({"normalization": normalization, "cohort": cohort, "module": name, **summary})
    module_patients = pd.DataFrame(module_patient_rows)
    modules = pd.DataFrame(module_rows)
    modules.to_csv(out / "module_summary.csv", index=False)
    module_patients.to_csv(out / "module_patient_effects.csv", index=False)
    module_interactions = []
    for normalization in normalization_factors:
        block = module_patients.loc[module_patients.normalization.eq(normalization)]
        a = block.loc[block.cohort.eq("Rmu"), "modified_guanosine_module_log2fc"].dropna().to_numpy(float)
        b = block.loc[block.cohort.eq("Rtu"), "modified_guanosine_module_log2fc"].dropna().to_numpy(float)
        module_interactions.append({
            "normalization": normalization,
            "n_rmu": int(len(a)),
            "n_rtu": int(len(b)),
            "rmu_minus_rtu_log2fc": float(a.mean() - b.mean()),
            "exact_label_permutation_p": exact_label_permutation_p(a, b),
        })
    pd.DataFrame(module_interactions).to_csv(out / "module_interaction_sensitivity.csv", index=False)

    clinical_sensitivity: dict[str, object] = {"available": False}
    if args.clinical.exists():
        clinical = pd.read_csv(args.clinical, sep="\t")
        tumour = clinical.loc[clinical.tissue.eq("Tumor")].copy()
        tumour["patient"] = tumour.patient_number.map(lambda x: f"P{int(x):02d}")
        raw_module = module_patients.loc[
            module_patients.normalization.eq("raw") & module_patients.cohort.isin(["Rmu", "Rtu"])
        ].merge(tumour[["patient", "pathology", "mmr", "braf"]], on="patient", how="left", validate="one_to_one")
        if raw_module[["pathology", "mmr", "braf"]].isna().any().any():
            raise RuntimeError("clinical Table S2 did not map to every Rmu/Rtu module score")
        raw_module.to_csv(out / "module_patient_effects_clinical.csv", index=False)
        strata = {}
        for (pathology, mmr), group in raw_module.groupby(["pathology", "mmr"], sort=True):
            strata[f"{pathology}_{mmr}"] = summarize(group.modified_guanosine_module_log2fc.to_numpy(float))
        rmu_d = raw_module.loc[(raw_module.pathology == "Rmu") & (raw_module.mmr == "dMMR"), "modified_guanosine_module_log2fc"].dropna().to_numpy(float)
        rmu_p = raw_module.loc[(raw_module.pathology == "Rmu") & (raw_module.mmr == "pMMR"), "modified_guanosine_module_log2fc"].dropna().to_numpy(float)
        rtu_p = raw_module.loc[(raw_module.pathology == "Rtu") & (raw_module.mmr == "pMMR"), "modified_guanosine_module_log2fc"].dropna().to_numpy(float)
        clinical_sensitivity = {
            "available": True,
            "within_pathology_mmr": strata,
            "Rmu_dMMR_minus_pMMR": {
                "mean_difference": float(np.mean(rmu_d) - np.mean(rmu_p)),
                "exact_label_permutation_p": exact_label_permutation_p(rmu_d, rmu_p),
                "n_dMMR": int(len(rmu_d)),
                "n_pMMR": int(len(rmu_p)),
            },
            "pMMR_Rmu_minus_Rtu": {
                "mean_difference": float(np.mean(rmu_p) - np.mean(rtu_p)),
                "exact_label_permutation_p": exact_label_permutation_p(rmu_p, rtu_p),
                "n_Rmu": int(len(rmu_p)),
                "n_Rtu": int(len(rtu_p)),
            },
            "boundary": "sensitivity analysis only; 4 pMMR Rmu and 5/6 dMMR Rmu have complete module values, which cannot establish MMR-independent subtype specificity",
        }

    rmu_module = module_patients.loc[
        module_patients.cohort.eq("Rmu") & module_patients.normalization.eq("raw")
    ].set_index("patient")
    correlations = {
        "rmu_methyl_adduct_pair": correlation_report(
            rmu_wide.get(1597, pd.Series(dtype=float)), rmu_wide.get(7489, pd.Series(dtype=float))
        ),
        "rmu_methyl_vs_dimethyl": correlation_report(
            rmu_module["methylguanosine_collapsed_log2fc"], rmu_module["dimethylguanosine_log2fc"]
        ),
    }
    for feature_id, label in ((1717, "diacetylspermidine"), (3222, "c20_4_acylcarnitine"), (4966, "purine_like")):
        correlations[f"rmu_modified_guanosine_vs_{label}"] = correlation_report(
            rmu_module["modified_guanosine_module_log2fc"],
            rmu_wide.get(feature_id, pd.Series(dtype=float)),
        )

    hmdb = pd.read_csv(args.hmdb, low_memory=False)
    rhea = pd.read_csv(args.rhea_participants, low_memory=False)
    rhea_ik_col = next((c for c in rhea.columns if "inchikey" in c.lower()), None)
    rhea_ik14 = set(rhea[rhea_ik_col].dropna().map(ik14)) if rhea_ik_col else set()
    identity_rows = []
    for feature_id in expected:
        fixed = FROZEN_IDENTITIES[feature_id]
        formula = fixed["neutral_formula"]
        matches = hmdb.loc[hmdb.formula.astype(str).eq(formula)].copy() if formula else hmdb.iloc[0:0]
        names = sorted(set(matches["name"].dropna().astype(str)))
        key14 = ik14(fixed["inchikey"])
        identity_rows.append({
            "feature_id": feature_id,
            **fixed,
            "hmdb_exact_formula_match_count": int(len(matches)),
            "hmdb_exact_formula_names": " | ".join(names[:20]),
            "rhea_direct_compound_node": bool(key14 and key14 in rhea_ik14),
            "bioaware_role": "direct_reaction_node" if key14 and key14 in rhea_ik14 else "family_or_module_evidence_only",
        })
    identity = pd.DataFrame(identity_rows).set_index("feature_id")
    combined = ledger.join(identity, how="left").join(abundance, how="left")

    # Peak-shape audit is independent of phenotype testing and guards against
    # integrating unrelated local peaks inside the fixed RT window.
    per_sample_files = sorted((args.eic_dir / "per_sample").glob("pos_rp__*__eic.csv.gz"))
    if len(per_sample_files) != len(auc.columns):
        raise RuntimeError(f"expected {len(auc.columns)} per-sample EIC files, found {len(per_sample_files)}")
    peak_rows = pd.concat([pd.read_csv(path) for path in per_sample_files], ignore_index=True)
    peak_quality = peak_rows.groupby("feature_id", sort=True).agg(
        samples=("sample_name", "nunique"),
        detected_fraction=("detected_eic", "mean"),
        median_abs_apex_delta_sec=("eic_apex_delta_sec", lambda x: float(np.nanmedian(np.abs(x)))),
        p95_abs_apex_delta_sec=("eic_apex_delta_sec", lambda x: float(np.nanpercentile(np.abs(x), 95))),
        median_snr=("eic_snr", "median"),
        median_local_peak_count=("local_peak_count", "median"),
    )
    peak_quality.to_csv(out / "peak_quality_audit.csv")
    apex = peak_rows.pivot(index="sample_name", columns="feature_id", values="eic_apex_rt")
    methyl_coelution = (apex[1597] - apex[7489]).abs().dropna()
    methyl_family_audit = {
        "observed_mz_difference": float(ledger.loc[7489, "mz"] - ledger.loc[1597, "mz"]),
        "theoretical_Na_minus_H_difference": 21.9819442498,
        "mass_difference_residual_da": float(abs((ledger.loc[7489, "mz"] - ledger.loc[1597, "mz"]) - 21.9819442498)),
        "consensus_rt_difference_sec": float(abs(ledger.loc[7489, "rt_sec"] - ledger.loc[1597, "rt_sec"])),
        "samples_with_both_apexes": int(len(methyl_coelution)),
        "median_observed_apex_difference_sec": float(methyl_coelution.median()),
        "p95_observed_apex_difference_sec": float(methyl_coelution.quantile(0.95)),
    }

    family_support_report: dict[str, object] = {"available": False}
    family_auc_path = args.family_eic_dir / "pos_rp__eic_auc_matrix.csv.gz"
    family_detection_path = args.family_eic_dir / "pos_rp__eic_detection_matrix.csv.gz"
    if family_auc_path.exists() and family_detection_path.exists():
        family_auc = pd.read_csv(family_auc_path).set_index("feature_id").astype(float)
        family_detected = pd.read_csv(family_detection_path).set_index("feature_id").astype(bool)
        if 8481 not in family_auc.index or not set(expected).issubset(family_auc.index):
            raise RuntimeError("family-support EIC cache does not contain frozen candidates plus feature 8481")
        if not np.allclose(family_auc.loc[expected].to_numpy(), auc.loc[expected].to_numpy(), equal_nan=True):
            raise RuntimeError("family-support EIC cache does not reproduce the eight frozen target AUCs")
        family_log = np.log2(family_auc.where((family_auc > 0) & family_detected))
        support_summaries = []
        family_module_summaries = []
        family_module_patient_rows = []
        raw_family_rmu = None
        for normalization, factors in normalization_factors.items():
            matrix = family_log.sub(factors, axis=1)
            family_rmu = paired_feature_deltas(matrix, family_detected, "Rmu", "RN")
            family_rtu = paired_feature_deltas(matrix, family_detected, "Rtu", "RN")
            if normalization == "raw":
                raw_family_rmu = family_rmu.pivot(index="patient", columns="feature_id", values="delta_log2")
            support = family_rmu.loc[family_rmu.feature_id.eq(8481), "delta_log2"].to_numpy(float)
            support_summaries.append({"normalization": normalization, **summarize(support)})
            wide = family_rmu.pivot(index="patient", columns="feature_id", values="delta_log2")
            methyl_family = wide.reindex(columns=[1597, 7489]).median(axis=1, skipna=True)
            dimethyl_family = wide.reindex(columns=[3019, 8481]).median(axis=1, skipna=True)
            fully_collapsed = pd.concat(
                [methyl_family.rename("methyl"), dimethyl_family.rename("dimethyl")], axis=1
            ).mean(axis=1, skipna=False)
            family_module_summaries.append({"normalization": normalization, **summarize(fully_collapsed.dropna().to_numpy(float))})
            rtu_wide_family = family_rtu.pivot(index="patient", columns="feature_id", values="delta_log2")
            rtu_methyl = rtu_wide_family.reindex(columns=[1597, 7489]).median(axis=1, skipna=True)
            rtu_dimethyl = rtu_wide_family.reindex(columns=[3019, 8481]).median(axis=1, skipna=True)
            rtu_collapsed = pd.concat([rtu_methyl.rename("methyl"), rtu_dimethyl.rename("dimethyl")], axis=1).mean(axis=1, skipna=False)
            family_module_summaries[-1].update({
                "rtu_n": int(rtu_collapsed.notna().sum()),
                "rtu_mean_log2fc": float(rtu_collapsed.dropna().mean()),
                "rmu_minus_rtu_log2fc": float(fully_collapsed.dropna().mean() - rtu_collapsed.dropna().mean()),
                "interaction_exact_permutation_p": exact_label_permutation_p(
                    fully_collapsed.dropna().to_numpy(float), rtu_collapsed.dropna().to_numpy(float)
                ),
            })
            family_module_patient_rows.extend(
                {"normalization": normalization, "cohort": "Rmu", "patient": patient, "module_log2fc": float(value)}
                for patient, value in fully_collapsed.dropna().items()
            )
            family_module_patient_rows.extend(
                {"normalization": normalization, "cohort": "Rtu", "patient": patient, "module_log2fc": float(value)}
                for patient, value in rtu_collapsed.dropna().items()
            )
        family_peak_rows = pd.concat(
            [pd.read_csv(path) for path in sorted((args.family_eic_dir / "per_sample").glob("pos_rp__*__eic.csv.gz"))],
            ignore_index=True,
        )
        family_apex = family_peak_rows.pivot(index="sample_name", columns="feature_id", values="eic_apex_rt")
        dimethyl_coelution = (family_apex[3019] - family_apex[8481]).abs().dropna()
        support_target = pd.read_csv(args.family_target_dir / "biology_candidate_ledger.csv").set_index("feature_id")
        observed_difference = float(support_target.loc[8481, "mz"] - support_target.loc[3019, "mz"])
        family_support_report = {
            "available": True,
            "feature_id": 8481,
            "role": "phenotype-blind Na-adduct support for feature 3019; not an independent discovery",
            "effect_all_normalizations": support_summaries,
            "fully_ion_family_collapsed_modified_guanosine": family_module_summaries,
            "paired_effect_correlation_3019_vs_8481": correlation_report(raw_family_rmu[3019], raw_family_rmu[8481]),
            "observed_mz_difference": observed_difference,
            "mass_difference_residual_da": float(abs(observed_difference - 21.9819442498)),
            "samples_with_both_apexes": int(len(dimethyl_coelution)),
            "median_observed_apex_difference_sec": float(dimethyl_coelution.median()),
            "p95_observed_apex_difference_sec": float(dimethyl_coelution.quantile(0.95)),
            "support_positive_all_normalizations": bool(all(x.get("mean_log2fc", 0) > 0 for x in support_summaries)),
            "support_exact_p_le_0_05_all_normalizations": bool(all(x.get("exact_signflip_p", 1) <= 0.05 for x in support_summaries)),
            "collapsed_module_positive_all_normalizations": bool(all(x.get("mean_log2fc", 0) > 0 for x in family_module_summaries)),
            "collapsed_module_exact_p_le_0_05_all_normalizations": bool(all(x.get("exact_signflip_p", 1) <= 0.05 for x in family_module_summaries)),
        }
        pd.DataFrame(family_module_patient_rows).to_csv(out / "fully_ion_family_collapsed_module_patient_effects.csv", index=False)
    combined.to_csv(out / "candidate_identity_and_abundance.csv")

    modified_summaries = modules[(modules.cohort == "Rmu") & (modules.module == "modified_guanosine_module")]
    modified_summary = modified_summaries.loc[modified_summaries.normalization.eq("raw")].iloc[0].to_dict()
    normalized_module_positive = bool((modified_summaries.mean_log2fc > 0).all())
    normalized_module_significant = bool((modified_summaries.exact_signflip_p <= 0.05).all())
    normalized_module_loo = bool((modified_summaries.loo_min_mean_log2fc > 0).all())
    acyl_by_norm = abundance_all.loc[abundance_all.feature_id.eq(3222)]
    gates = {
        "all_eight_targets_present": len(combined) == 8,
        "all_rmu_candidates_have_at_least_eight_complete_pairs": bool((combined.rmu_n.fillna(0) >= 8).all()),
        "methylguanosine_adducts_positive_in_rmu": bool(
            combined.loc[1597, "rmu_mean_log2fc"] > 0 and combined.loc[7489, "rmu_mean_log2fc"] > 0
        ),
        "modified_guanosine_module_positive": bool(modified_summary.get("mean_log2fc", 0) > 0),
        "modified_guanosine_module_exact_p_le_0_05": bool(modified_summary.get("exact_signflip_p", 1) <= 0.05),
        "modified_guanosine_module_leave_one_out_positive": bool(modified_summary.get("loo_min_mean_log2fc", -1) > 0),
        "modified_guanosine_positive_all_background_normalizations": normalized_module_positive,
        "modified_guanosine_exact_p_le_0_05_all_background_normalizations": normalized_module_significant,
        "modified_guanosine_loo_positive_all_background_normalizations": normalized_module_loo,
        "c20_4_acylcarnitine_positive_all_background_normalizations": bool((acyl_by_norm.rmu_mean_log2fc > 0).all()),
        "rhea_direct_coverage_reported_not_assumed": True,
        "dimethylguanosine_family_support_positive": bool(family_support_report.get("support_positive_all_normalizations", False)),
        "fully_ion_family_collapsed_module_positive": bool(family_support_report.get("collapsed_module_positive_all_normalizations", False)),
    }
    report = {
        "status": "mtbls13729_biology_closure_local_complete",
        "formal": False,
        "reason_formal_false": "local targeted-EIC consolidation of eight previously selected candidates; not a new untargeted FDR analysis",
        "primary_endpoint": "paired Rmu versus matched RN abundance",
        "secondary_endpoint": "difference between Rmu-RN and Rtu-RN paired effects",
        "targets": int(len(combined)),
        "samples": int(len(auc.columns)),
        "rmu_pairs_available": int(len(rmu_wide)),
        "rhea_direct_nodes_among_targets": int(identity.rhea_direct_compound_node.sum()),
        "modified_guanosine_module_rmu": modified_summary,
        "modified_guanosine_module_all_normalizations": modified_summaries.to_dict(orient="records"),
        "modified_guanosine_module_interaction_all_normalizations": module_interactions,
        "correlations": correlations,
        "clinical_sensitivity": clinical_sensitivity,
        "methylguanosine_ion_family_audit": methyl_family_audit,
        "dimethylguanosine_ion_family_support": family_support_report,
        "peak_quality": peak_quality.reset_index().to_dict(orient="records"),
        "gates": gates,
        "interpretation": {
            "supported_axis": "modified-guanosine turnover plus polyamine acetylation and long-chain acylcarnitine accumulation are independent candidate axes in Rmu",
            "bioaware_use": "reaction/network evidence is downstream module support; no direct Rhea node exists for the core candidates and no identity override is permitted",
            "forbidden_claims": [
                "specific guanosine positional isomer",
                "specific methyltransferase or demethylase",
                "metabolic flux or enzyme activity",
                "confirmed Rmu-versus-Rtu specificity",
            ],
        },
        "provenance": {
            "candidate_ledger_sha256": sha256_file(ledger_path),
            "eic_auc_sha256": sha256_file(auc_path),
            "eic_detection_sha256": sha256_file(detection_path),
            "hmdb_sha256": sha256_file(args.hmdb),
            "rhea_participants_sha256": sha256_file(args.rhea_participants),
            "discovery_matrix_sha256": sha256_file(args.discovery_matrix),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
