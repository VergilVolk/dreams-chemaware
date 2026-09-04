"""Phenotype-blind acylcarnitine class-level paired analysis.

Species enter the score only through MS/MS motif support, coverage and chain
length.  Tumour labels are not used for feature selection.  The endpoint is a
patient-level median of within-pair log2 changes across the selected panel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_ind, wilcoxon


SAMPLE_RE = re.compile(r"^(P\d{2})-(Ltu|Rtu|Rmu|LN|RN)$")


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def exact_label_permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.r_[a, b]
    n_a = len(a)
    observed = abs(float(np.mean(a) - np.mean(b)))
    exceed = 0
    total = 0
    indices = np.arange(len(pooled))
    for choice in itertools.combinations(indices, n_a):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(choice)] = True
        statistic = abs(float(np.mean(pooled[mask]) - np.mean(pooled[~mask])))
        exceed += int(statistic >= observed - 1e-12)
        total += 1
    return float(exceed / total)


def pqn_factors(log_matrix: pd.DataFrame) -> pd.Series:
    reference = log_matrix.median(axis=1, skipna=True)
    return log_matrix.sub(reference, axis=0).median(axis=0, skipna=True)


def pair_scores(
    matrix: pd.DataFrame,
    suffix: str,
    normal_suffix: str,
    min_features: int,
    order_map: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for patient_id in range(1, 31):
        patient = f"P{patient_id:02d}"
        tumour = f"{patient}-{suffix}"
        normal = f"{patient}-{normal_suffix}"
        if tumour not in matrix or normal not in matrix:
            continue
        delta = matrix[tumour] - matrix[normal]
        delta = delta[np.isfinite(delta)]
        if len(delta) < min_features:
            continue
        rows.append({
            "patient": patient,
            "tumour_suffix": suffix,
            "n_features": int(len(delta)),
            "class_median_log2fc": float(np.median(delta)),
            "class_mean_log2fc": float(np.mean(delta)),
            "pair_center_order": float(np.mean([order_map.get(tumour, np.nan), order_map.get(normal, np.nan)])),
        })
    return pd.DataFrame(rows)


def pair_scores_collapsed(
    matrix: pd.DataFrame,
    feature_to_chain: dict[int, str],
    suffix: str,
    normal_suffix: str,
    min_groups: int,
    order_map: dict[str, float],
) -> pd.DataFrame:
    """Collapse repeated RT features to one mass-based Cn:u group per pair."""
    rows = []
    for patient_id in range(1, 31):
        patient = f"P{patient_id:02d}"
        tumour = f"{patient}-{suffix}"
        normal = f"{patient}-{normal_suffix}"
        if tumour not in matrix or normal not in matrix:
            continue
        delta = matrix[tumour] - matrix[normal]
        frame = pd.DataFrame(
            {
                "delta": delta,
                "chain": [feature_to_chain.get(int(idx)) for idx in delta.index],
            }
        ).dropna()
        chain_delta = frame.groupby("chain")["delta"].median()
        chain_delta = chain_delta[np.isfinite(chain_delta)]
        if len(chain_delta) < min_groups:
            continue
        rows.append({
            "patient": patient,
            "tumour_suffix": suffix,
            "n_features": int(len(chain_delta)),
            "class_median_log2fc": float(np.median(chain_delta)),
            "class_mean_log2fc": float(np.mean(chain_delta)),
            "pair_center_order": float(np.mean([order_map.get(tumour, np.nan), order_map.get(normal, np.nan)])),
        })
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> dict[str, float | int]:
    values = scores.class_median_log2fc.to_numpy(float)
    finite_order = np.isfinite(scores.pair_center_order.to_numpy(float)) & np.isfinite(values)
    if finite_order.sum() >= 3:
        order_rho, order_p = spearmanr(
            scores.loc[finite_order, "pair_center_order"].to_numpy(float), values[finite_order]
        )
    else:
        order_rho, order_p = np.nan, np.nan
    return {
        "n_pairs": int(len(values)),
        "mean_class_log2fc": float(np.mean(values)),
        "median_class_log2fc": float(np.median(values)),
        "ttest_p": float(ttest_1samp(values, 0).pvalue),
        "wilcoxon_p": float(wilcoxon(values).pvalue),
        "exact_signflip_p": exact_signflip_p(values),
        "min_features_per_pair": int(scores.n_features.min()),
        "median_features_per_pair": float(scores.n_features.median()),
        "pair_delta_vs_order_spearman": float(order_rho),
        "pair_delta_vs_order_p": float(order_p),
    }


def clinical_sensitivity(scores: pd.DataFrame, clinical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Audit the conservative score against published MMR/BRAF labels.

    This is deliberately a sensitivity analysis, not covariate-adjusted
    inference: the study contains only 10 Rmu pairs (4 pMMR, 6 dMMR).
    """
    tumour = clinical[clinical.tissue.eq("Tumor")].copy()
    tumour["patient"] = "P" + tumour.patient_number.astype(int).astype(str).str.zfill(2)
    linked = scores.merge(tumour[["patient", "pathology", "braf", "mmr"]], on="patient", how="left")
    if linked[["pathology", "braf", "mmr"]].isna().any().any():
        raise ValueError("Clinical Table S2 could not be linked to every paired score.")
    report: dict[str, dict] = {
        "purpose": (
            "MMR/BRAF sensitivity only; small subgroup sizes preclude a definitive "
            "adjusted pathology-specific claim."
        ),
        "variants": {},
    }
    for variant, frame in linked.groupby("normalization", sort=False):
        variant_report: dict[str, dict] = {"within_pathology_mmr": {}, "pMMR_Rmu_vs_Rtu": {}}
        for (pathology, mmr), group in frame.groupby(["pathology", "mmr"], sort=False):
            values = group.class_median_log2fc.to_numpy(float)
            if len(values) >= 2:
                variant_report["within_pathology_mmr"][f"{pathology}_{mmr}"] = summarize(group)
        rmu = frame[(frame.pathology == "Rmu") & (frame.mmr == "pMMR")].class_median_log2fc.to_numpy(float)
        rtu = frame[(frame.pathology == "Rtu") & (frame.mmr == "pMMR")].class_median_log2fc.to_numpy(float)
        if len(rmu) >= 2 and len(rtu) >= 2:
            variant_report["pMMR_Rmu_vs_Rtu"] = {
                "n_Rmu": int(len(rmu)),
                "n_Rtu": int(len(rtu)),
                "difference_in_mean_class_log2fc": float(np.mean(rmu) - np.mean(rtu)),
                "exact_label_permutation_p": exact_label_permutation_p(rmu, rtu),
            }
        report["variants"][variant] = variant_report
    return linked, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--species", type=Path, default=Path("data/mtbls13729/acylcarnitine_panel/acylcarnitine_rt_species.csv"))
    p.add_argument("--auc", type=Path, default=Path("data/mtbls13729/ms1_eic_requant_peakresolved/pos_rp__eic_auc_matrix.csv.gz"))
    p.add_argument("--detected", type=Path, default=Path("data/mtbls13729/ms1_eic_requant_peakresolved/pos_rp__eic_detection_matrix.csv.gz"))
    p.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/acylcarnitine_panel"))
    p.add_argument("--acquisition-audit", type=Path, default=Path("data/mtbls13729/ms1_acquisition_audit/file_audit.csv"))
    p.add_argument("--drift-diagnostics", type=Path, default=Path("data/mtbls13729/ms1_paired_analysis_peakresolved/pos_rp__drift_diagnostics.csv.gz"))
    p.add_argument("--clinical", type=Path, default=Path("data/mtbls13729/clinical_metadata_s2.tsv"))
    p.add_argument("--min-carbon", type=int, default=12)
    p.add_argument("--min-ms2-samples", type=int, default=5)
    p.add_argument("--min-features-per-pair", type=int, default=5)
    args = p.parse_args()

    species = pd.read_csv(args.species)
    selected = species[
        (species.carbon >= args.min_carbon)
        & (species.n_samples_with_ms2 >= args.min_ms2_samples)
        & species.feature_id.notna()
    ].copy()
    # Isobaric [M+H]+/[M+Na]+ hypotheses can map to the same observed feature.
    # They contribute once to the class score; structural ambiguity is retained.
    auc = pd.read_csv(args.auc).set_index("feature_id")
    detected = pd.read_csv(args.detected).set_index("feature_id").astype(bool)
    feature_ids = sorted(set(selected.feature_id.astype(int).unique()) & set(auc.index.astype(int)))
    selected["used_in_class_score"] = selected.feature_id.astype("Int64").isin(feature_ids)
    # A feature can have several isobaric Cn:u/adduct hypotheses. Retain the
    # most widely MS2-supported hypothesis only as a phenotype-blind grouping
    # label for the redundancy sensitivity analysis below.
    representative_hypotheses = (
        selected[selected["used_in_class_score"]]
        .sort_values(["n_samples_with_ms2", "n_ms2_spectra"], ascending=[False, False])
        .drop_duplicates("feature_id", keep="first")
    )
    feature_to_chain = dict(
        zip(
            representative_hypotheses.feature_id.astype(int),
            representative_hypotheses.acyl_chain.astype(str),
        )
    )
    full = auc.where(detected & (auc > 0))
    positive = full.stack()
    pseudo = float(np.percentile(positive, 1) / 2) if len(positive) else 1.0
    log_raw = np.log2(full + pseudo)
    factors = pqn_factors(log_raw)
    pqn_matrix = log_raw.loc[feature_ids].sub(factors, axis=1)
    if args.acquisition_audit.exists():
        acquisition = pd.read_csv(args.acquisition_audit)
        acquisition = acquisition[acquisition.panel == "pos_rp"]
        order_map = dict(zip(acquisition.sample_name, acquisition.injection_order))
    else:
        print(f"[acylcarnitine] optional acquisition audit absent: {args.acquisition_audit}", flush=True)
        order_map = {}
    order_values = np.asarray([order_map.get(column, np.nan) for column in pqn_matrix.columns], dtype=float)
    order_center = float(np.nanmedian(order_values))
    variants = {
        "log_raw": log_raw.loc[feature_ids],
        "pqn": pqn_matrix,
    }
    drift_available = args.drift_diagnostics.exists() and bool(order_map)
    if drift_available:
        drift = pd.read_csv(args.drift_diagnostics).set_index("feature_id")
        drifted = pqn_matrix.copy()
        for feature_id in feature_ids:
            slope = float(drift.loc[feature_id, "applied_slope"]) if feature_id in drift.index else 0.0
            valid = np.isfinite(order_values)
            drifted.loc[feature_id, np.asarray(drifted.columns)[valid]] = (
                pqn_matrix.loc[feature_id, np.asarray(drifted.columns)[valid]].to_numpy(float)
                - slope * (order_values[valid] - order_center)
            )
        variants["pqn_pair_drift"] = drifted
    else:
        print(
            f"[acylcarnitine] pair-drift sensitivity unavailable; diagnostics={args.drift_diagnostics}",
            flush=True,
        )

    pair_frames = []
    collapsed_pair_frames = []
    report = {
        "status": "complete",
        "selection_is_phenotype_blind": True,
        "n_selected_rt_hypotheses": int(len(selected)),
        "n_unique_ms1_features": int(len(feature_ids)),
        "parameters": {
            "min_carbon": args.min_carbon,
            "min_ms2_samples": args.min_ms2_samples,
            "min_features_per_pair": args.min_features_per_pair,
            "pseudocount": pseudo,
            "pair_drift_sensitivity_available": drift_available,
        },
        "variants": {},
        "chain_collapsed_sensitivity": {
            "description": (
                "Repeated RT features sharing the same phenotype-blind mass-based Cn:u hypothesis "
                "are collapsed to one within-patient chain effect before calculating the class score."
            ),
            "n_chain_hypotheses": int(len(set(feature_to_chain.values()))),
            "variants": {},
        },
        "interpretation_limit": (
            "The class score supports steady-state abundance remodeling only. "
            "It does not establish beta-oxidation flux, enzyme activity, double-bond position or a unique isomer."
        ),
    }
    for variant, matrix in variants.items():
        rmu = pair_scores(matrix, "Rmu", "RN", args.min_features_per_pair, order_map)
        rtu = pair_scores(matrix, "Rtu", "RN", args.min_features_per_pair, order_map)
        rmu["normalization"] = variant
        rtu["normalization"] = variant
        pair_frames.extend([rmu, rtu])
        rmu_summary = summarize(rmu)
        rtu_summary = summarize(rtu)
        a = rmu.class_median_log2fc.to_numpy(float)
        b = rtu.class_median_log2fc.to_numpy(float)
        interaction = {
            "difference_in_mean_class_log2fc": float(np.mean(a) - np.mean(b)),
            "welch_p": float(ttest_ind(a, b, equal_var=False).pvalue),
            "exact_label_permutation_p": exact_label_permutation_p(a, b),
        }
        report["variants"][variant] = {"Rmu_vs_RN": rmu_summary, "Rtu_vs_RN": rtu_summary, "interaction": interaction}

        collapsed_rmu = pair_scores_collapsed(
            matrix, feature_to_chain, "Rmu", "RN", args.min_features_per_pair, order_map
        )
        collapsed_rtu = pair_scores_collapsed(
            matrix, feature_to_chain, "Rtu", "RN", args.min_features_per_pair, order_map
        )
        collapsed_rmu["normalization"] = variant
        collapsed_rtu["normalization"] = variant
        collapsed_pair_frames.extend([collapsed_rmu, collapsed_rtu])
        ca = collapsed_rmu.class_median_log2fc.to_numpy(float)
        cb = collapsed_rtu.class_median_log2fc.to_numpy(float)
        report["chain_collapsed_sensitivity"]["variants"][variant] = {
            "Rmu_vs_RN": summarize(collapsed_rmu),
            "Rtu_vs_RN": summarize(collapsed_rtu),
            "interaction": {
                "difference_in_mean_class_log2fc": float(np.mean(ca) - np.mean(cb)),
                "welch_p": float(ttest_ind(ca, cb, equal_var=False).pvalue),
                "exact_label_permutation_p": exact_label_permutation_p(ca, cb),
            },
        }

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "acylcarnitine_class_score_features.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(out / "acylcarnitine_class_pair_scores.csv", index=False)
    pd.concat(collapsed_pair_frames, ignore_index=True).to_csv(
        out / "acylcarnitine_chain_collapsed_pair_scores.csv", index=False
    )
    if args.clinical.exists():
        clinical = pd.read_csv(args.clinical, sep="\t")
        clinical_scores, clinical_report = clinical_sensitivity(
            pd.concat(collapsed_pair_frames, ignore_index=True), clinical
        )
        clinical_scores.to_csv(
            out / "acylcarnitine_chain_collapsed_pair_scores_clinical.tsv",
            sep="\t",
            index=False,
        )
        report["clinical_sensitivity_table_s2"] = clinical_report
    else:
        report["clinical_sensitivity_table_s2"] = {
            "status": "not_run",
            "reason": f"missing optional clinical table {args.clinical}",
        }
    (out / "class_score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
