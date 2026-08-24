#!/usr/bin/env python
"""Paired, sensitivity-aware MS1 analysis for MTBLS13729.

Primary endpoint: Rmu tumour versus its matched RN normal (abundance only).
Secondary endpoint: subtype interaction, (Rmu-RN) versus (Rtu-RN).
The script reports three normalization views and only marks a feature robust
when the biological direction is unchanged across them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, ttest_ind, wilcoxon


SAMPLE_RE = re.compile(r"^(P\d{2})-(Ltu|Rtu|Rmu|LN|RN)$")


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    if not len(p):
        return pd.Series(out, index=p_values.index)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    valid_idx = np.where(valid)[0]
    out[valid_idx[order]] = adjusted
    return pd.Series(out, index=p_values.index)


def parse_samples(columns: list[str]) -> pd.DataFrame:
    rows = []
    for sample in columns:
        match = SAMPLE_RE.match(sample)
        if not match:
            continue
        patient, suffix = match.groups()
        rows.append(
            {
                "sample_name": sample,
                "patient": patient,
                "suffix": suffix,
                "tissue": "tumor" if suffix in {"Ltu", "Rtu", "Rmu"} else "normal",
                "histology": "mucinous" if suffix == "Rmu" else ("tubular" if suffix in {"Ltu", "Rtu"} else "normal"),
            }
        )
    return pd.DataFrame(rows)


def pqn(log_matrix: pd.DataFrame) -> pd.DataFrame:
    reference = log_matrix.median(axis=1, skipna=True)
    quotients = log_matrix.sub(reference, axis=0)
    factors = quotients.median(axis=0, skipna=True)
    return log_matrix.sub(factors, axis=1)


def pair_center_drift_correct(matrix: pd.DataFrame, sample_meta: pd.DataFrame, order_map: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected = matrix.copy()
    diagnostics = []
    patient_to_samples = sample_meta.groupby("patient")["sample_name"].apply(list).to_dict()
    tissue_map = sample_meta.set_index("sample_name")["tissue"].to_dict()
    x_all = np.asarray([order_map.get(col, math.nan) for col in matrix.columns], dtype=float)
    center_x = float(np.nanmedian(x_all))
    for feature_id, row in matrix.iterrows():
        xs, ys = [], []
        for patient, samples in patient_to_samples.items():
            present = [sample for sample in samples if sample in row.index and np.isfinite(row[sample]) and sample in order_map]
            tumor = [s for s in present if tissue_map[s] == "tumor"]
            normal = [s for s in present if tissue_map[s] == "normal"]
            if len(tumor) == 1 and len(normal) == 1:
                xs.append((order_map[tumor[0]] + order_map[normal[0]]) / 2)
                ys.append((row[tumor[0]] + row[normal[0]]) / 2)
        slope = 0.0
        rho, p = math.nan, math.nan
        if len(xs) >= 10:
            rho, p = spearmanr(xs, ys)
            if np.isfinite(rho) and abs(rho) >= 0.30 and p < 0.10:
                # Median pairwise slope (Theil-Sen) is stable to biological outliers.
                slopes = []
                for i in range(len(xs)):
                    for j in range(i + 1, len(xs)):
                        if xs[j] != xs[i]:
                            slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
                slope = float(np.median(slopes)) if slopes else 0.0
                valid_cols = np.isfinite(x_all)
                corrected.loc[feature_id, np.asarray(matrix.columns)[valid_cols]] = (
                    row[np.asarray(matrix.columns)[valid_cols]].to_numpy(float) - slope * (x_all[valid_cols] - center_x)
                )
        diagnostics.append({"feature_id": feature_id, "n_pair_centers": len(xs), "run_order_rho": rho, "run_order_p": p, "applied_slope": slope})
    return corrected, pd.DataFrame(diagnostics).set_index("feature_id")


def paired_deltas(matrix: pd.DataFrame, sample_meta: pd.DataFrame, tumor_suffix: str, normal_suffix: str) -> dict[int, np.ndarray]:
    meta = sample_meta.set_index(["patient", "suffix"])["sample_name"]
    patients = sorted(set(sample_meta.loc[sample_meta["suffix"] == tumor_suffix, "patient"]))
    result = {}
    pairs = []
    for patient in patients:
        try:
            pairs.append((meta.loc[(patient, tumor_suffix)], meta.loc[(patient, normal_suffix)]))
        except KeyError:
            continue
    for feature_id, row in matrix.iterrows():
        values = [row[t] - row[n] for t, n in pairs if np.isfinite(row[t]) and np.isfinite(row[n])]
        result[feature_id] = np.asarray(values, dtype=float)
    return result


def summarize_deltas(deltas: dict[int, np.ndarray], prefix: str, min_pairs: int) -> pd.DataFrame:
    rows = []
    for feature_id, values in deltas.items():
        n = len(values)
        if n < min_pairs:
            rows.append({"feature_id": feature_id, f"{prefix}_n": n})
            continue
        mean = float(np.mean(values))
        median = float(np.median(values))
        t_p = float(ttest_1samp(values, 0.0).pvalue) if n >= 2 and np.std(values) > 0 else 1.0
        try:
            w_p = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            w_p = 1.0
        loo = [float(np.mean(np.delete(values, i))) for i in range(n)] if n > 1 else [mean]
        sign = np.sign(mean)
        loo_stability = float(np.mean([np.sign(item) == sign for item in loo])) if sign else 0.0
        rows.append(
            {
                "feature_id": feature_id,
                f"{prefix}_n": n,
                f"{prefix}_mean_log2fc": mean,
                f"{prefix}_median_log2fc": median,
                f"{prefix}_ttest_p": t_p,
                f"{prefix}_wilcoxon_p": w_p,
                f"{prefix}_loo_sign_stability": loo_stability,
            }
        )
    frame = pd.DataFrame(rows).set_index("feature_id")
    p_col = f"{prefix}_ttest_p"
    if p_col not in frame:
        frame[p_col] = math.nan
    frame[f"{prefix}_ttest_q"] = bh_adjust(frame[p_col])
    return frame


def interaction(rmu: dict[int, np.ndarray], rtu: dict[int, np.ndarray], min_pairs: int) -> pd.DataFrame:
    rows = []
    for feature_id in rmu:
        a, b = rmu[feature_id], rtu.get(feature_id, np.asarray([]))
        if len(a) < min_pairs or len(b) < min_pairs:
            rows.append({"feature_id": feature_id, "interaction_n_rmu": len(a), "interaction_n_rtu": len(b)})
            continue
        effect = float(np.mean(a) - np.mean(b))
        p = float(ttest_ind(a, b, equal_var=False).pvalue) if np.std(np.r_[a, b]) > 0 else 1.0
        rows.append({"feature_id": feature_id, "interaction_n_rmu": len(a), "interaction_n_rtu": len(b), "interaction_log2fc": effect, "interaction_p": p})
    frame = pd.DataFrame(rows).set_index("feature_id")
    if "interaction_p" not in frame:
        frame["interaction_p"] = math.nan
    frame["interaction_q"] = bh_adjust(frame["interaction_p"])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eic-dir", type=Path, default=Path("data/mtbls13729/ms1_eic_requant"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--audit", type=Path, default=Path("data/mtbls13729/ms1_acquisition_audit/file_audit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_paired_analysis"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--min-pairs", type=int, default=6)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit)
    report = {"status": "complete", "panels": {}}
    for panel in args.panels:
        auc = pd.read_csv(args.eic_dir / f"{panel}__eic_auc_matrix.csv.gz").set_index("feature_id")
        detected = pd.read_csv(args.eic_dir / f"{panel}__eic_detection_matrix.csv.gz").set_index("feature_id").astype(bool)
        targets = pd.read_csv(args.consensus_dir / f"{panel}__requantification_targets.csv.gz").set_index("feature_id")
        auc = auc.where(detected & (auc > 0))
        meta = parse_samples(list(auc.columns))
        order_rows = audit[(audit["panel"] == panel) & audit["sample_name"].isin(auc.columns)]
        order_map = dict(zip(order_rows["sample_name"], order_rows["injection_order"]))

        positive = auc.stack()
        pseudo = float(np.percentile(positive, 1) / 2) if len(positive) else 1.0
        log_raw = np.log2(auc + pseudo)
        normalized = {"log_raw": log_raw, "pqn": pqn(log_raw)}
        drifted, drift_diag = pair_center_drift_correct(normalized["pqn"], meta, order_map)
        normalized["pqn_pair_drift"] = drifted
        drift_diag.to_csv(out / f"{panel}__drift_diagnostics.csv.gz")

        variant_tables = []
        for variant, matrix in normalized.items():
            rmu = paired_deltas(matrix, meta, "Rmu", "RN")
            rtu = paired_deltas(matrix, meta, "Rtu", "RN")
            ltu = paired_deltas(matrix, meta, "Ltu", "LN")
            table = summarize_deltas(rmu, "rmu_vs_rn", args.min_pairs)
            table = table.join(summarize_deltas(rtu, "rtu_vs_rn", args.min_pairs), how="outer")
            table = table.join(summarize_deltas(ltu, "ltu_vs_ln", args.min_pairs), how="outer")
            table = table.join(interaction(rmu, rtu, args.min_pairs), how="outer")
            table.insert(0, "normalization", variant)
            table.to_csv(out / f"{panel}__{variant}__paired_stats.csv.gz")
            variant_tables.append(table.reset_index())

        combined = pd.concat(variant_tables, ignore_index=True)
        pivot = combined.pivot(index="feature_id", columns="normalization", values="rmu_vs_rn_mean_log2fc")
        p_pivot = combined.pivot(index="feature_id", columns="normalization", values="rmu_vs_rn_ttest_p")
        n_pivot = combined.pivot(index="feature_id", columns="normalization", values="rmu_vs_rn_n")
        interaction_pivot = combined.pivot(index="feature_id", columns="normalization", values="interaction_log2fc")
        interaction_p_pivot = combined.pivot(index="feature_id", columns="normalization", values="interaction_p")
        signs = np.sign(pivot)
        robust = (pivot.notna().sum(axis=1) == len(normalized)) & (signs.nunique(axis=1, dropna=True) == 1)
        interaction_same_direction = (np.sign(interaction_pivot) == np.sign(pivot)).all(axis=1)
        stable = pd.DataFrame(
            {
                "direction_consistent_all_normalizations": robust,
                "min_abs_rmu_log2fc": pivot.abs().min(axis=1),
                "max_abs_rmu_log2fc": pivot.abs().max(axis=1),
                "max_rmu_p_across_normalizations": p_pivot.max(axis=1),
                "min_rmu_pairs_across_normalizations": n_pivot.min(axis=1),
                "interaction_direction_matches_rmu": interaction_same_direction,
                "max_interaction_p_across_normalizations": interaction_p_pivot.max(axis=1),
            }
        )
        final = targets.join(stable, how="left")
        final["chromatography_edge_flag"] = (final["rt_sec"] < 60.0) | (final["rt_sec"] > 750.0)
        final["discovery_priority"] = (
            final["direction_consistent_all_normalizations"].fillna(False)
            & (final["min_abs_rmu_log2fc"] >= 0.5)
            & (final["max_rmu_p_across_normalizations"] < 0.05)
            & (final["min_rmu_pairs_across_normalizations"] >= 8)
            & final["interaction_direction_matches_rmu"].fillna(False)
            & (final["max_interaction_p_across_normalizations"] < 0.05)
            & (~final["chromatography_edge_flag"])
        )
        final.to_csv(out / f"{panel}__normalization_stability.csv.gz")
        final.loc[final["discovery_priority"]].to_csv(out / f"{panel}__discovery_priority_features.csv")
        report["panels"][panel] = {
            "n_samples": len(auc.columns),
            "n_targets": len(auc),
            "pseudocount": pseudo,
            "n_direction_consistent": int(stable["direction_consistent_all_normalizations"].sum()),
            "n_discovery_priority": int(final["discovery_priority"].sum()),
            "n_drift_corrected_features": int((drift_diag["applied_slope"] != 0).sum()),
            "primary_endpoint": "paired Rmu versus matched RN abundance",
            "secondary_endpoint": "(Rmu-RN) versus (Rtu-RN) interaction",
        }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
