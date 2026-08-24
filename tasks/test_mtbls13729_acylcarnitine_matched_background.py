"""Matched-feature-set specificity test for the acylcarnitine class score.

Random feature panels are matched on m/z, RT and detection prevalence without
using phenotype labels.  The test asks whether the observed Rmu-specific class
shift is stronger than generic features from the same analytical region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pqn(log_matrix: pd.DataFrame) -> pd.DataFrame:
    reference = log_matrix.median(axis=1, skipna=True)
    factors = log_matrix.sub(reference, axis=0).median(axis=0, skipna=True)
    return log_matrix.sub(factors, axis=1)


def group_scores(matrix: pd.DataFrame, features: list[int], tumour_suffix: str, min_features: int) -> np.ndarray:
    rows = []
    for patient_id in range(1, 31):
        patient = f"P{patient_id:02d}"
        tumour = f"{patient}-{tumour_suffix}"
        normal = f"{patient}-RN"
        if tumour not in matrix or normal not in matrix:
            continue
        delta = matrix.loc[features, tumour] - matrix.loc[features, normal]
        delta = delta[np.isfinite(delta)]
        if len(delta) >= min_features:
            rows.append(float(np.median(delta)))
    return np.asarray(rows, dtype=float)


def statistic(matrix: pd.DataFrame, features: list[int], min_features: int) -> tuple[float, float]:
    rmu = group_scores(matrix, features, "Rmu", min_features)
    rtu = group_scores(matrix, features, "Rtu", min_features)
    if len(rmu) < 8 or len(rtu) < 8:
        return np.nan, np.nan
    return float(np.mean(rmu)), float(np.mean(rmu) - np.mean(rtu))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--class-features", type=Path, default=Path("data/mtbls13729/acylcarnitine_panel/acylcarnitine_class_score_features.csv"))
    p.add_argument("--consensus", type=Path, default=Path("data/mtbls13729/ms1_consensus/pos_rp__consensus_metadata.csv.gz"))
    p.add_argument("--auc", type=Path, default=Path("data/mtbls13729/ms1_eic_requant_peakresolved/pos_rp__eic_auc_matrix.csv.gz"))
    p.add_argument("--detected", type=Path, default=Path("data/mtbls13729/ms1_eic_requant_peakresolved/pos_rp__eic_detection_matrix.csv.gz"))
    p.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/acylcarnitine_panel"))
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--min-features-per-pair", type=int, default=5)
    args = p.parse_args()

    class_rows = pd.read_csv(args.class_features)
    selected = sorted(class_rows.loc[class_rows.used_in_class_score.astype(bool), "feature_id"].dropna().astype(int).unique())
    consensus = pd.read_csv(args.consensus).set_index("feature_id")
    auc = pd.read_csv(args.auc).set_index("feature_id")
    detected = pd.read_csv(args.detected).set_index("feature_id").astype(bool)
    full = auc.where(detected & (auc > 0))
    positive = full.stack()
    pseudo = float(np.percentile(positive, 1) / 2) if len(positive) else 1.0
    matrix = pqn(np.log2(full + pseudo))

    acyl_all = set(class_rows.feature_id.dropna().astype(int))
    background = consensus.loc[consensus.index.intersection(matrix.index)].copy()
    background = background[background.keep_for_requantification.astype(bool)]
    background = background.loc[~background.index.isin(acyl_all)]
    pools: dict[int, np.ndarray] = {}
    for feature_id in selected:
        target = consensus.loc[feature_id]
        pool = background[
            (np.abs(background.mz - target.mz) <= 25.0)
            & (np.abs(background.rt_sec - target.rt_sec) <= 45.0)
            & (np.abs(background.global_prevalence - target.global_prevalence) <= 0.15)
        ]
        if len(pool) < 20:
            pool = background[
                (np.abs(background.mz - target.mz) <= 50.0)
                & (np.abs(background.rt_sec - target.rt_sec) <= 90.0)
                & (np.abs(background.global_prevalence - target.global_prevalence) <= 0.25)
            ]
        if pool.empty:
            raise SystemExit(f"No technical background matches for feature {feature_id}")
        pools[feature_id] = pool.index.to_numpy(int)

    observed_rmu, observed_interaction = statistic(matrix, selected, args.min_features_per_pair)
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for iteration in range(args.permutations):
        chosen: list[int] = []
        used: set[int] = set()
        for feature_id in selected:
            options = pools[feature_id]
            available = options[~np.isin(options, list(used))]
            if not len(available):
                available = options
            pick = int(rng.choice(available))
            chosen.append(pick)
            used.add(pick)
        rmu_stat, interaction_stat = statistic(matrix, chosen, args.min_features_per_pair)
        null_rows.append({
            "iteration": iteration,
            "rmu_mean_class_log2fc": rmu_stat,
            "interaction_difference": interaction_stat,
        })
    null = pd.DataFrame(null_rows)
    valid_rmu = null.rmu_mean_class_log2fc.dropna().to_numpy(float)
    valid_interaction = null.interaction_difference.dropna().to_numpy(float)
    report = {
        "status": "complete",
        "n_acylcarnitine_features": len(selected),
        "n_permutations": args.permutations,
        "observed_rmu_mean_class_log2fc": observed_rmu,
        "observed_interaction_difference": observed_interaction,
        "matched_background_one_sided_p_rmu": float((1 + np.sum(valid_rmu >= observed_rmu)) / (1 + len(valid_rmu))),
        "matched_background_one_sided_p_interaction": float(
            (1 + np.sum(valid_interaction >= observed_interaction)) / (1 + len(valid_interaction))
        ),
        "null_rmu_p95": float(np.percentile(valid_rmu, 95)),
        "null_interaction_p95": float(np.percentile(valid_interaction, 95)),
        "matching": "m/z, RT and global detection prevalence; phenotype-blind",
        "interpretation_limit": "Matched-set enrichment reduces, but cannot eliminate, acquisition-block confounding in a cohort without pooled QC.",
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    null.to_csv(out / "acylcarnitine_matched_background_null.csv.gz", index=False)
    (out / "matched_background_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
