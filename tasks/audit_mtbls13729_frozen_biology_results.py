#!/usr/bin/env python
"""Audit the frozen-panel MTBLS13729 paired biology result without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NORMALIZATIONS = ("log_raw", "pqn", "pqn_pair_drift")
E6 = "e6_fixed_v2_sw2"


def finite_max(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].max(axis=1, skipna=True)


def scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=Path("data/mtbls13729/frozen_biology_panel_v1"))
    parser.add_argument("--biology-dir", type=Path, default=Path("data/mtbls13729/threeway_biology_v1"))
    parser.add_argument("--eic-dir", type=Path, default=Path("data/mtbls13729/ms1_eic_requant_threeway_frozen_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/threeway_biology_audit_v2"))
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty biology audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "mtbls13729_frozen_biology_result_audit_complete",
        "formal": False,
        "panels": {},
    }
    for panel in ("neg_rp", "pos_rp"):
        stable = pd.read_csv(args.biology_dir / f"{panel}__normalization_stability.csv.gz")
        detected = pd.read_csv(args.eic_dir / f"{panel}__eic_detection_matrix.csv.gz").set_index("feature_id")
        stats = []
        for normalization in NORMALIZATIONS:
            frame = pd.read_csv(args.biology_dir / f"{panel}__{normalization}__paired_stats.csv.gz")
            keep = [
                "feature_id", "rmu_vs_rn_n", "rmu_vs_rn_mean_log2fc", "rmu_vs_rn_ttest_p",
                "rmu_vs_rn_ttest_q", "rmu_vs_rn_loo_sign_stability", "interaction_log2fc",
                "interaction_p", "interaction_q",
            ]
            frame = frame[[column for column in keep if column in frame.columns]].copy()
            frame = frame.rename(columns={column: f"{normalization}__{column}" for column in frame.columns if column != "feature_id"})
            stats.append(frame)
        combined = stable.copy()
        for frame in stats:
            combined = combined.merge(frame, on="feature_id", how="left", validate="one_to_one")
        combined["targeted_eic_detection_fraction"] = combined["feature_id"].map(detected.mean(axis=1))
        rmu_q = [f"{normalization}__rmu_vs_rn_ttest_q" for normalization in NORMALIZATIONS]
        interaction_q = [f"{normalization}__interaction_q" for normalization in NORMALIZATIONS]
        combined["max_rmu_q_across_normalizations"] = finite_max(combined, rmu_q)
        combined["max_interaction_q_across_normalizations"] = finite_max(combined, interaction_q)
        combined["primary_rmu_fdr10_robust"] = (
            combined["discovery_priority"].fillna(False)
            & combined["max_rmu_q_across_normalizations"].lt(0.10)
        )
        combined["primary_rmu_fdr05_robust"] = (
            combined["discovery_priority"].fillna(False)
            & combined["max_rmu_q_across_normalizations"].lt(0.05)
        )
        combined["secondary_interaction_fdr10_robust"] = (
            combined["discovery_priority"].fillna(False)
            & combined["max_interaction_q_across_normalizations"].lt(0.10)
        )
        combined["secondary_interaction_fdr05_robust"] = (
            combined["discovery_priority"].fillna(False)
            & combined["max_interaction_q_across_normalizations"].lt(0.05)
        )
        combined["joint_fdr10_robust"] = (
            combined["primary_rmu_fdr10_robust"] & combined["secondary_interaction_fdr10_robust"]
        )
        combined["joint_fdr05_robust"] = (
            combined["primary_rmu_fdr05_robust"] & combined["secondary_interaction_fdr05_robust"]
        )
        combined.to_csv(out / f"{panel}__audited_features.csv.gz", index=False, compression="gzip")
        priority = combined.loc[combined["discovery_priority"].fillna(False)].copy()
        priority.to_csv(out / f"{panel}__priority_features.csv", index=False)

        anchor_rows = combined.loc[combined.get("predeclared_c20_4_anchor", False).fillna(False)] if "predeclared_c20_4_anchor" in combined else combined.iloc[0:0]
        anchor = None
        if len(anchor_rows):
            row = anchor_rows.iloc[0]
            anchor_columns = [
                "feature_id", "mz", "rt_sec", "analysis_tier", "targeted_eic_detection_fraction",
                "direction_consistent_all_normalizations", "discovery_priority",
                "primary_rmu_fdr10_robust", "primary_rmu_fdr05_robust",
                "secondary_interaction_fdr10_robust", "secondary_interaction_fdr05_robust",
                "joint_fdr10_robust", "joint_fdr05_robust",
                "max_rmu_q_across_normalizations", "max_interaction_q_across_normalizations",
            ]
            for normalization in NORMALIZATIONS:
                anchor_columns += [
                    f"{normalization}__rmu_vs_rn_n", f"{normalization}__rmu_vs_rn_mean_log2fc",
                    f"{normalization}__rmu_vs_rn_ttest_p", f"{normalization}__rmu_vs_rn_ttest_q",
                    f"{normalization}__interaction_log2fc", f"{normalization}__interaction_p",
                    f"{normalization}__interaction_q",
                ]
            anchor = {column: scalar(row[column]) for column in anchor_columns if column in row.index}
        identity_columns = [
            f"{E6}_name", f"{E6}_inchikey", f"{E6}_smiles", "analysis_tier",
        ]
        summary_columns = [
            "feature_id", "mz", "rt_sec", *identity_columns, "targeted_eic_detection_fraction",
            "min_abs_rmu_log2fc", "max_abs_rmu_log2fc", "max_rmu_q_across_normalizations",
            "max_interaction_q_across_normalizations", "primary_rmu_fdr10_robust",
            "secondary_interaction_fdr10_robust", "joint_fdr10_robust",
        ]
        candidate_summaries = [
            {column: scalar(row[column]) for column in summary_columns if column in row.index}
            for _, row in priority.sort_values(
                ["max_rmu_q_across_normalizations", "max_interaction_q_across_normalizations", "feature_id"],
                kind="stable",
            ).iterrows()
        ]
        report["panels"][panel] = {
            "targets": int(len(combined)),
            "median_detection_fraction": float(combined["targeted_eic_detection_fraction"].median()),
            "direction_consistent": int(combined["direction_consistent_all_normalizations"].fillna(False).sum()),
            "nominal_discovery_priority": int(combined["discovery_priority"].fillna(False).sum()),
            "primary_rmu_fdr10_robust": int(combined["primary_rmu_fdr10_robust"].sum()),
            "primary_rmu_fdr05_robust": int(combined["primary_rmu_fdr05_robust"].sum()),
            "secondary_interaction_fdr10_robust": int(combined["secondary_interaction_fdr10_robust"].sum()),
            "secondary_interaction_fdr05_robust": int(combined["secondary_interaction_fdr05_robust"].sum()),
            "joint_fdr10_robust": int(combined["joint_fdr10_robust"].sum()),
            "joint_fdr05_robust": int(combined["joint_fdr05_robust"].sum()),
            "nominal_priority_candidates": candidate_summaries,
            "anchor": anchor,
        }

    report["claim_limit"] = (
        "Static abundance discovery in 10 Rmu pairs. The preregistered primary Rmu-vs-RN endpoint is reported "
        "separately from the secondary Rmu-vs-Rtu interaction; joint FDR is an additional stronger tier, not the "
        "definition of primary-endpoint success. Annotation identity remains Level 2 unless confirmed by a standard; "
        "abundance does not establish flux or enzyme activity."
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
