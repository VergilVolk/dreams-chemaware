#!/usr/bin/env python
"""Analyze the frozen full-space MTBLS13729 targeted-EIC panel.

This is a technical re-extraction audit in the discovery cohort.  Candidate
selection preceded EIC outcomes, but selection used the same cohort's
peak-picker matrix.  The analysis therefore measures technical retention and
matched-control behavior; it is not independent biological confirmation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind

from audit_mtbls13729_full_requantifiable_space import (
    exact_signflip_p,
    paired_summary,
    pqn,
    sample_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data/mtbls13729/full_space_eic_targets_v1"
EIC = ROOT / "data/mtbls13729/full_space_eic_v1"
CONSENSUS = ROOT / "data/mtbls13729/ms1_consensus"
OUT = ROOT / "data/mtbls13729/full_space_eic_analysis_v1"
PANELS = ("neg_rp", "pos_rp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def interaction(
    matrix: pd.DataFrame,
    rmu_pairs: list[tuple[str, str]],
    rtu_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    a = matrix[[x for pair in rmu_pairs for x in pair]]
    b = matrix[[x for pair in rtu_pairs for x in pair]]
    da = a[[p[0] for p in rmu_pairs]].to_numpy(float) - a[[p[1] for p in rmu_pairs]].to_numpy(float)
    db = b[[p[0] for p in rtu_pairs]].to_numpy(float) - b[[p[1] for p in rtu_pairs]].to_numpy(float)
    rows = []
    for left, right in zip(da, db):
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        if len(left) < 2 or len(right) < 2:
            rows.append({"interaction_n_rmu": len(left), "interaction_n_rtu": len(right)})
        else:
            rows.append({
                "interaction_n_rmu": len(left),
                "interaction_n_rtu": len(right),
                "interaction_log2fc": float(left.mean() - right.mean()),
                "interaction_p": float(ttest_ind(left, right, equal_var=False).pvalue),
            })
    return pd.DataFrame(rows, index=matrix.index)


def full_background_pqn(panel: str, samples: list[str]) -> pd.Series:
    matrix = pd.read_csv(CONSENSUS / f"{panel}__discovery_intensity_matrix.csv.gz").set_index("feature_id")
    targets = pd.read_csv(CONSENSUS / f"{panel}__requantification_targets.csv.gz").set_index("feature_id")
    target_ids = targets.index.intersection(matrix.index)
    matrix = matrix.loc[target_ids, samples].astype(float).where(lambda x: x > 0.0)
    positive = matrix.stack().to_numpy(float)
    pseudo = float(np.percentile(positive, 1) / 2.0)
    log_matrix = np.log2(matrix + pseudo)
    _, factors = pqn(log_matrix, targets.loc[target_ids, "global_prevalence"])
    return factors


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    combined_rows = []
    for panel in PANELS:
        target_path = TARGETS / f"{panel}__requantification_targets.csv.gz"
        targets = pd.read_csv(target_path).set_index("feature_id")
        auc = pd.read_csv(EIC / f"{panel}__eic_auc_matrix.csv.gz").set_index("feature_id")
        detected = pd.read_csv(EIC / f"{panel}__eic_detection_matrix.csv.gz").set_index("feature_id")
        detected = detected.apply(lambda column: column.astype(str).str.lower().isin({"true", "1"}))
        if set(targets.index) != set(auc.index):
            raise RuntimeError(f"{panel}: target/EIC feature mismatch")
        auc = auc.loc[targets.index].where(detected.loc[targets.index] & (auc.loc[targets.index] > 0.0))
        samples = list(auc.columns)
        positive = auc.stack().to_numpy(float)
        pseudo = float(np.percentile(positive, 1) / 2.0)
        raw = np.log2(auc + pseudo)
        factors = full_background_pqn(panel, samples)
        normalized = {"log_raw": raw, "global_pqn_prev60": raw.sub(factors, axis=1)}
        rmu_pairs = sample_pairs(samples, "Rmu", "RN")
        rtu_pairs = sample_pairs(samples, "Rtu", "RN")
        stats = []
        for name, matrix in normalized.items():
            summary = paired_summary(matrix, rmu_pairs, "rmu")
            summary = summary.join(interaction(matrix, rmu_pairs, rtu_pairs))
            summary["normalization"] = name
            stats.append(summary.reset_index())
        stats = pd.concat(stats, ignore_index=True)
        mean = stats.pivot(index="feature_id", columns="normalization", values="rmu_mean_log2fc")
        pvalue = stats.pivot(index="feature_id", columns="normalization", values="rmu_exact_signflip_p")
        n_pairs = stats.pivot(index="feature_id", columns="normalization", values="rmu_n")
        loo = stats.pivot(index="feature_id", columns="normalization", values="rmu_loo_direction_stable")
        interaction_effect = stats.pivot(index="feature_id", columns="normalization", values="interaction_log2fc")
        interaction_p = stats.pivot(index="feature_id", columns="normalization", values="interaction_p")
        result = targets.copy()
        result["eic_detection_fraction"] = detected.loc[result.index].mean(axis=1)
        result["eic_raw_rmu_log2fc"] = mean["log_raw"]
        result["eic_pqn_rmu_log2fc"] = mean["global_pqn_prev60"]
        result["eic_min_abs_rmu_log2fc"] = mean.abs().min(axis=1)
        result["eic_max_exact_p"] = pvalue.max(axis=1)
        result["eic_min_pairs"] = n_pairs.min(axis=1)
        result["eic_loo_stable_all"] = loo.fillna(False).all(axis=1)
        result["eic_direction_consistent"] = np.sign(mean).nunique(axis=1, dropna=True) == 1
        result["eic_matches_discovery_direction"] = (
            np.sign(result["eic_raw_rmu_log2fc"]) == np.sign(result["raw_mean_log2fc"])
        ) & (
            np.sign(result["eic_pqn_rmu_log2fc"]) == np.sign(result["pqn_mean_log2fc"])
        )
        result["technical_retention_gate"] = (
            (result["eic_detection_fraction"] >= 0.80)
            & (result["eic_min_pairs"] >= 8)
            & result["eic_direction_consistent"]
            & result["eic_matches_discovery_direction"]
            & (result["eic_min_abs_rmu_log2fc"] >= 0.50)
            & (result["eic_max_exact_p"] <= 0.05)
            & result["eic_loo_stable_all"]
        )
        result["control_false_positive_gate"] = (
            result["target_role"].eq("matched_null_control")
            & (result["eic_detection_fraction"] >= 0.80)
            & (result["eic_min_pairs"] >= 8)
            & result["eic_direction_consistent"]
            & (result["eic_min_abs_rmu_log2fc"] >= 0.50)
            & (result["eic_max_exact_p"] <= 0.05)
            & result["eic_loo_stable_all"]
        )
        result["raw_interaction_log2fc"] = interaction_effect["log_raw"]
        result["pqn_interaction_log2fc"] = interaction_effect["global_pqn_prev60"]
        result["max_interaction_p"] = interaction_p.max(axis=1)
        result = result.reset_index()
        result.insert(0, "panel", panel)
        result.to_csv(OUT / f"{panel}__targeted_eic_results.csv", index=False)
        combined_rows.append(result)

        candidate = result[~result["target_role"].eq("matched_null_control")]
        control = result[result["target_role"].eq("matched_null_control")]
        candidate_pass = int(candidate["technical_retention_gate"].sum())
        control_pass = int(control["control_false_positive_gate"].sum())
        table = [[candidate_pass, len(candidate) - candidate_pass], [control_pass, len(control) - control_pass]]
        odds, fisher_p = fisher_exact(table, alternative="greater") if len(candidate) and len(control) else (math.nan, math.nan)
        family = candidate.groupby("ion_family_id").agg(
            targets=("feature_id", "count"),
            retained=("technical_retention_gate", "max"),
            best_exact_p=("eic_max_exact_p", "min"),
            largest_abs_effect=("eic_min_abs_rmu_log2fc", "max"),
            annotated_name=("best_name", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
        ).reset_index()
        family.to_csv(OUT / f"{panel}__candidate_family_retention.csv", index=False)
        reports[panel] = {
            "targets": int(len(result)),
            "candidate_targets": int(len(candidate)),
            "matched_null_controls": int(len(control)),
            "candidate_technical_retention": candidate_pass,
            "candidate_retention_fraction": float(candidate_pass / len(candidate)) if len(candidate) else math.nan,
            "matched_control_false_positive": control_pass,
            "matched_control_false_positive_fraction": float(control_pass / len(control)) if len(control) else math.nan,
            "candidate_vs_control_fisher_odds": float(odds),
            "candidate_vs_control_fisher_p": float(fisher_p),
            "candidate_families": int(candidate["ion_family_id"].nunique()),
            "retained_candidate_families": int(family["retained"].sum()),
            "chromatography_edge_candidate_targets": int(((candidate["rt_sec"] < 60.0) | (candidate["rt_sec"] > 750.0)).sum()),
            "retained_nonedge_candidate_targets": int((candidate["technical_retention_gate"] & (candidate["rt_sec"] >= 60.0) & (candidate["rt_sec"] <= 750.0)).sum()),
            "provenance": {
                "targets_sha256": sha256(target_path),
                "auc_sha256": sha256(EIC / f"{panel}__eic_auc_matrix.csv.gz"),
                "detection_sha256": sha256(EIC / f"{panel}__eic_detection_matrix.csv.gz"),
            },
        }
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(OUT / "all_targeted_eic_results.csv", index=False)
    report = {
        "status": "mtbls13729_full_space_eic_analysis_complete",
        "formal": False,
        "reason_formal_false": "same-cohort technical re-extraction after discovery-matrix selection",
        "panels": reports,
        "primary_endpoint": "paired Rmu versus matched RN abundance",
        "secondary_endpoint": "Rmu-RN versus Rtu-RN interaction; not a subtype-specificity claim",
        "claim_limit": (
            "Technical retention does not establish full-space biological FDR, independent replication, "
            "metabolite identity, subtype specificity, flux, enzyme activity, or causal mechanism."
        ),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
