#!/usr/bin/env python
"""Freeze a targeted-EIC panel from the full re-quantifiable-space audit.

The panel includes every feature passing the predeclared parametric sensitivity
screen (effect/direction gate plus full-space t-test q <= 0.10), all negative-
mode effect/direction candidates, the six prior phenotype-blind annotated
screen candidates, and one phenotype-blind coordinate/prevalence-matched null
control for each selected ion family where available.

Selection uses the same cohort and therefore targeted EIC is a technical
re-extraction, not independent biological validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/mtbls13729/full_requantifiable_space_audit_v1"
CONSENSUS = ROOT / "data/mtbls13729/ms1_consensus"
ANNOTATED_PRIORITY = ROOT / "data/mtbls13729/full_annotated_feature_audit_v1/all_priority.csv"
OUT = ROOT / "data/mtbls13729/full_space_eic_targets_v1"
PANELS = ("neg_rp", "pos_rp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matched_controls(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    pool = frame[
        (~frame["effect_and_direction_gate"].fillna(False))
        & (frame["max_exact_p"].fillna(1.0) >= 0.50)
        & (frame["max_ttest_p"].fillna(1.0) >= 0.50)
        & (frame["min_pairs"].fillna(0) >= 8)
    ].copy()
    if pool.empty or selected.empty:
        return pool.iloc[:0].copy()
    used_features: set[int] = set()
    used_families = set(selected["ion_family_id"].dropna().astype(int))
    rows = []
    scales = {
        "log_mz": max(float(np.log(frame["mz"]).std()), 1e-6),
        "rt_sec": max(float(frame["rt_sec"].std()), 1e-6),
        "global_prevalence": max(float(frame["global_prevalence"].std()), 1e-6),
    }
    pool = pool.assign(log_mz=np.log(pool["mz"]))
    for target in selected.sort_values(["ttest_q", "max_exact_p"]).itertuples(index=False):
        available = pool[
            (~pool["feature_id"].astype(int).isin(used_features))
            & (~pool["ion_family_id"].astype(int).isin(used_families))
        ].copy()
        if available.empty:
            break
        distance = (
            ((available["log_mz"] - np.log(float(target.mz))) / scales["log_mz"]) ** 2
            + ((available["rt_sec"] - float(target.rt_sec)) / scales["rt_sec"]) ** 2
            + ((available["global_prevalence"] - float(target.global_prevalence)) / scales["global_prevalence"]) ** 2
        )
        index = distance.idxmin()
        row = available.loc[index].drop(labels=["log_mz"]).copy()
        row["matched_to_feature_id"] = int(target.feature_id)
        row["match_distance"] = float(distance.loc[index])
        rows.append(row)
        used_features.add(int(row["feature_id"]))
        used_families.add(int(row["ion_family_id"]))
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    annotated = pd.read_csv(ANNOTATED_PRIORITY)
    reports = {}
    all_selection = []
    for panel in PANELS:
        audit_path = AUDIT / f"{panel}__full_feature_audit.csv.gz"
        frame = pd.read_csv(audit_path)
        if panel == "pos_rp":
            discovery = frame[
                frame["effect_and_direction_gate"].fillna(False)
                & (frame["ttest_q"] <= 0.10)
            ].copy()
        else:
            # Negative mode has no full-space parametric FDR candidate; retain
            # all effect/direction candidates as an explicitly exploratory arm.
            discovery = frame[frame["effect_and_direction_gate"].fillna(False)].copy()
        discovery["target_role"] = np.where(
            discovery["ttest_q"] <= 0.10,
            "full_space_ttest_fdr10_sensitivity",
            "negative_mode_effect_direction_exploratory",
        )
        prior_ids = set(
            annotated.loc[annotated["panel"].eq(panel), "feature_id"].astype(int)
        )
        prior = frame[
            frame["feature_id"].astype(int).isin(prior_ids)
            & (~frame["feature_id"].astype(int).isin(discovery["feature_id"].astype(int)))
        ].copy()
        prior["target_role"] = "prior_phenotype_blind_annotated_screen"
        selected = pd.concat([discovery, prior], ignore_index=True).drop_duplicates("feature_id")
        controls = matched_controls(frame, selected.drop_duplicates("ion_family_id"))
        if len(controls):
            controls["target_role"] = "matched_null_control"
        selection = pd.concat([selected, controls], ignore_index=True).drop_duplicates("feature_id")
        selection["panel"] = panel
        ordered_columns = ["panel"] + [column for column in selection.columns if column != "panel"]
        selection = selection[ordered_columns]
        selection.to_csv(OUT / f"{panel}__selection_ledger.csv", index=False)

        source_targets = pd.read_csv(CONSENSUS / f"{panel}__requantification_targets.csv.gz")
        target_columns = list(source_targets.columns)
        targets = source_targets[
            source_targets["feature_id"].astype(int).isin(selection["feature_id"].astype(int))
        ].merge(
            selection[[
                "feature_id", "target_role", "ion_family_id", "ion_family_size",
                "raw_mean_log2fc", "pqn_mean_log2fc", "max_exact_p", "exact_q",
                "max_ttest_p", "ttest_q", "best_name", "annotation_evidence_tier",
                "matched_to_feature_id", "match_distance",
            ]],
            on="feature_id",
            how="left",
            validate="one_to_one",
        )
        if len(targets) != len(selection):
            raise RuntimeError(f"{panel}: selection/target mismatch {len(selection)} != {len(targets)}")
        targets.to_csv(OUT / f"{panel}__requantification_targets.csv.gz", index=False, compression="gzip")
        samples = pd.read_csv(CONSENSUS / f"{panel}__samples.csv")
        samples.to_csv(OUT / f"{panel}__samples.csv", index=False)
        all_selection.append(selection)
        reports[panel] = {
            "targets": int(len(targets)),
            "discovery_targets": int(len(discovery)),
            "prior_annotated_additions": int(len(prior)),
            "matched_null_controls": int(len(controls)),
            "unique_ion_families": int(selection["ion_family_id"].nunique()),
            "chromatography_edge_targets": int(((selection["rt_sec"] < 60.0) | (selection["rt_sec"] > 750.0)).sum()),
            "samples": int(len(samples)),
            "targets_sha256": sha256(OUT / f"{panel}__requantification_targets.csv.gz"),
            "source_target_columns_preserved": bool(all(column in targets.columns for column in target_columns)),
        }
    ledger = pd.concat(all_selection, ignore_index=True)
    ledger.to_csv(OUT / "all_selection_ledger.csv", index=False)
    report = {
        "status": "mtbls13729_full_space_eic_panel_frozen",
        "formal": False,
        "panels": reports,
        "parameters": {
            "ppm": 5.0,
            "rt_half_window_sec": 20.0,
            "resolve_local_peaks": True,
            "max_apex_delta_sec": 12.0,
        },
        "selection_contract": (
            "All full-space t-test-FDR10 sensitivity candidates are retained before EIC outcomes; "
            "negative mode is exploratory because no feature passed full-space FDR. Matched null controls "
            "are selected without phenotype-effect outcomes beyond exclusion from the effect gate."
        ),
        "claim_limit": (
            "Same-cohort targeted EIC is technical re-extraction. It cannot independently replicate abundance, "
            "confirm identity, or establish subtype specificity, flux, enzyme activity, or causality."
        ),
        "provenance": {
            "audit_report_sha256": sha256(AUDIT / "report.json"),
            "annotated_priority_sha256": sha256(ANNOTATED_PRIORITY),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
