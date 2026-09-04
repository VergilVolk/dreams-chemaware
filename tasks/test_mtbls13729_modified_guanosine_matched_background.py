#!/usr/bin/env python
"""Phenotype-blind matched-background test for the modified-guanosine module.

The target module contains two neutral families, each represented by an H/Na
adduct pair: 1597/7489 and 3019/8481.  Random panels match each of the four
technical features on m/z, RT and global prevalence before any phenotype is
read.  After the random targets have been requantified from mzML with exactly
the same EIC protocol, the analysis collapses each random pair and then the two
families exactly as for the observed module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_IDS = (1597, 7489, 3019, 8481)
EXCLUDED_IDS = {4966, 3019, 1597, 7489, 1717, 3222, 3180, 16425, 8481}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pools(consensus: pd.DataFrame) -> dict[int, tuple[np.ndarray, str]]:
    background = consensus.loc[
        consensus.keep_for_requantification.astype(bool) & ~consensus.index.isin(EXCLUDED_IDS)
    ].copy()
    pools: dict[int, tuple[np.ndarray, str]] = {}
    windows = (
        (25.0, 45.0, 0.15, "strict"),
        (50.0, 90.0, 0.25, "expanded"),
        (100.0, 150.0, 0.35, "fallback"),
    )
    for feature_id in TARGET_IDS:
        target = consensus.loc[feature_id]
        chosen = pd.DataFrame()
        level = ""
        for mz_window, rt_window, prevalence_window, label in windows:
            chosen = background.loc[
                background.mz.sub(target.mz).abs().le(mz_window)
                & background.rt_sec.sub(target.rt_sec).abs().le(rt_window)
                & background.global_prevalence.sub(target.global_prevalence).abs().le(prevalence_window)
            ]
            # Reuse across null panels is allowed; the null unit is a panel,
            # not a unique feature.  Prefer the tight window whenever it has
            # at least ten alternatives rather than diluting matching merely
            # to inflate the distinct-target count.
            if len(chosen) >= 10:
                level = label
                break
        if len(chosen) < 10:
            raise RuntimeError(f"insufficient phenotype-blind pool for target {feature_id}: {len(chosen)}")
        pools[feature_id] = (chosen.index.to_numpy(int), level)
    return pools


def prepare(args: argparse.Namespace) -> None:
    consensus = pd.read_csv(args.consensus).set_index("feature_id")
    if not set(TARGET_IDS).issubset(consensus.index):
        raise RuntimeError("consensus metadata is missing a target ion-family member")
    pools = build_pools(consensus)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for panel_id in range(args.permutations):
        used: set[int] = set()
        for slot, target_id in enumerate(TARGET_IDS):
            options, level = pools[target_id]
            available = options[~np.isin(options, np.fromiter(used, dtype=int))]
            if not len(available):
                raise RuntimeError(f"no unique match remained for panel {panel_id}, target {target_id}")
            random_id = int(rng.choice(available))
            used.add(random_id)
            rows.append({
                "panel_id": panel_id,
                "slot": slot,
                "target_feature_id": target_id,
                "random_feature_id": random_id,
                "match_level": level,
            })
    mapping = pd.DataFrame(rows)
    out = args.output_dir
    target_out = args.target_dir
    out.mkdir(parents=True, exist_ok=True)
    target_out.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out / "matched_panel_members.csv.gz", index=False)
    unique_ids = sorted(mapping.random_feature_id.unique())
    targets = consensus.loc[unique_ids].reset_index()
    targets.to_csv(target_out / "pos_rp__requantification_targets.csv.gz", index=False)
    sample_source = args.consensus.parent / "pos_rp__samples.csv"
    if not sample_source.exists():
        raise FileNotFoundError(sample_source)
    shutil.copyfile(sample_source, target_out / "pos_rp__samples.csv")
    report = {
        "status": "mtbls13729_modified_guanosine_matched_background_prepared",
        "formal": True,
        "panels": int(args.permutations),
        "random_targets": int(len(unique_ids)),
        "target_slots": list(TARGET_IDS),
        "matching": "m/z, RT and global prevalence only; phenotype-blind",
        "match_levels": mapping.match_level.value_counts().to_dict(),
        "provenance": {
            "consensus_sha256": sha256_file(args.consensus),
            "mapping_sha256": sha256_file(out / "matched_panel_members.csv.gz"),
            "targets_sha256": sha256_file(target_out / "pos_rp__requantification_targets.csv.gz"),
            "samples_sha256": sha256_file(target_out / "pos_rp__samples.csv"),
        },
    }
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def family_module_effect(
    panel: pd.DataFrame,
    log_auc: pd.DataFrame,
    detected: pd.DataFrame,
    factors: pd.Series,
) -> np.ndarray:
    feature_ids = panel.sort_values("slot").random_feature_id.astype(int).tolist()
    if len(feature_ids) != 4:
        raise RuntimeError("each random panel must contain exactly four slots")
    values: list[float] = []
    matrix = log_auc.sub(factors, axis=1)
    for patient_number in range(21, 31):
        patient = f"P{patient_number:02d}"
        tumour = f"{patient}-Rmu"
        normal = f"{patient}-RN"
        feature_delta: list[float] = []
        for feature_id in feature_ids:
            if bool(detected.loc[feature_id, tumour]) and bool(detected.loc[feature_id, normal]):
                feature_delta.append(float(matrix.loc[feature_id, tumour] - matrix.loc[feature_id, normal]))
            else:
                feature_delta.append(np.nan)
        family_a = np.nanmedian(feature_delta[0:2]) if np.isfinite(feature_delta[0:2]).any() else np.nan
        family_b = np.nanmedian(feature_delta[2:4]) if np.isfinite(feature_delta[2:4]).any() else np.nan
        values.append(float(np.mean([family_a, family_b])) if np.isfinite(family_a) and np.isfinite(family_b) else np.nan)
    return np.asarray(values, dtype=float)


def analyze(args: argparse.Namespace) -> None:
    mapping_path = args.output_dir / "matched_panel_members.csv.gz"
    auc_path = args.eic_dir / "pos_rp__eic_auc_matrix.csv.gz"
    detection_path = args.eic_dir / "pos_rp__eic_detection_matrix.csv.gz"
    for path in (mapping_path, auc_path, detection_path, args.observed_effects, args.normalization_factors):
        if not path.exists():
            raise FileNotFoundError(path)
    mapping = pd.read_csv(mapping_path)
    auc = pd.read_csv(auc_path).set_index("feature_id").astype(float)
    detected = pd.read_csv(detection_path).set_index("feature_id").astype(bool)
    required = set(mapping.random_feature_id.astype(int))
    if not required.issubset(auc.index) or list(auc.columns) != list(detected.columns):
        raise RuntimeError("requantified EIC cache does not match the frozen random-target set")
    log_auc = np.log2(auc.where((auc > 0) & detected))

    factor_table = pd.read_csv(args.normalization_factors)
    factors = {"raw": pd.Series(0.0, index=log_auc.columns)}
    for name, group in factor_table.groupby("normalization"):
        series = group.set_index("sample").log2_factor.reindex(log_auc.columns)
        if series.isna().any():
            raise RuntimeError(f"normalization factor {name} does not cover the EIC sample set")
        factors[str(name)] = series.astype(float)

    observed = pd.read_csv(args.observed_effects)
    observed = observed.loc[observed.cohort.eq("Rmu")]
    observed_by_norm = {
        name: group.set_index("patient").module_log2fc.dropna().to_numpy(float)
        for name, group in observed.groupby("normalization")
    }
    rows: list[dict[str, object]] = []
    for normalization, factor in factors.items():
        if normalization not in observed_by_norm:
            raise RuntimeError(f"observed module is missing normalization {normalization}")
        for panel_id, panel in mapping.groupby("panel_id", sort=True):
            values = family_module_effect(panel, log_auc, detected, factor)
            finite = values[np.isfinite(values)]
            rows.append({
                "normalization": normalization,
                "panel_id": int(panel_id),
                "n_pairs": int(len(finite)),
                "mean_log2fc": float(finite.mean()) if len(finite) else np.nan,
                "positive_fraction": float(np.mean(finite > 0)) if len(finite) else np.nan,
                "all_positive": bool(len(finite) == 10 and np.all(finite > 0)),
            })
    null = pd.DataFrame(rows)
    null.to_csv(args.output_dir / "matched_background_null.csv.gz", index=False)

    reports: list[dict[str, object]] = []
    for normalization, observed_values in observed_by_norm.items():
        block = null.loc[(null.normalization == normalization) & (null.n_pairs == 10)].copy()
        observed_mean = float(observed_values.mean())
        observed_all_positive = bool(len(observed_values) == 10 and np.all(observed_values > 0))
        mean_p = float((1 + np.sum(block.mean_log2fc >= observed_mean)) / (1 + len(block)))
        joint_exceed = (block.mean_log2fc >= observed_mean) & block.all_positive
        joint_p = float((1 + joint_exceed.sum()) / (1 + len(block)))
        all_positive_p = float((1 + block.all_positive.sum()) / (1 + len(block)))
        reports.append({
            "normalization": normalization,
            "observed_n": int(len(observed_values)),
            "observed_mean_log2fc": observed_mean,
            "observed_all_positive": observed_all_positive,
            "evaluable_random_panels": int(len(block)),
            "matched_background_one_sided_p_mean": mean_p,
            "matched_background_p_all_positive": all_positive_p,
            "matched_background_joint_p_mean_and_all_positive": joint_p,
            "null_mean_p95": float(block.mean_log2fc.quantile(0.95)),
            "null_mean_p99": float(block.mean_log2fc.quantile(0.99)),
        })
    gate = all(
        row["evaluable_random_panels"] >= int(args.permutations * 0.75)
        and row["matched_background_one_sided_p_mean"] <= 0.05
        and row["matched_background_joint_p_mean_and_all_positive"] <= 0.05
        for row in reports
    )
    report = {
        "status": "mtbls13729_modified_guanosine_matched_background_complete",
        "formal": True,
        "protocol": "phenotype-blind m/z/RT/global-prevalence matching followed by identical targeted-EIC and ion-family collapse",
        "results": reports,
        "gates": {
            "at_least_75pct_panels_evaluable_every_normalization": bool(all(x["evaluable_random_panels"] >= int(args.permutations * 0.75) for x in reports)),
            "observed_mean_above_matched_background_p05_every_normalization": bool(all(x["matched_background_one_sided_p_mean"] <= 0.05 for x in reports)),
            "joint_mean_and_all_positive_p05_every_normalization": bool(all(x["matched_background_joint_p_mean_and_all_positive"] <= 0.05 for x in reports)),
            "pass": bool(gate),
        },
        "claim_limit": "The matched-background test supports module specificity among technically similar features; it does not confirm positional isomers, flux, enzymes, or external-cohort replication.",
        "provenance": {
            "mapping_sha256": sha256_file(mapping_path),
            "auc_sha256": sha256_file(auc_path),
            "detection_sha256": sha256_file(detection_path),
            "observed_effects_sha256": sha256_file(args.observed_effects),
        },
    }
    (args.output_dir / "matched_background_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "analyze"), required=True)
    parser.add_argument("--consensus", type=Path, default=Path("data/mtbls13729/ms1_consensus/pos_rp__consensus_metadata.csv.gz"))
    parser.add_argument("--target-dir", type=Path, default=Path("data/mtbls13729/modified_guanosine_matched_targets_v1"))
    parser.add_argument("--eic-dir", type=Path, default=Path("data/mtbls13729/modified_guanosine_matched_eic_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/modified_guanosine_matched_background_v1"))
    parser.add_argument("--observed-effects", type=Path, default=Path("data/mtbls13729/biology_closure_analysis_v1/fully_ion_family_collapsed_module_patient_effects.csv"))
    parser.add_argument("--normalization-factors", type=Path, default=Path("data/mtbls13729/biology_closure_analysis_v1/phenotype_blind_normalization_factors.csv"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
