#!/usr/bin/env python
"""Matched ChemProp2 sensitivity for library-supported versus unresolved features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from analyze_chemprop2_response_reproducibility_m1 import (
    cluster_bootstrap_delta,
    feature_qc_mask,
    median_center,
    nonempty_mgf_scans,
    parse_samples,
    response_profiles,
    retrieval,
    robust_log2,
    sha256,
    summarize,
)


def mgf_peak_counts(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    scan: int | None = None
    peaks = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                scan, peaks = None, 0
            elif line.startswith("SCANS="):
                scan = int(line.split("=", 1)[1])
            elif line == "END IONS":
                if scan is not None:
                    result[scan] = peaks
            elif line and "=" not in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        float(fields[0]); float(fields[1])
                    except ValueError:
                        continue
                    peaks += 1
    return result


def robust_z(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(float)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return np.nan_to_num((values - center) / scale, nan=0.0, posinf=10.0, neginf=-10.0)


def optimal_match(
    covariates: pd.DataFrame, supported: np.ndarray, unresolved: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    standardized = robust_z(covariates)
    left = np.flatnonzero(supported)
    right = np.flatnonzero(unresolved)
    # Squared Euclidean distance in robustly standardized covariates.
    distances = np.sum((standardized[left, None, :] - standardized[None, right, :]) ** 2, axis=2)
    left_pos, right_pos = linear_sum_assignment(distances)
    chosen = right[right_pos]
    selected = np.zeros(len(covariates), dtype=bool)
    selected[chosen] = True
    pairs = pd.DataFrame({
        "supported_index": left[left_pos],
        "unresolved_index": chosen,
        "distance": distances[left_pos, right_pos],
    })
    return selected, pairs


def balance(covariates: pd.DataFrame, left: np.ndarray, right: np.ndarray) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for column in covariates:
        a = covariates.loc[left, column].to_numpy(float)
        b = covariates.loc[right, column].to_numpy(float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        output[column] = {
            "supported_mean": float(np.mean(a)),
            "matched_unresolved_mean": float(np.mean(b)),
            "standardized_mean_difference": float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else 0.0,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/reference/chemprop2_m1_20261011"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/chemprop2_matched_unresolved_20261011"),
    )
    parser.add_argument("--library-score", type=float, default=0.80)
    parser.add_argument("--blank-ratio", type=float, default=3.0)
    parser.add_argument("--qc-cv-max", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")

    paths = {
        "quantification": args.source_dir / "quantification.csv",
        "mgf": args.source_dir / "consensus.mgf",
        "library": args.source_dir / "library_matches.tsv",
    }
    quant = pd.read_csv(paths["quantification"])
    samples = parse_samples(list(quant.columns))
    scan_ids = nonempty_mgf_scans(paths["mgf"])
    mask, audit = feature_qc_mask(quant, samples, scan_ids, args.blank_ratio, args.qc_cv_max)
    libraries = pd.read_csv(paths["library"], sep="\t")
    libraries["#Scan#"] = pd.to_numeric(libraries["#Scan#"], errors="raise").astype(int)
    libraries["MQScore"] = pd.to_numeric(libraries["MQScore"], errors="raise")
    supported_ids = set(libraries.loc[libraries["MQScore"].ge(args.library_score), "#Scan#"])
    supported_global = quant["row ID"].astype(int).isin(supported_ids).to_numpy() & mask
    unresolved_global = mask & ~supported_global

    study_values = quant[samples["column"].tolist()].to_numpy(float).T
    profiles, meta = response_profiles(
        median_center(robust_log2(study_values)), samples, mask, "trajectory_auc",
    )
    peak_counts = mgf_peak_counts(paths["mgf"])
    local_audit = audit.loc[mask].reset_index(drop=True)
    local_values = study_values[:, mask]
    quality_covariates = pd.DataFrame({
        "log_study_q95": np.log1p(local_audit["study_q95"].to_numpy(float)),
        "qc_detection_rate": local_audit["qc_detection_rate"].to_numpy(float),
        "log_qc_cv": np.log1p(np.clip(local_audit["qc_cv"].to_numpy(float), 0, 10)),
        "study_detection_rate": np.mean(local_values > 0, axis=0),
        "log_ms2_peak_count": np.log1p([peak_counts.get(int(row), 0) for row in local_audit["row_id"]]),
    })
    response_magnitude = np.nanmedian(np.abs(profiles), axis=0)
    full_covariates = quality_covariates.assign(log_response_magnitude=np.log1p(response_magnitude))
    supported = supported_global[mask]
    unresolved = unresolved_global[mask]

    matched_quality, pairs_quality = optimal_match(quality_covariates, supported, unresolved)
    matched_full, pairs_full = optimal_match(full_covariates, supported, unresolved)
    masks = {
        "library_supported": supported,
        "unresolved_all": unresolved,
        "unresolved_quality_matched": matched_quality,
        "unresolved_quality_response_matched": matched_full,
    }
    frames: dict[str, pd.DataFrame] = {name: retrieval(profiles, meta, selected) for name, selected in masks.items()}
    summaries = {name: {"features": int(masks[name].sum()), **summarize(frame)} for name, frame in frames.items()}
    deltas = {
        name: {
            "recall1": cluster_bootstrap_delta(frame, frames["library_supported"], "recall1", args.seed + i),
            "mrr": cluster_bootstrap_delta(frame, frames["library_supported"], "reciprocal_rank", args.seed + 100 + i),
        }
        for i, (name, frame) in enumerate(frames.items()) if name.startswith("unresolved_")
    }

    args.output_dir.mkdir(parents=True)
    pairs_quality.to_csv(args.output_dir / "quality_match_pairs.csv", index=False)
    pairs_full.to_csv(args.output_dir / "quality_response_match_pairs.csv", index=False)
    tagged = []
    for name, frame in frames.items():
        item = frame.copy(); item.insert(0, "universe", name); tagged.append(item)
    pd.concat(tagged, ignore_index=True).to_csv(args.output_dir / "retrieval_results.csv", index=False)
    report = {
        "status": "chemprop2_matched_unresolved_sensitivity_complete",
        "formal": False,
        "evidence_role": "development_consumed_sensitivity",
        "dataset": {"accession": "MSV000094899", "queries": len(meta), "drugs": int(meta["drug"].nunique())},
        "information_used_for_retrieval": "MS1 abundance response on natural feature IDs",
        "matching": {
            "quality_covariates": list(quality_covariates.columns),
            "response_matched_covariates": list(full_covariates.columns),
            "response_matching_warning": "conservative sensitivity; matching on response magnitude can control away biological signal",
            "quality_balance": balance(quality_covariates, supported, matched_quality),
            "quality_response_balance": balance(full_covariates, supported, matched_full),
        },
        "retrieval": summaries,
        "deltas_vs_library_supported": deltas,
        "source_hashes": {name: sha256(path) for name, path in paths.items()},
        "does_not_establish": [
            "MS/MS or embedding increment",
            "cross-study generalization",
            "authentic-standard identity of the supported class",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"retrieval": summaries, "deltas": deltas}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
