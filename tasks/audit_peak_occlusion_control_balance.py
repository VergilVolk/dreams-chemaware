"""Audit whether targeted and random peak deletions are physically balanced."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pilot_paired_layer_cka import preprocess_spectrum


def parse_mz(text: object) -> np.ndarray:
    if pd.isna(text) or not str(text):
        return np.empty(0)
    return np.asarray([float(value) for value in str(text).split(";") if value], float)


def resolve(clean: np.ndarray, values: np.ndarray, tolerance: float = 0.005) -> np.ndarray:
    indices, used = [], set()
    for value in values:
        candidates = [
            i for i in range(1, len(clean))
            if i not in used and clean[i, 0] > 0 and abs(float(clean[i, 0]) - value) <= tolerance
        ]
        if candidates:
            index = min(candidates, key=lambda i: abs(float(clean[i, 0]) - value))
            indices.append(index)
            used.add(index)
    return np.asarray(indices, int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("data/validation/large_all_failure_targeted_peak_occlusion"))
    parser.add_argument("--localization-dir", type=Path, default=Path("data/validation/large_all_failure_peak_localization"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    args = parser.parse_args()

    results = pd.read_csv(args.results_dir / "perturbation_results.csv")
    mappings = []
    for split in results["split"].unique():
        evidence = pd.read_csv(args.localization_dir / f"{split}_peak_evidence.csv")
        mappings.append(evidence[["split", "query_index", "query_hdf5_row", "query_precursor_mz"]])
    mapping = pd.concat(mappings).drop_duplicates(["split", "query_index"])
    results = results.merge(mapping, on=["split", "query_index"], how="left", validate="many_to_one")

    cache = {}
    rows = []
    with h5py.File(args.data, "r") as handle:
        for row in results.itertuples(index=False):
            key = (row.split, int(row.query_index))
            if key not in cache:
                cache[key] = preprocess_spectrum(
                    np.asarray(handle["spectrum"][int(row.query_hdf5_row)]),
                    float(row.query_precursor_mz), args.n_highest_peaks,
                ).numpy()
            clean = cache[key]
            selected = resolve(clean, parse_mz(row.removed_mz))
            intensity = clean[selected, 1] if len(selected) else np.empty(0)
            mz = clean[selected, 0] if len(selected) else np.empty(0)
            rows.append({
                "split": row.split, "query_index": int(row.query_index), "formula": row.formula,
                "evidence": row.evidence, "condition": row.condition, "repeat": int(row.repeat),
                "resolved_count": len(selected),
                "sum_relative_intensity": float(intensity.sum()),
                "mean_relative_intensity": float(intensity.mean()) if len(intensity) else np.nan,
                "mean_mz": float(mz.mean()) if len(mz) else np.nan,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.results_dir / "control_balance_rows.csv", index=False)
    random = frame.loc[frame["condition"] == "matched_random"].groupby(
        ["split", "query_index", "formula", "evidence"], as_index=False
    ).agg(
        random_count=("resolved_count", "mean"),
        random_sum_intensity=("sum_relative_intensity", "mean"),
        random_mean_intensity=("mean_relative_intensity", "mean"),
        random_mean_mz=("mean_mz", "mean"),
    )
    targeted = frame.loc[frame["condition"] == "targeted"].rename(columns={
        "resolved_count": "target_count", "sum_relative_intensity": "target_sum_intensity",
        "mean_relative_intensity": "target_mean_intensity", "mean_mz": "target_mean_mz",
    })
    paired = targeted.merge(random, on=["split", "query_index", "formula", "evidence"], validate="one_to_one")
    paired["count_difference"] = paired["target_count"] - paired["random_count"]
    paired["sum_intensity_difference"] = paired["target_sum_intensity"] - paired["random_sum_intensity"]
    paired["mean_intensity_difference"] = paired["target_mean_intensity"] - paired["random_mean_intensity"]
    paired["mean_mz_difference"] = paired["target_mean_mz"] - paired["random_mean_mz"]
    paired.to_csv(args.results_dir / "control_balance_paired.csv", index=False)
    report = {}
    for evidence, group in paired.groupby("evidence"):
        report[evidence] = {
            "queries": len(group),
            "mean_count_difference": float(group["count_difference"].mean()),
            "mean_sum_intensity_difference": float(group["sum_intensity_difference"].mean()),
            "median_absolute_sum_intensity_difference": float(group["sum_intensity_difference"].abs().median()),
            "mean_mean_intensity_difference": float(group["mean_intensity_difference"].mean()),
            "mean_mz_difference": float(group["mean_mz_difference"].mean()),
            "median_absolute_mean_mz_difference": float(group["mean_mz_difference"].abs().median()),
        }
    (args.results_dir / "control_balance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
