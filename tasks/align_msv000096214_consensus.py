#!/usr/bin/env python
"""Consensus alignment and EIC gap filling for MSV000096214 (contract 3.4).

Per polarity, genotype-blind:
  1. RT reference = pooled-QC file with the median feature count.
  2. Each other file: Theil-Sen RT correction from m/z-matched seeds
     (5 ppm, +/-60 s initial window).
  3. Greedy consensus grouping by descending intensity: an unused feature
     anchors a group; every file contributes its closest unused feature
     within 5 ppm m/z and 10 s warped RT.
  4. Gap filling: for group/file cells without a feature, integrate the raw
     EIC (+/-5 ppm, group median RT +/-15 s). Cells whose apex is below the
     file's MS1-point Q90 (from the extraction summaries) stay missing;
     missing is not zero.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyopenms as oms

PPM = 5.0
RT_WINDOW_SEED = 60.0
RT_WINDOW_GROUP = 10.0
RT_WINDOW_GAP = 15.0


def load_file_frames(features_dir: Path, polarity: str, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for item in manifest.loc[manifest["polarity"] == polarity].itertuples(index=False):
        path = features_dir / "features" / f"{polarity}__{Path(item.filename).stem}__features.csv.gz"
        frame = pd.read_csv(path)
        frame["file"] = Path(item.filename).stem
        frame["role"] = item.role
        frames[Path(item.filename).stem] = frame
    return frames


def theilsen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return float(np.median(y)) if n else 0.0, 1.0
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    dy = y[j] - y[i]
    valid = np.abs(dx) > 1e-9
    if not valid.any():
        return float(np.median(y)), 1.0
    slope = float(np.median(dy[valid] / dx[valid]))
    intercept = float(np.median(y - slope * x))
    return intercept, slope


def match_seed_deltas(reference: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """m/z-matched (ref_rt - other_rt) within PPM and seed window."""
    ref_mz = reference["mz"].to_numpy()
    order = np.argsort(ref_mz)
    ref_mz_sorted = ref_mz[order]
    ref_rt_sorted = reference["rt_sec"].to_numpy()[order]
    low = np.searchsorted(ref_mz_sorted, other["mz"].to_numpy() * (1 - PPM * 1e-6))
    high = np.searchsorted(ref_mz_sorted, other["mz"].to_numpy() * (1 + PPM * 1e-6))
    other_rt = other["rt_sec"].to_numpy()
    deltas, n_seeds = [], 0
    for lo, hi, rt in zip(low, high, other_rt):
        if hi <= lo:
            continue
        shifts = ref_rt_sorted[lo:hi] - rt
        best = int(np.argmin(np.abs(shifts)))
        if abs(shifts[best]) <= RT_WINDOW_SEED:
            deltas.append(float(shifts[best]))
            n_seeds += 1
    return np.asarray(deltas), np.asarray([n_seeds])


def fit_warp(reference: pd.DataFrame, other: pd.DataFrame) -> tuple[float, float, int]:
    """Return (intercept, slope) so that rt*ref ~ other.warped matches reference.

    Model: other_rt + delta ≈ ref_rt with delta = a + b*other_rt.
    """
    deltas, _ = match_seed_deltas(reference, other)
    other_mz = other["mz"].to_numpy()
    # recompute matched other-side rt for the fit
    ref_mz = reference["mz"].to_numpy()
    order = np.argsort(ref_mz)
    ref_mz_sorted = ref_mz[order]
    ref_rt_sorted = reference["rt_sec"].to_numpy()[order]
    other_rt = other["rt_sec"].to_numpy()
    low = np.searchsorted(ref_mz_sorted, other_mz * (1 - PPM * 1e-6))
    high = np.searchsorted(ref_mz_sorted, other_mz * (1 + PPM * 1e-6))
    xs, ys = [], []
    for lo, hi, rt in zip(low, high, other_rt):
        if hi <= lo:
            continue
        shifts = ref_rt_sorted[lo:hi] - rt
        best = int(np.argmin(np.abs(shifts)))
        if abs(shifts[best]) <= RT_WINDOW_SEED:
            xs.append(rt)
            ys.append(float(shifts[best]))
    if len(xs) < 10:
        return 0.0, 1.0, len(xs)
    intercept, slope = theilsen(np.asarray(xs), np.asarray(ys))
    return intercept, slope, len(xs)


def greedy_groups(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Anchor-by-intensity greedy grouping, one feature per file per group."""
    all_mz = np.concatenate([f["mz"].to_numpy() for f in frames.values()])
    all_rt = np.concatenate([f["rt_warped"].to_numpy() for f in frames.values()])
    all_intensity = np.concatenate([f["intensity"].to_numpy() for f in frames.values()])
    file_index = np.concatenate([
        np.full(len(f), i, dtype=np.int64) for i, f in enumerate(frames.values())])
    row_index = np.concatenate([
        np.arange(len(f), dtype=np.int64) for f in frames.values()])
    file_names = list(frames.keys())

    order = np.argsort(-all_intensity)
    used = np.zeros(len(all_mz), dtype=bool)
    mz_order = np.argsort(all_mz)
    mz_sorted = all_mz[mz_order]

    groups: list[dict[str, Any]] = []
    for anchor in order:
        if used[anchor]:
            continue
        mz0 = all_mz[anchor]
        lo = np.searchsorted(mz_sorted, mz0 * (1 - PPM * 1e-6) - 1e-9)
        hi = np.searchsorted(mz_sorted, mz0 * (1 + PPM * 1e-6) + 1e-9)
        window = mz_order[lo:hi]
        window = window[~used[window]]
        if len(window) == 0:
            continue
        rt0 = all_rt[anchor]
        keep = np.abs(all_rt[window] - rt0) <= RT_WINDOW_GROUP
        window = window[keep]
        if len(window) == 0:
            used[anchor] = True
            continue
        # closest warped RT per file; the anchor file contributes the anchor itself.
        # members maps file_index -> (global entry index, per-frame row, warped rt)
        members = {int(file_index[anchor]): (int(anchor), int(row_index[anchor]), float(all_rt[anchor]))}
        for candidate in window:
            fi = int(file_index[candidate])
            d = abs(all_rt[candidate] - rt0)
            if fi not in members or d < abs(members[fi][2] - rt0):
                members[fi] = (int(candidate), int(row_index[candidate]), float(all_rt[candidate]))
        chosen = [entry for entry, _, _ in members.values()]
        used[chosen] = True
        groups.append({
            "group_id": len(groups),
            "mz": float(np.median(all_mz[chosen])),
            "rt_sec": float(np.median(all_rt[chosen])),
            "n_files": len(chosen),
            "files": {file_names[fi]: frames[file_names[fi]].iloc[row].to_dict()
                      for fi, (_, row, _) in members.items()},
        })
    return groups


def load_ms1_once(path: Path) -> oms.MSExperiment:
    experiment = oms.MSExperiment()
    loader = oms.MzXMLFile()
    try:
        options = loader.getOptions()
        options.setMSLevels([1])
        loader.setOptions(options)
    except AttributeError:
        pass
    loader.load(str(path), experiment)
    experiment.sortSpectra(True)
    return experiment


def gap_fill_eic(experiment: oms.MSExperiment, mz: float, rt: float,
                 floor: float) -> tuple[float, float]:
    """Apex intensity and apex RT inside the EIC window; (nan, nan) if absent."""
    best_intensity, best_rt = -1.0, math.nan
    for spectrum in experiment:
        rt_s = float(spectrum.getRT())
        if abs(rt_s - rt) > RT_WINDOW_GAP:
            continue
        mzs, intensities = spectrum.get_peaks()
        mzs = np.asarray(mzs, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float64)
        lo = np.searchsorted(mzs, mz * (1 - PPM * 1e-6))
        hi = np.searchsorted(mzs, mz * (1 + PPM * 1e-6))
        if hi <= lo:
            continue
        local = intensities[lo:hi]
        peak = float(local.max())
        if peak > best_intensity:
            best_intensity = peak
            best_rt = rt_s
    if best_intensity < 0 or best_intensity < floor:
        return math.nan, math.nan
    return best_intensity, best_rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/external/msv000096214/ms1_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/msv000096214/consensus"))
    parser.add_argument("--max-fill-files", type=int, default=200)
    args = parser.parse_args()

    manifest = pd.read_csv(args.root / "download_manifest.csv")
    manifest = manifest.loc[manifest["ok"]]
    extraction = pd.read_csv(args.features_dir / "extraction_manifest.csv")
    summaries = {row["sample"]: json.loads(
        (args.features_dir / "features" / f"{row['polarity']}__{row['sample']}__summary.json"
         ).read_text(encoding="utf-8")) for row in extraction.itertuples(index=False)}

    for polarity in ("positive", "negative"):
        out_dir = args.output_dir / polarity
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        frames = load_file_frames(args.features_dir, polarity, manifest)
        qc_names = [name for name, frame in frames.items() if str(frame["role"].iloc[0]) == "pooled_qc"]
        qc_counts = {name: len(frames[name]) for name in qc_names}
        reference_name = sorted(qc_names, key=lambda n: qc_counts[n])[len(qc_names) // 2]
        reference = frames[reference_name]

        warps: dict[str, dict[str, float]] = {}
        for name, frame in frames.items():
            if name == reference_name:
                frame["rt_warped"] = frame["rt_sec"]
                warps[name] = {"intercept": 0.0, "slope": 1.0, "n_seeds": len(frame)}
                continue
            intercept, slope, n_seeds = fit_warp(reference, frame)
            frame["rt_warped"] = frame["rt_sec"] + intercept + slope * frame["rt_sec"]
            warps[name] = {"intercept": intercept, "slope": slope, "n_seeds": n_seeds}

        groups = greedy_groups(frames)
        print(f"[{polarity}] groups={len(groups)} after {time.time()-started:.0f}s", flush=True)

        sample_names = sorted(frames.keys())
        intensity_rows, detected_rows = [], []
        for group in groups:
            intensity_rows.append({
                "group_id": group["group_id"], "mz": group["mz"], "rt_sec": group["rt_sec"]})
            detected_rows.append({
                "group_id": group["group_id"],
                **{name: (group["files"][name]["intensity"] if name in group["files"] else np.nan)
                   for name in sample_names}})
        intensity_frame = pd.DataFrame(intensity_rows)
        matrix = pd.DataFrame(detected_rows).set_index("group_id")
        intensity_frame.to_csv(out_dir / "groups.csv", index=False)
        matrix.to_csv(out_dir / "matrix_detected.csv.gz", compression="gzip")

        # gap filling against raw mzXML for cells missing in the detected matrix
        missing_mask = matrix.isna()
        fill_counts = {}
        for name in sample_names:
            missing_groups = matrix.index[missing_mask[name]]
            if len(missing_groups) == 0:
                fill_counts[name] = 0
                continue
            source = args.root / "mzxml" / polarity / f"{name}.mzXML"
            experiment = load_ms1_once(source)
            floor = float(summaries[name].get("intensity_q90") or 0.0)
            warp = warps[name]
            denominator = 1.0 + warp["slope"]
            filled = 0
            group_lookup = intensity_frame.set_index("group_id")
            for gid in missing_groups:
                mz = float(group_lookup.loc[gid, "mz"])
                rt_warped = float(group_lookup.loc[gid, "rt_sec"])
                rt_file = (rt_warped - warp["intercept"]) / denominator
                intensity, _ = gap_fill_eic(experiment, mz, rt_file, floor)
                if not np.isnan(intensity):
                    matrix.loc[gid, name] = intensity
                    filled += 1
            fill_counts[name] = filled
            matrix.to_csv(out_dir / "matrix_filled.partial.csv.gz", compression="gzip")
            print(f"[{polarity}] gap-filled {name}: {filled}/{len(missing_groups)}", flush=True)
        matrix.to_csv(out_dir / "matrix_filled.csv.gz", compression="gzip")

        report = {
            "status": "MSV000096214_CONSENSUS_COMPLETE",
            "polarity": polarity,
            "data_role": "development/consumed",
            "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
            "reference_file": reference_name,
            "qc_feature_counts": qc_counts,
            "warp_params": warps,
            "n_groups": len(groups),
            "n_samples": len(sample_names),
            "gap_fill_counts": fill_counts,
            "genotype_column_read": False,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in
                          ("polarity", "reference_file", "n_groups", "n_samples")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
