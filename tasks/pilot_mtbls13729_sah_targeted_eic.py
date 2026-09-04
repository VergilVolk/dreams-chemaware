#!/usr/bin/env python
"""Phenotype-blind exact-mass pilot for an SAH-like feature in MTBLS13729 HILIC+."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyopenms as oms
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


SAMPLE_RE = re.compile(r"^(P\d{2})-(Ltu|Rtu|Rmu|LN|RN)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/pos_hilic")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/sah_exact_mass_pilot_v1"),
    )
    parser.add_argument("--mz", type=float, default=385.1289)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--cluster-gap-sec", type=float, default=12.0)
    parser.add_argument("--integration-half-window-sec", type=float, default=15.0)
    return parser.parse_args()


def load_trace(path: Path, target_mz: float, ppm: float) -> tuple[np.ndarray, np.ndarray]:
    experiment = oms.MSExperiment()
    loader = oms.MzMLFile()
    options = loader.getOptions()
    options.setMSLevels([1])
    loader.setOptions(options)
    loader.load(str(path), experiment)
    tolerance = target_mz * ppm * 1e-6
    times, signal = [], []
    for spectrum in experiment:
        mzs, intensities = spectrum.get_peaks()
        mzs = np.asarray(mzs, dtype=float)
        intensities = np.asarray(intensities, dtype=float)
        left = int(np.searchsorted(mzs, target_mz - tolerance, side="left"))
        right = int(np.searchsorted(mzs, target_mz + tolerance, side="right"))
        times.append(float(spectrum.getRT()))
        signal.append(float(np.max(intensities[left:right])) if right > left else 0.0)
    return np.asarray(times), np.asarray(signal)


def candidate_peaks(sample: str, times: np.ndarray, signal: np.ndarray) -> list[dict]:
    baseline = float(np.percentile(signal, 20))
    corrected = np.maximum(signal - baseline, 0)
    noise = max(1.4826 * float(np.median(np.abs(signal - np.median(signal)))), 1.0)
    smoothed = gaussian_filter1d(corrected, sigma=1.0, mode="nearest")
    prominence = max(5.0 * noise, 0.02 * float(smoothed.max()))
    peaks, properties = find_peaks(smoothed, prominence=prominence, distance=3)
    if len(peaks) > 5:
        keep = np.argsort(properties["prominences"])[-5:]
        peaks = peaks[keep]
        prominences = properties["prominences"][keep]
    else:
        prominences = properties["prominences"]
    return [
        {
            "sample": sample,
            "apex_rt_sec": float(times[index]),
            "apex_intensity": float(corrected[index]),
            "prominence": float(prom),
        }
        for index, prom in zip(peaks, prominences)
    ]


def cluster_peaks(peaks: pd.DataFrame, gap: float) -> pd.DataFrame:
    ordered = peaks.sort_values("apex_rt_sec").copy()
    cluster = []
    current = -1
    previous = None
    for value in ordered["apex_rt_sec"]:
        if previous is None or value - previous > gap:
            current += 1
        cluster.append(current)
        previous = value
    ordered["cluster"] = cluster
    return ordered


def paired_effect(frame: pd.DataFrame, tumor: str, normal: str) -> dict:
    t = frame.loc[frame["tissue"].eq(tumor), ["patient", "area"]].rename(
        columns={"area": "T"}
    )
    n = frame.loc[frame["tissue"].eq(normal), ["patient", "area"]].rename(
        columns={"area": "N"}
    )
    paired = t.merge(n, on="patient").loc[
        lambda item: (item["T"] > 0) & (item["N"] > 0)
    ]
    delta = np.log2(paired["T"].to_numpy() / paired["N"].to_numpy())
    if not len(delta):
        return {"n": 0}
    return {
        "n": int(len(delta)),
        "mean_log2fc": float(delta.mean()),
        "median_log2fc": float(np.median(delta)),
        "positive_pairs": int(np.sum(delta > 0)),
        "negative_pairs": int(np.sum(delta < 0)),
        "paired_t_p": float(stats.ttest_1samp(delta, 0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(delta).pvalue),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.mzml_dir.glob("*.mzML"))
    if len(paths) != 60:
        raise RuntimeError(f"expected 60 HILIC+ mzML files, got {len(paths)}")
    traces = {}
    peak_rows = []
    for index, path in enumerate(paths, 1):
        match = SAMPLE_RE.match(path.stem)
        if not match:
            raise RuntimeError(f"unexpected sample name: {path.stem}")
        times, signal = load_trace(path, args.mz, args.ppm)
        traces[path.stem] = (times, signal)
        peak_rows.extend(candidate_peaks(path.stem, times, signal))
        if index % 10 == 0:
            print(f"[SAH pilot] {index}/{len(paths)}", flush=True)
    peaks = cluster_peaks(pd.DataFrame(peak_rows), args.cluster_gap_sec)
    clusters = (
        peaks.groupby("cluster")
        .agg(
            samples=("sample", "nunique"),
            median_rt_sec=("apex_rt_sec", "median"),
            median_prominence=("prominence", "median"),
            maximum_prominence=("prominence", "max"),
        )
        .reset_index()
        .sort_values(["samples", "median_prominence"], ascending=[False, False])
    )
    chosen = clusters.iloc[0]
    center = float(chosen["median_rt_sec"])
    abundance_rows = []
    for sample, (times, signal) in traces.items():
        active = np.abs(times - center) <= args.integration_half_window_sec
        local_t, local_y = times[active], signal[active]
        baseline = float(np.percentile(local_y, 20)) if len(local_y) else 0.0
        corrected = np.maximum(local_y - baseline, 0)
        area = float(np.trapz(corrected, local_t)) if len(local_t) > 1 else 0.0
        match = SAMPLE_RE.match(sample)
        assert match is not None
        patient, tissue = match.groups()
        abundance_rows.append(
            {"sample": sample, "patient": patient, "tissue": tissue, "area": area}
        )
    abundance = pd.DataFrame(abundance_rows)
    effects = {
        "Rmu_vs_RN": paired_effect(abundance, "Rmu", "RN"),
        "Rtu_vs_RN": paired_effect(abundance, "Rtu", "RN"),
        "Ltu_vs_LN": paired_effect(abundance, "Ltu", "LN"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(args.output_dir / "candidate_peaks.csv", index=False)
    clusters.to_csv(args.output_dir / "rt_clusters.csv", index=False)
    abundance.to_csv(args.output_dir / "chosen_cluster_eic.csv", index=False)
    report = {
        "status": "MTBLS13729_SAH_exact_mass_pilot_complete",
        "formal": False,
        "target": {"mz": args.mz, "ppm": args.ppm, "putative_adduct": "[M+H]+"},
        "chosen_rt_cluster": chosen.to_dict(),
        "effects": effects,
        "selection": "highest sample-prevalence exact-mass RT cluster, phenotype blind",
        "claim_limit": (
            "Exact-mass SAH-like pilot only. Without diagnostic MS2 or a coeluting standard, "
            "the feature is not identified as SAH and is not evidence of AHCY activity."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
