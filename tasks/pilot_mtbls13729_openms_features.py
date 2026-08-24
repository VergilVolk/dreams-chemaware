#!/usr/bin/env python
"""Parameter pilot for untargeted MTBLS13729 MS1 feature detection.

Uses the established OpenMS metabolomics sequence:
MassTraceDetection -> ElutionPeakDetection -> FeatureFindingMetabo.
The pilot is deliberately small and compares noise thresholds before a full
cohort run is committed.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyopenms as oms


DEFAULT_PANELS = ("neg_rp", "pos_rp")
DEFAULT_SAMPLES = ("P01-Ltu", "P01-LN", "P21-Rmu", "P21-RN")


def load_ms1(path: Path) -> tuple[oms.MSExperiment, dict[str, Any]]:
    experiment = oms.MSExperiment()
    loader = oms.MzMLFile()
    options = loader.getOptions()
    options.setMSLevels([1])
    loader.setOptions(options)
    started = time.time()
    loader.load(str(path), experiment)
    experiment.sortSpectra(True)

    intensities: list[np.ndarray] = []
    spectrum_types: dict[str, int] = {}
    for spectrum in experiment:
        spectrum_type = str(spectrum.getType())
        spectrum_types[spectrum_type] = spectrum_types.get(spectrum_type, 0) + 1
        _, intensity = spectrum.get_peaks()
        if len(intensity):
            intensities.append(np.asarray(intensity, dtype=np.float64))
    pooled = np.concatenate(intensities) if intensities else np.asarray([], dtype=float)
    positive = pooled[pooled > 0]
    summary = {
        "n_ms1": int(experiment.size()),
        "spectrum_types": spectrum_types,
        "n_ms1_points": int(pooled.size),
        "intensity_p10": float(np.percentile(positive, 10)) if positive.size else math.nan,
        "intensity_p25": float(np.percentile(positive, 25)) if positive.size else math.nan,
        "intensity_p50": float(np.percentile(positive, 50)) if positive.size else math.nan,
        "intensity_p75": float(np.percentile(positive, 75)) if positive.size else math.nan,
        "intensity_p90": float(np.percentile(positive, 90)) if positive.size else math.nan,
        "load_seconds": time.time() - started,
    }
    return experiment, summary


def detect_features(
    experiment: oms.MSExperiment,
    noise_threshold: float,
    mass_error_ppm: float,
    min_trace_length: float,
    max_trace_length: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    mass_traces: list[Any] = []
    detector = oms.MassTraceDetection()
    detector.setLogType(oms.LogType.NONE)
    params = detector.getDefaults()
    params.setValue("mass_error_ppm", float(mass_error_ppm))
    params.setValue("noise_threshold_int", float(noise_threshold))
    params.setValue("min_trace_length", float(min_trace_length))
    params.setValue("max_trace_length", float(max_trace_length))
    params.setValue("trace_termination_criterion", "outlier")
    params.setValue("trace_termination_outliers", 5)
    detector.setParameters(params)
    detector.run(experiment, mass_traces, 0)

    split_traces: list[Any] = []
    peak_detector = oms.ElutionPeakDetection()
    peak_detector.setLogType(oms.LogType.NONE)
    epd_params = peak_detector.getDefaults()
    epd_params.setValue("width_filtering", "auto")
    peak_detector.setParameters(epd_params)
    peak_detector.detectPeaks(mass_traces, split_traces)
    final_traces: list[Any] = []
    peak_detector.filterByPeakWidth(split_traces, final_traces)

    feature_map = oms.FeatureMap()
    feature_chromatograms: list[Any] = []
    finder = oms.FeatureFindingMetabo()
    finder.setLogType(oms.LogType.NONE)
    ffm_params = finder.getDefaults()
    ffm_params.setValue("isotope_filtering_model", "none")
    ffm_params.setValue("mz_scoring_13C", "true")
    ffm_params.setValue("remove_single_traces", "false")
    ffm_params.setValue("report_convex_hulls", "false")
    ffm_params.setValue("report_summed_ints", "false")
    finder.setParameters(ffm_params)
    finder.run(final_traces, feature_map, feature_chromatograms)
    feature_map.setUniqueIds()

    rows = []
    for i, feature in enumerate(feature_map):
        rows.append(
            {
                "feature_index": i,
                "mz": float(feature.getMZ()),
                "rt_sec": float(feature.getRT()),
                "intensity": float(feature.getIntensity()),
                "charge": int(feature.getCharge()),
                "quality": float(feature.getOverallQuality()),
                "width_sec": float(feature.getWidth()),
            }
        )
    feature_df = pd.DataFrame(rows)
    summary = {
        "noise_threshold": float(noise_threshold),
        "n_mass_traces": len(mass_traces),
        "n_split_traces": len(split_traces),
        "n_final_traces": len(final_traces),
        "n_features": int(feature_map.size()),
        "feature_intensity_median": float(feature_df["intensity"].median()) if len(feature_df) else math.nan,
        "feature_quality_median": float(feature_df["quality"].median()) if len(feature_df) else math.nan,
        "detect_seconds": time.time() - started,
    }
    return feature_df, summary


def greedy_match(a: pd.DataFrame, b: pd.DataFrame, ppm: float, rt_sec: float) -> dict[str, float]:
    if a.empty or b.empty:
        return {"matches": 0, "match_fraction_min": 0.0, "feature_jaccard": 0.0}
    b_sorted = b.sort_values("mz").reset_index(drop=True)
    b_mz = b_sorted["mz"].to_numpy(dtype=float)
    b_rt = b_sorted["rt_sec"].to_numpy(dtype=float)
    used: set[int] = set()
    matches = 0
    for row in a.sort_values("intensity", ascending=False).itertuples(index=False):
        tolerance = float(row.mz) * ppm * 1e-6
        lo = bisect.bisect_left(b_mz, float(row.mz) - tolerance)
        hi = bisect.bisect_right(b_mz, float(row.mz) + tolerance)
        best = None
        best_cost = math.inf
        for j in range(lo, hi):
            if j in used:
                continue
            delta_rt = abs(float(row.rt_sec) - b_rt[j])
            if delta_rt > rt_sec:
                continue
            delta_ppm = abs(b_mz[j] - float(row.mz)) / float(row.mz) * 1e6
            cost = (delta_ppm / ppm) ** 2 + (delta_rt / rt_sec) ** 2
            if cost < best_cost:
                best = j
                best_cost = cost
        if best is not None:
            used.add(best)
            matches += 1
    union = len(a) + len(b) - matches
    return {
        "matches": int(matches),
        "match_fraction_min": float(matches / min(len(a), len(b))),
        "feature_jaccard": float(matches / union) if union else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_feature_pilot"))
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS))
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--noise-thresholds", nargs="+", type=float, default=[1e3, 1e4, 1e5])
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--match-ppm", type=float, default=5.0)
    parser.add_argument("--match-rt-sec", type=float, default=10.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    args = parser.parse_args()

    root = args.mzml_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    feature_dir = out / "features"
    feature_dir.mkdir(exist_ok=True)

    summaries: list[dict[str, Any]] = []
    feature_tables: dict[tuple[str, str, float], pd.DataFrame] = {}
    for panel in args.panels:
        for sample in args.samples:
            path = root / panel / f"{sample}.mzML"
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Loading {panel}/{sample}...", flush=True)
            experiment, load_summary = load_ms1(path)
            print(json.dumps(load_summary, ensure_ascii=False), flush=True)
            for threshold in args.noise_thresholds:
                print(f"  threshold={threshold:g}", flush=True)
                feature_df, detection = detect_features(
                    experiment,
                    noise_threshold=threshold,
                    mass_error_ppm=args.mass_error_ppm,
                    min_trace_length=args.min_trace_sec,
                    max_trace_length=args.max_trace_sec,
                )
                feature_df.insert(0, "sample_name", sample)
                feature_df.insert(0, "panel", panel)
                feature_df.insert(2, "noise_threshold", threshold)
                feature_tables[(panel, sample, threshold)] = feature_df
                feature_df.to_csv(feature_dir / f"{panel}__{sample}__noise_{threshold:g}.csv.gz", index=False)
                summaries.append({"panel": panel, "sample_name": sample, **load_summary, **detection})
                print(json.dumps(detection, ensure_ascii=False), flush=True)
            del experiment

    pair_rows: list[dict[str, Any]] = []
    pairs = (("P01-Ltu", "P01-LN"), ("P21-Rmu", "P21-RN"))
    for panel in args.panels:
        for threshold in args.noise_thresholds:
            for tumor, normal in pairs:
                a = feature_tables[(panel, tumor, threshold)]
                b = feature_tables[(panel, normal, threshold)]
                matched = greedy_match(a, b, ppm=args.match_ppm, rt_sec=args.match_rt_sec)
                pair_rows.append(
                    {
                        "panel": panel,
                        "noise_threshold": threshold,
                        "tumor": tumor,
                        "normal": normal,
                        "n_tumor_features": len(a),
                        "n_normal_features": len(b),
                        **matched,
                    }
                )

    summary_df = pd.DataFrame(summaries)
    pair_df = pd.DataFrame(pair_rows)
    summary_df.to_csv(out / "detection_summary.csv", index=False)
    pair_df.to_csv(out / "pair_reproducibility.csv", index=False)
    report = {
        "status": "pilot_complete",
        "parameters": vars(args) | {"mzml_root": str(root), "output_dir": str(out)},
        "detection_summary": summaries,
        "pair_reproducibility": pair_rows,
        "interpretation": "Select a threshold only after balancing feature yield, paired reproducibility, and downstream blank/QC limitations.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved pilot results to {out}", flush=True)


if __name__ == "__main__":
    main()
