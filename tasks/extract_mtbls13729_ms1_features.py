#!/usr/bin/env python
"""Checkpointed OpenMS MS1 feature extraction for MTBLS13729.

The same detector parameters are used for every sample within a panel. Missing
features are *not* interpreted as zero here; consensus linking and targeted EIC
gap filling are separate downstream stages.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  
import pyopenms as oms


SAMPLE_RE = re.compile(r"^P\d{2}-(?:Ltu|Rtu|Rmu|LN|RN)$")


def load_ms1(path: Path) -> tuple[oms.MSExperiment, dict[str, Any]]:
    """Load only MS1 scans from one mzML file."""

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
    """Run the frozen OpenMS metabolomics feature-detection recipe."""

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

    rows = [
        {
            "feature_index": i,
            "mz": float(feature.getMZ()),
            "rt_sec": float(feature.getRT()),
            "intensity": float(feature.getIntensity()),
            "charge": int(feature.getCharge()),
            "quality": float(feature.getOverallQuality()),
            "width_sec": float(feature.getWidth()),
        }
        for i, feature in enumerate(feature_map)
    ]
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


def load_exclusions(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path, sep="\t")
    return {(str(row.panel), str(row.sample_name)) for row in frame.itertuples(index=False)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_features_full"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    parser.add_argument("--exclusions", type=Path, default=Path("data/mtbls13729/ms1_acquisition_audit/exclusions.tsv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.mzml_root.resolve()
    out = args.output_dir.resolve()
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    exclusions = load_exclusions(args.exclusions)
    rows: list[dict[str, object]] = []

    for panel in args.panels:
        paths = sorted((root / panel).glob("*.mzML"))
        paths = [p for p in paths if SAMPLE_RE.match(p.stem)]
        for i, path in enumerate(paths, start=1):
            sample = path.stem
            target = feature_dir / f"{panel}__{sample}__noise_{args.noise_threshold:g}.csv.gz"
            summary_path = feature_dir / f"{panel}__{sample}__summary.json"
            base = {"panel": panel, "sample_name": sample, "source": str(path), "output": str(target)}
            if (panel, sample) in exclusions:
                record = {**base, "status": "excluded", "reason": "listed in exclusions.tsv"}
                rows.append(record)
                print(json.dumps(record), flush=True)
                continue
            if target.exists() and summary_path.exists() and not args.force:
                record = json.loads(summary_path.read_text(encoding="utf-8"))
                record["status"] = "reused"
                rows.append(record)
                print(json.dumps({"panel": panel, "sample": sample, "status": "reused"}), flush=True)
                continue

            print(f"[{panel} {i}/{len(paths)}] Loading {sample}", flush=True)
            started = time.time()
            try:
                experiment, load_summary = load_ms1(path)
                features, detection = detect_features(
                    experiment,
                    noise_threshold=args.noise_threshold,
                    mass_error_ppm=args.mass_error_ppm,
                    min_trace_length=args.min_trace_sec,
                    max_trace_length=args.max_trace_sec,
                )
                features.insert(0, "sample_name", sample)
                features.insert(0, "panel", panel)
                features.to_csv(target, index=False)
                record = {
                    **base,
                    "status": "complete",
                    "elapsed_seconds": time.time() - started,
                    **load_summary,
                    **detection,
                }
                summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                rows.append(record)
                print(json.dumps({"panel": panel, "sample": sample, "n_features": len(features), "seconds": record["elapsed_seconds"]}), flush=True)
            except Exception as exc:  # preserve cohort progress and surface failures
                record = {**base, "status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started}
                rows.append(record)
                print(json.dumps(record), flush=True)

            pd.DataFrame(rows).to_csv(out / "extraction_manifest.partial.csv", index=False)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "extraction_manifest.csv", index=False)
    report = {
        "status": "complete_with_failures" if (manifest["status"] == "failed").any() else "complete",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "counts": manifest["status"].value_counts().to_dict(),
        "manifest": str(out / "extraction_manifest.csv"),
        "interpretation": "Detection absence is not zero abundance; targeted EIC gap filling is required after consensus construction.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
