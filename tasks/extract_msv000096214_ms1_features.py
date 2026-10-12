#!/usr/bin/env python
"""Checkpointed OpenMS MS1 feature extraction for MSV000096214 (contract 3.3).

Detection parameters are frozen except noise_threshold_int, which comes from
the genotype-blind pooled-QC calibration. Every file is processed with its
polarity's frozen threshold. MS2 precursor tables are captured alongside as
the DDA linkage layer. Never reads genotype metadata; detection absence is
not zero abundance (gap filling is a separate stage).
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


def load_ms1(path: Path) -> tuple[oms.MSExperiment, dict[str, Any], dict[str, Any]]:
    """Load one mzXML; return MS1 experiment, MS1 summary and MS2 precursor table."""
    experiment = oms.MSExperiment()
    loader = oms.MzXMLFile()
    try:
        options = loader.getOptions()
        options.setMSLevels([1])
        loader.setOptions(options)
    except AttributeError:
        pass  # loader ignores MS-level filtering; MS2 handled below
    loader.load(str(path), experiment)
    experiment.sortSpectra(True)

    # mzXML loaders may keep all levels; split MS1 vs MS2 explicitly.
    ms1 = oms.MSExperiment()
    precursors: list[dict[str, float]] = []
    ms1_intensities: list[np.ndarray] = []
    rt_span: list[float] = []
    for spectrum in experiment:
        level = spectrum.getMSLevel()
        if level == 1:
            ms1.addSpectrum(spectrum)
            _, intensity = spectrum.get_peaks()
            if len(intensity):
                ms1_intensities.append(np.asarray(intensity, dtype=np.float64))
            rt_span.append(float(spectrum.getRT()))
        elif level == 2:
            charge = 0
            precursor_mz = math.nan
            for precursor in spectrum.getPrecursors():
                precursor_mz = float(precursor.getMZ())
                charge = int(precursor.getCharge())
                break
            _, intensity = spectrum.get_peaks()
            precursors.append({
                "precursor_mz": precursor_mz,
                "rt_sec": float(spectrum.getRT()),
                "charge": charge,
                "base_intensity": float(np.max(intensity)) if len(intensity) else 0.0,
                "n_peaks": int(len(intensity)),
            })
    ms1.updateRanges()
    pooled = np.concatenate(ms1_intensities) if ms1_intensities else np.asarray([], dtype=float)
    positive = pooled[pooled > 0]
    summary = {
        "n_ms1": int(ms1.size()),
        "n_ms2": len(precursors),
        "n_ms1_points": int(pooled.size),
        "intensity_p50": float(np.percentile(positive, 50)) if positive.size else math.nan,
        "intensity_q90": float(np.percentile(positive, 90)) if positive.size else math.nan,
        "rt_min_sec": min(rt_span) if rt_span else math.nan,
        "rt_max_sec": max(rt_span) if rt_span else math.nan,
    }
    return ms1, summary, {"precursors": pd.DataFrame(precursors)}


def detect_features(
    experiment: oms.MSExperiment, noise_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    mass_traces: list[Any] = []
    detector = oms.MassTraceDetection()
    detector.setLogType(oms.LogType.NONE)
    params = detector.getDefaults()
    params.setValue("mass_error_ppm", 5.0)
    params.setValue("noise_threshold_int", float(noise_threshold))
    params.setValue("min_trace_length", 5.0)
    params.setValue("max_trace_length", 120.0)
    params.setValue("trace_termination_criterion", "outlier")
    params.setValue("trace_termination_outliers", 5)
    detector.setParameters(params)
    mass_traces = detector.run(experiment, 0)

    peak_detector = oms.ElutionPeakDetection()
    peak_detector.setLogType(oms.LogType.NONE)
    epd_params = peak_detector.getDefaults()
    epd_params.setValue("width_filtering", "auto")
    peak_detector.setParameters(epd_params)
    split_traces = peak_detector.detectPeaks(mass_traces)
    final_traces = peak_detector.filterByPeakWidth(split_traces)

    feature_map = oms.FeatureMap()
    finder = oms.FeatureFindingMetabo()
    finder.setLogType(oms.LogType.NONE)
    ffm_params = finder.getDefaults()
    ffm_params.setValue("isotope_filtering_model", "none")
    ffm_params.setValue("mz_scoring_13C", "true")
    ffm_params.setValue("remove_single_traces", "false")
    ffm_params.setValue("report_convex_hulls", "false")
    ffm_params.setValue("report_summed_ints", "false")
    finder.setParameters(ffm_params)
    finder.run(final_traces, feature_map)
    feature_map.setUniqueIds()

    rows = [
        {"mz": float(f.getMZ()), "rt_sec": float(f.getRT()),
         "intensity": float(f.getIntensity()), "charge": int(f.getCharge()),
         "quality": float(f.getOverallQuality()), "width_sec": float(f.getWidth())}
        for f in feature_map
    ]
    feature_df = pd.DataFrame(rows)
    summary = {
        "noise_threshold": float(noise_threshold),
        "n_mass_traces": len(mass_traces),
        "n_final_traces": len(final_traces),
        "n_features": int(feature_map.size()),
        "detect_seconds": time.time() - started,
    }
    return feature_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/msv000096214/ms1_features"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    calibration = json.loads((args.root / "calibration.json").read_text(encoding="utf-8"))
    thresholds = {polarity: float(body["noise_threshold_int"])
                  for polarity, body in calibration["polarity"].items()}
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    manifest = pd.read_csv(args.root / "download_manifest.csv")
    manifest = manifest.loc[manifest["ok"]].sort_values(["polarity", "filename"])
    total = len(manifest)
    for i, item in enumerate(manifest.itertuples(index=False), start=1):
        source = args.root / "mzxml" / item.polarity / item.filename
        sample = Path(item.filename).stem
        target = feature_dir / f"{item.polarity}__{sample}__features.csv.gz"
        summary_path = feature_dir / f"{item.polarity}__{sample}__summary.json"
        prec_path = feature_dir / f"{item.polarity}__{sample}__ms2_precursors.csv.gz"
        base = {"polarity": item.polarity, "sample": sample, "role": item.role,
                "source": str(source), "output": str(target)}
        if target.exists() and summary_path.exists() and prec_path.exists() and not args.force:
            record = json.loads(summary_path.read_text(encoding="utf-8"))
            record["status"] = "reused"
            rows.append(record)
            continue

        started = time.time()
        try:
            experiment, load_summary, precursor_table = load_ms1(source)
            features, detection = detect_features(experiment, thresholds[item.polarity])
            features.to_csv(target, index=False, compression="gzip")
            precursor_table["precursors"].to_csv(prec_path, index=False, compression="gzip")
            record = {**base, "status": "complete",
                      "elapsed_seconds": time.time() - started,
                      **load_summary, **detection}
            summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            rows.append(record)
            print(json.dumps({"polarity": item.polarity, "sample": sample,
                              "n_features": len(features),
                              "n_ms2": load_summary["n_ms2"],
                              "seconds": round(record["elapsed_seconds"], 1)}), flush=True)
        except Exception as exc:
            rows.append({**base, "status": "failed", "error": repr(exc),
                         "elapsed_seconds": time.time() - started})
            print(json.dumps(rows[-1]), flush=True)
        if i % 5 == 0 or i == total:
            pd.DataFrame(rows).to_csv(args.output_dir / "extraction_manifest.partial.csv", index=False)
            print(f"[progress {i}/{total}]", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "extraction_manifest.csv", index=False)
    report = {
        "status": "complete_with_failures" if (frame["status"] == "failed").any() else "complete",
        "data_role": "development/consumed",
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
        "thresholds": thresholds,
        "counts": frame["status"].value_counts().to_dict(),
        "genotype_column_read": False,
        "interpretation": "Detection absence is not zero abundance; gap filling follows.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
