#!/usr/bin/env python
"""Genotype-blind noise-threshold calibration for MSV000096214 (contract 3.2).

Runs the frozen OpenMS detection recipe on the pooled-QC mzXML files only,
scans the pre-registered threshold grid from stringent to permissive, and
freezes the highest threshold whose median QC feature count reaches 2,000.
Writes calibration.json which
the extraction stage consumes read-only. Never reads genotype metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyopenms as oms

GRID = [50, 100, 200, 400, 800, 1600, 3200]  # v1.1 scale-aware amendment (contract revision)
TARGET_COUNT = 2000


def load_ms1(path: Path) -> oms.MSExperiment:
    experiment = oms.MSExperiment()
    loader = oms.MzXMLFile()
    options = loader.getOptions()
    options.setMSLevels([1])
    loader.setOptions(options)
    loader.load(str(path), experiment)
    experiment.sortSpectra(True)
    return experiment


def detect(experiment: oms.MSExperiment, noise_threshold: float) -> int:
    mass_traces = []
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
    return int(feature_map.size())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    args = parser.parse_args()
    out_path = args.root / "calibration.json"
    if out_path.exists():
        raise FileExistsError(out_path)

    report: dict[str, object] = {
        "status": "MSV000096214_NOISE_CALIBRATION_FROZEN",
        "data_role": "development/consumed",
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
        "rule": "highest grid threshold whose median QC feature count >= 2000",
        "grid": GRID,
        "target_count": TARGET_COUNT,
        "genotype_column_read": False,
        "polarity": {},
    }
    for polarity in ("positive", "negative"):
        qc_dir = args.root / "mzxml" / polarity
        qc_files = sorted(
            path for path in qc_dir.glob("*.mzXML")
            if path.name.startswith("SJ-Mutant-Flavo-CQ"))
        if not qc_files:
            raise FileNotFoundError(f"no pooled-QC mzXML under {qc_dir}")
        experiments = {path.name: load_ms1(path) for path in qc_files}
        curve: dict[str, dict[str, int]] = {}
        chosen: float | None = None
        # Feature count is non-increasing as the intensity threshold rises.
        # Descending search therefore selects the most stringent threshold
        # that retains the pre-registered minimum information floor.
        for threshold in reversed(GRID):
            counts = {}
            for name, experiment in experiments.items():
                counts[name] = detect(experiment, threshold)
            values = sorted(counts.values())
            median = values[len(values) // 2]
            curve[f"{threshold:g}"] = {"counts": counts, "median": median}
            print(f"[{polarity}] T={threshold:g} counts={counts} median={median}", flush=True)
            if median >= TARGET_COUNT:
                chosen = threshold
                break
        if chosen is None:
            chosen = max(GRID, key=lambda t: curve[f"{t:g}"]["median"])
        report["polarity"][polarity] = {
            "qc_files": [path.name for path in qc_files],
            "qc_sha256": {path.name: sha256(path) for path in qc_files},
            "curve": curve,
            "noise_threshold_int": float(chosen),
        }
        print(f"[{polarity}] FROZEN noise_threshold_int={chosen:g}", flush=True)

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["polarity"], indent=2), flush=True)


if __name__ == "__main__":
    main()
