#!/usr/bin/env python
"""Align OpenMS pilot feature tables and construct a consensus intensity matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyopenms as oms

from pilot_mtbls13729_openms_features import greedy_match


def dataframe_to_feature_map(frame: pd.DataFrame, source: str) -> oms.FeatureMap:
    uid = oms.UniqueIdGenerator()
    feature_map = oms.FeatureMap()
    feature_map.setUniqueId(uid.getUniqueId())
    feature_map.setPrimaryMSRunPath([source.encode()])
    for row in frame.itertuples(index=False):
        feature = oms.Feature()
        feature.setMZ(float(row.mz))
        feature.setRT(float(row.rt_sec))
        feature.setIntensity(float(row.intensity))
        feature.setCharge(int(row.charge))
        feature.setOverallQuality(float(row.quality))
        feature.setWidth(float(row.width_sec))
        feature.setUniqueId(uid.getUniqueId())
        feature_map.push_back(feature)
    feature_map.updateRanges()
    return feature_map


def feature_map_to_dataframe(feature_map: oms.FeatureMap) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mz": float(feature.getMZ()),
                "rt_sec": float(feature.getRT()),
                "intensity": float(feature.getIntensity()),
                "charge": int(feature.getCharge()),
                "quality": float(feature.getOverallQuality()),
                "width_sec": float(feature.getWidth()),
            }
            for feature in feature_map
        ]
    )


def align_feature_maps(
    maps: list[oms.FeatureMap],
    reference_index: int,
    max_shift_sec: float,
) -> tuple[list[oms.FeatureMap], list[dict[str, Any]]]:
    aligner = oms.MapAlignmentAlgorithmPoseClustering()
    aligner.setLogType(oms.LogType.NONE)
    params = aligner.getDefaults()
    params.setValue("max_num_peaks_considered", 2000)
    params.setValue("superimposer:mz_pair_max_distance", 0.01)
    params.setValue("superimposer:max_shift", float(max_shift_sec))
    params.setValue("pairfinder:distance_MZ:max_difference", 10.0)
    params.setValue("pairfinder:distance_MZ:unit", "ppm")
    params.setValue("pairfinder:distance_RT:max_difference", float(max_shift_sec))
    aligner.setParameters(params)
    aligner.setReference(maps[reference_index])

    diagnostics: list[dict[str, Any]] = []
    for i, feature_map in enumerate(maps):
        if i == reference_index:
            diagnostics.append({"map_index": i, "reference": True, "n_transform_points": 0})
            continue
        transformation = oms.TransformationDescription()
        aligner.align(feature_map, transformation)
        n_points = len(transformation.getDataPoints())
        oms.MapAlignmentTransformer().transformRetentionTimes(feature_map, transformation, False)
        diagnostics.append({"map_index": i, "reference": False, "n_transform_points": n_points})
    return maps, diagnostics


def group_to_matrix(
    maps: list[oms.FeatureMap],
    sample_names: list[str],
    mz_ppm: float,
    rt_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouping = oms.FeatureGroupingAlgorithmQT()
    params = grouping.getDefaults()
    params.setValue("distance_MZ:max_difference", float(mz_ppm))
    params.setValue("distance_MZ:unit", "ppm")
    params.setValue("distance_RT:max_difference", float(rt_sec))
    params.setValue("ignore_charge", "true")
    grouping.setParameters(params)
    consensus = oms.ConsensusMap()
    grouping.group(maps, consensus)

    metadata_rows: list[dict[str, Any]] = []
    intensity_rows: list[dict[str, Any]] = []
    for feature_id, consensus_feature in enumerate(consensus):
        intensity_row: dict[str, Any] = {name: math.nan for name in sample_names}
        handles = list(consensus_feature.getFeatureList())
        for handle in handles:
            map_index = int(handle.getMapIndex())
            if 0 <= map_index < len(sample_names):
                intensity_row[sample_names[map_index]] = float(handle.getIntensity())
        metadata_rows.append(
            {
                "feature_id": feature_id,
                "mz": float(consensus_feature.getMZ()),
                "rt_sec": float(consensus_feature.getRT()),
                "n_samples_detected": len(handles),
                "prevalence": len(handles) / len(sample_names),
            }
        )
        intensity_rows.append({"feature_id": feature_id, **intensity_row})
    return pd.DataFrame(metadata_rows), pd.DataFrame(intensity_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/mtbls13729/ms1_feature_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_feature_alignment_pilot"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--samples", nargs="+", default=["P01-Ltu", "P01-LN", "P21-Rmu", "P21-RN"])
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--group-ppm", type=float, default=5.0)
    parser.add_argument("--group-rt-sec", type=float, default=10.0)
    parser.add_argument("--alignment-max-shift-sec", type=float, default=60.0)
    args = parser.parse_args()

    pilot = args.pilot_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "alignment_pilot_complete", "panels": {}}

    for panel in args.panels:
        frames: list[pd.DataFrame] = []
        maps: list[oms.FeatureMap] = []
        for sample in args.samples:
            path = pilot / "features" / f"{panel}__{sample}__noise_{args.noise_threshold:g}.csv.gz"
            frame = pd.read_csv(path)
            frames.append(frame)
            maps.append(dataframe_to_feature_map(frame, source=f"{panel}/{sample}.mzML"))

        counts = [len(frame) for frame in frames]
        reference_index = min(range(len(counts)), key=lambda i: abs(counts[i] - float(np.median(counts))))
        pre = {}
        for tumor_idx, normal_idx in ((0, 1), (2, 3)):
            key = f"{args.samples[tumor_idx]}__{args.samples[normal_idx]}"
            pre[key] = greedy_match(frames[tumor_idx], frames[normal_idx], args.group_ppm, args.group_rt_sec)

        maps, diagnostics = align_feature_maps(maps, reference_index, args.alignment_max_shift_sec)
        aligned_frames = [feature_map_to_dataframe(feature_map) for feature_map in maps]
        post = {}
        for tumor_idx, normal_idx in ((0, 1), (2, 3)):
            key = f"{args.samples[tumor_idx]}__{args.samples[normal_idx]}"
            post[key] = greedy_match(aligned_frames[tumor_idx], aligned_frames[normal_idx], args.group_ppm, args.group_rt_sec)

        metadata, matrix = group_to_matrix(maps, args.samples, args.group_ppm, args.group_rt_sec)
        metadata.to_csv(out / f"{panel}__feature_metadata.csv", index=False)
        matrix.to_csv(out / f"{panel}__intensity_matrix.csv", index=False)
        for sample, frame in zip(args.samples, aligned_frames):
            frame.to_csv(out / f"{panel}__{sample}__aligned.csv.gz", index=False)

        panel_report = {
            "noise_threshold": args.noise_threshold,
            "reference_sample": args.samples[reference_index],
            "feature_counts": dict(zip(args.samples, counts)),
            "alignment_diagnostics": diagnostics,
            "pair_matching_before": pre,
            "pair_matching_after": post,
            "n_consensus_features": int(len(metadata)),
            "consensus_prevalence": {
                "detected_all_samples": int((metadata["n_samples_detected"] == len(args.samples)).sum()),
                "detected_at_least_half": int((metadata["prevalence"] >= 0.5).sum()),
                "median": float(metadata["prevalence"].median()),
            },
        }
        report["panels"][panel] = panel_report
        print(json.dumps({panel: panel_report}, indent=2, ensure_ascii=False), flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved alignment pilot to {out}", flush=True)


if __name__ == "__main__":
    main()
