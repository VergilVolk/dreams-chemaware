#!/usr/bin/env python
"""Build a cross-sample OpenMS consensus target list for MTBLS13729.

No retention-time transform is applied. Feature detection is only the discovery
stage; the resulting consensus list is re-quantified uniformly from raw MS1 EICs
by a separate script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


SAMPLE_RE = re.compile(r"^(P\d{2})-(Ltu|Rtu|Rmu|LN|RN)$")


def dataframe_to_feature_map(frame: pd.DataFrame, source: str) -> oms.FeatureMap:
    """Convert a frozen feature table into an OpenMS FeatureMap."""

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


def sample_class(name: str) -> tuple[str, str, str]:
    match = SAMPLE_RE.match(name)
    if not match:
        return "unknown", "unknown", "unknown"
    patient, suffix = match.groups()
    tissue = "tumor" if suffix in {"Ltu", "Rtu", "Rmu"} else "normal"
    histology = "mucinous" if suffix == "Rmu" else ("tubular" if suffix in {"Ltu", "Rtu"} else "normal")
    return patient, tissue, histology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("data/mtbls13729/ms1_features_full/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--mz-ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    parser.add_argument("--global-min-prevalence", type=float, default=0.20)
    parser.add_argument("--group-min-prevalence", type=float, default=0.40)
    parser.add_argument("--min-group-size", type=int, default=8)
    args = parser.parse_args()

    source = args.feature_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"status": "complete", "panels": {}}

    for panel in args.panels:
        paths = sorted(source.glob(f"{panel}__*__noise_{args.noise_threshold:g}.csv.gz"))
        if not paths:
            raise FileNotFoundError(f"No extracted features found for {panel} in {source}")
        sample_names = [path.name.split("__")[1] for path in paths]
        maps = []
        counts = {}
        for path, sample in zip(paths, sample_names):
            frame = pd.read_csv(path)
            maps.append(dataframe_to_feature_map(frame, source=f"{panel}/{sample}.mzML"))
            counts[sample] = len(frame)

        grouping = oms.FeatureGroupingAlgorithmQT()
        params = grouping.getDefaults()
        params.setValue("distance_MZ:max_difference", float(args.mz_ppm))
        params.setValue("distance_MZ:unit", "ppm")
        params.setValue("distance_RT:max_difference", float(args.rt_sec))
        params.setValue("ignore_charge", "true")
        grouping.setParameters(params)
        consensus = oms.ConsensusMap()
        grouping.group(maps, consensus)

        meta_rows = []
        intensity_rows = []
        for feature_id, cf in enumerate(consensus):
            present = np.zeros(len(sample_names), dtype=bool)
            intensities = np.full(len(sample_names), np.nan, dtype=float)
            for handle in cf.getFeatureList():
                idx = int(handle.getMapIndex())
                if 0 <= idx < len(sample_names):
                    present[idx] = True
                    intensities[idx] = float(handle.getIntensity())
            global_prevalence = float(present.mean())
            group_prev = {}
            for label in ("tumor", "normal", "mucinous", "tubular"):
                idxs = []
                for i, sample in enumerate(sample_names):
                    _, tissue, histology = sample_class(sample)
                    if label == tissue or label == histology:
                        idxs.append(i)
                if len(idxs) >= args.min_group_size:
                    group_prev[label] = float(present[idxs].mean())
            keep = global_prevalence >= args.global_min_prevalence or any(
                value >= args.group_min_prevalence for value in group_prev.values()
            )
            meta_rows.append(
                {
                    "feature_id": feature_id,
                    "mz": float(cf.getMZ()),
                    "rt_sec": float(cf.getRT()),
                    "width_sec": float(cf.getWidth()),
                    "n_samples_detected": int(present.sum()),
                    "global_prevalence": global_prevalence,
                    **{f"prevalence_{key}": value for key, value in group_prev.items()},
                    "keep_for_requantification": bool(keep),
                }
            )
            intensity_rows.append({"feature_id": feature_id, **dict(zip(sample_names, intensities))})

        metadata = pd.DataFrame(meta_rows)
        discovery = pd.DataFrame(intensity_rows)
        metadata.to_csv(out / f"{panel}__consensus_metadata.csv.gz", index=False)
        discovery.to_csv(out / f"{panel}__discovery_intensity_matrix.csv.gz", index=False)
        targets = metadata[metadata["keep_for_requantification"]].copy()
        targets.to_csv(out / f"{panel}__requantification_targets.csv.gz", index=False)
        sample_table = []
        for i, sample in enumerate(sample_names):
            patient, tissue, histology = sample_class(sample)
            sample_table.append({"map_index": i, "sample_name": sample, "patient": patient, "tissue": tissue, "histology": histology})
        pd.DataFrame(sample_table).to_csv(out / f"{panel}__samples.csv", index=False)

        panel_report = {
            "n_samples": len(sample_names),
            "feature_counts": counts,
            "n_raw_consensus": len(metadata),
            "n_requantification_targets": len(targets),
            "n_detected_all_samples": int((metadata["n_samples_detected"] == len(sample_names)).sum()),
            "median_global_prevalence": float(metadata["global_prevalence"].median()),
            "parameters": {
                "mz_ppm": args.mz_ppm,
                "rt_sec": args.rt_sec,
                "global_min_prevalence": args.global_min_prevalence,
                "group_min_prevalence": args.group_min_prevalence,
            },
        }
        report["panels"][panel] = panel_report
        print(json.dumps({panel: panel_report}, indent=2), flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved consensus outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
