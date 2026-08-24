#!/usr/bin/env python
"""Build a cross-sample OpenMS consensus target list for MTBLS1905 (positive HILIC).

Mirrors tasks/build_mtbls13729_ms1_consensus.py: feature detection is only the
discovery stage, and no retention-time transform is applied here.  The consensus
list is re-quantified uniformly from raw MS1 EICs by a separate script (once QC
files land, RT alignment and QC/blank filtering run against them).

The MTBLS1905 sample naming is <patient><C|M|N> where:
  C = core tumour tissue, M = edge tumour tissue, N = adjacent non-tumour tissue.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms

from align_mtbls13729_openms_features import dataframe_to_feature_map

SAMPLE_RE = re.compile(r"^(\d+)([CMN])$")
REGION_LABEL = {"C": "core", "M": "edge", "N": "normal"}


def sample_class(name: str) -> tuple[str, str]:
    match = SAMPLE_RE.match(name)
    if not match:
        return "unknown", "unknown"
    patient, region = match.groups()
    return patient, REGION_LABEL[region]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/external/MTBLS1905/metadata/positive_ms1_processing_manifest.tsv"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/external/MTBLS1905/ms1_openms_features/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/MTBLS1905/ms1_consensus"))
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--mz-ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    parser.add_argument("--global-min-prevalence", type=float, default=0.50)
    parser.add_argument("--group-min-prevalence", type=float, default=0.70)
    parser.add_argument("--min-group-size", type=int, default=6)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    manifest = manifest[manifest["sample_role"] == "patient"].copy()
    manifest = manifest.sort_values("sample_name", kind="stable")
    sample_names = manifest["sample_name"].tolist()
    region_of = dict(zip(manifest["sample_name"], manifest["tissue_type"]))

    source = args.feature_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    maps: list[oms.FeatureMap] = []
    counts: dict[str, int] = {}
    loaded: list[str] = []
    skipped: list[str] = []
    for name in sample_names:
        path = source / f"{name}__noise_{args.noise_threshold:g}.csv.gz"
        if not path.exists():
            skipped.append(name)
            continue
        frame = pd.read_csv(path)
        maps.append(dataframe_to_feature_map(frame, source=f"HILIC_positive/{name}.mzML"))
        counts[name] = len(frame)
        loaded.append(name)
    if not maps:
        raise FileNotFoundError(f"No feature tables found in {source}")
    sample_names = loaded

    grouping = oms.FeatureGroupingAlgorithmQT()
    params = grouping.getDefaults()
    params.setValue("distance_MZ:max_difference", float(args.mz_ppm))
    params.setValue("distance_MZ:unit", "ppm")
    params.setValue("distance_RT:max_difference", float(args.rt_sec))
    params.setValue("ignore_charge", "true")
    grouping.setParameters(params)
    consensus = oms.ConsensusMap()
    grouping.group(maps, consensus)

    region_idx: dict[str, list[int]] = {label: [] for label in ("core", "edge", "normal")}
    for i, name in enumerate(sample_names):
        _, region = sample_class(name)
        region_idx.setdefault(region, []).append(i)

    meta_rows: list[dict[str, object]] = []
    intensity_rows: list[dict[str, object]] = []
    for feature_id, cf in enumerate(consensus):
        present = np.zeros(len(sample_names), dtype=bool)
        intensities = np.full(len(sample_names), np.nan, dtype=float)
        for handle in cf.getFeatureList():
            idx = int(handle.getMapIndex())
            if 0 <= idx < len(sample_names):
                present[idx] = True
                intensities[idx] = float(handle.getIntensity())
        global_prevalence = float(present.mean())
        group_prev: dict[str, float] = {}
        for label, idxs in region_idx.items():
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
    metadata.to_csv(out / "consensus_metadata.csv.gz", index=False)
    discovery.to_csv(out / "discovery_intensity_matrix.csv.gz", index=False)
    targets = metadata[metadata["keep_for_requantification"]].copy()
    targets.to_csv(out / "requantification_targets.csv.gz", index=False)

    sample_table = []
    for i, name in enumerate(sample_names):
        patient, region = sample_class(name)
        sample_table.append(
            {
                "map_index": i,
                "sample_name": name,
                "patient": patient,
                "region": region,
                "tissue_type": region_of.get(name, ""),
            }
        )
    pd.DataFrame(sample_table).to_csv(out / "samples.csv", index=False)

    report = {
        "study": "MTBLS1905",
        "stage": "MS1 discovery consensus (no RT transform; requantify from raw EICs later)",
        "n_samples": len(sample_names),
        "feature_counts": counts,
        "n_raw_consensus": int(len(metadata)),
        "n_requantification_targets": int(len(targets)),
        "n_detected_all_samples": int((metadata["n_samples_detected"] == len(sample_names)).sum()),
        "median_global_prevalence": float(metadata["global_prevalence"].median()),
        "parameters": {
            "mz_ppm": args.mz_ppm,
            "rt_sec": args.rt_sec,
            "global_min_prevalence": args.global_min_prevalence,
            "group_min_prevalence": args.group_min_prevalence,
            "min_group_size": args.min_group_size,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved consensus outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
