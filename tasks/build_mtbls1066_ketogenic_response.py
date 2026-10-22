#!/usr/bin/env python3
"""Build two directed, MS2-linked MTBLS1066 ketogenic response objects.

Feature discovery and MS2 linkage are phenotype blind.  Group labels are used
only after the consensus feature universe has been frozen.  This first-pass
development object uses observed feature intensities and retains missingness;
it does not silently gap-fill a missing chromatographic peak with zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyopenms as oms

from extract_mtbls13729_ms1_features import detect_features, load_ms1


CONTRASTS = {
    "healthy_lct_mct8_kd_minus_control": ("H_MCT", "H_C"),
    "breast_cancer_chemo_lct_mct8_kd_minus_control": ("BC_MCT", "BC_C"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"filename", "group", "replicate", "panel", "local_path", "sha256"}
    if missing := required - set(frame):
        raise ValueError(f"staged manifest lacks {sorted(missing)}")
    if len(frame) != 15 or set(frame.panel) != {"POS_RP"}:
        raise ValueError("expected the frozen 15-file POS-RP manifest")
    expected = {"H_C": 4, "H_MCT": 4, "BC_C": 4, "BC_MCT": 3}
    if frame.groupby("group").size().astype(int).to_dict() != expected:
        raise ValueError("unexpected biological-unit topology")
    if frame.filename.duplicated().any():
        raise ValueError("duplicate filenames in manifest")
    for row in frame.itertuples(index=False):
        path = Path(row.local_path)
        if not path.is_file():
            raise FileNotFoundError(path)
    return frame.sort_values(["group", "replicate", "filename"]).reset_index(drop=True)


def dataframe_to_feature_map(frame: pd.DataFrame, source: str) -> oms.FeatureMap:
    generator = oms.UniqueIdGenerator()
    feature_map = oms.FeatureMap()
    feature_map.setUniqueId(generator.getUniqueId())
    feature_map.setPrimaryMSRunPath([source.encode()])
    for row in frame.itertuples(index=False):
        feature = oms.Feature()
        feature.setMZ(float(row.mz))
        feature.setRT(float(row.rt_sec))
        feature.setIntensity(float(row.intensity))
        feature.setCharge(int(row.charge))
        feature.setOverallQuality(float(row.quality))
        feature.setWidth(float(row.width_sec))
        feature.setUniqueId(generator.getUniqueId())
        feature_map.push_back(feature)
    feature_map.updateRanges()
    return feature_map


def extract_or_reuse_features(
    manifest: pd.DataFrame,
    work_dir: Path,
    *,
    noise_threshold: float,
    mass_error_ppm: float,
    min_trace_sec: float,
    max_trace_sec: float,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    feature_dir = work_dir / "per_sample_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    parameter_record = {
        "noise_threshold": float(noise_threshold),
        "mass_error_ppm": float(mass_error_ppm),
        "min_trace_sec": float(min_trace_sec),
        "max_trace_sec": float(max_trace_sec),
    }
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        source = Path(row.local_path)
        stem = source.stem
        output = feature_dir / f"{stem}.csv.gz"
        meta_path = feature_dir / f"{stem}.json"
        reusable = False
        if output.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reusable = (
                meta.get("source_sha256") == row.sha256
                and meta.get("parameters") == parameter_record
            )
        if reusable:
            frame = pd.read_csv(output)
            status = "reused"
            elapsed = 0.0
            load_summary: dict[str, Any] = {}
            detection: dict[str, Any] = {"n_features": len(frame)}
        else:
            started = time.time()
            print(f"[feature {number}/{len(manifest)}] {source.name}", flush=True)
            experiment, load_summary = load_ms1(source)
            frame, detection = detect_features(
                experiment,
                noise_threshold=noise_threshold,
                mass_error_ppm=mass_error_ppm,
                min_trace_length=min_trace_sec,
                max_trace_length=max_trace_sec,
            )
            frame.to_csv(output, index=False)
            elapsed = time.time() - started
            status = "complete"
            meta_path.write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "source_sha256": row.sha256,
                        "parameters": parameter_record,
                        "n_features": len(frame),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        frames.append(frame)
        audit.append(
            {
                "filename": row.filename,
                "group": row.group,
                "replicate": row.replicate,
                "status": status,
                "n_features": int(len(frame)),
                "n_ms1": int(load_summary.get("n_ms1", 0)),
                "elapsed_seconds": elapsed,
                "detector_n_mass_traces": int(detection.get("n_mass_traces", 0)),
            }
        )
    return frames, audit


def build_consensus(
    frames: list[pd.DataFrame], manifest: pd.DataFrame, mz_ppm: float, rt_sec: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = [
        dataframe_to_feature_map(frame, str(Path(row.local_path)))
        for frame, row in zip(frames, manifest.itertuples(index=False))
    ]
    grouping = oms.FeatureGroupingAlgorithmQT()
    params = grouping.getDefaults()
    params.setValue("distance_MZ:max_difference", float(mz_ppm))
    params.setValue("distance_MZ:unit", "ppm")
    params.setValue("distance_RT:max_difference", float(rt_sec))
    params.setValue("ignore_charge", "true")
    grouping.setParameters(params)
    consensus = oms.ConsensusMap()
    grouping.group(maps, consensus)

    names = manifest.filename.astype(str).tolist()
    meta_rows: list[dict[str, object]] = []
    intensity_rows: list[dict[str, object]] = []
    for feature_id, feature in enumerate(consensus):
        values = np.full(len(names), np.nan, dtype=float)
        for handle in feature.getFeatureList():
            map_index = int(handle.getMapIndex())
            if 0 <= map_index < len(values):
                values[map_index] = float(handle.getIntensity())
        meta_rows.append(
            {
                "feature_id": feature_id,
                "raw_entity_id": f"mtbls1066_pos_rp_{feature_id}",
                "mz": float(feature.getMZ()),
                "rt_sec": float(feature.getRT()),
                "width_sec": float(feature.getWidth()),
                "samples_detected": int(np.isfinite(values).sum()),
                "global_prevalence": float(np.isfinite(values).mean()),
            }
        )
        intensity_rows.append({"feature_id": feature_id, **dict(zip(names, values))})
    return pd.DataFrame(meta_rows), pd.DataFrame(intensity_rows)


def group_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    n = finite.sum(axis=1).astype(int)
    mean = np.divide(
        np.nansum(values, axis=1), n,
        out=np.full(values.shape[0], np.nan), where=n > 0,
    )
    centered = np.where(finite, values - mean[:, None], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), n - 1,
        out=np.full(values.shape[0], np.nan), where=n > 1,
    )
    return n, mean, variance


def estimate_responses(
    intensities: pd.DataFrame,
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    minimum_units: int,
    minimum_group_prevalence: float,
) -> pd.DataFrame:
    columns = manifest.filename.astype(str).tolist()
    raw = intensities[columns].to_numpy(float)
    transformed = np.full(raw.shape, np.nan, dtype=float)
    positive = np.isfinite(raw) & (raw > 0)
    transformed[positive] = np.log2(raw[positive])
    offsets = np.nanmedian(transformed, axis=0)
    if np.any(~np.isfinite(offsets)):
        raise RuntimeError("a biological sample has no positive detected feature")
    normalized = transformed - offsets[None, :]
    groups = manifest.group.to_numpy(str)

    outputs: list[pd.DataFrame] = []
    for contrast_id, (positive_group, negative_group) in CONTRASTS.items():
        positive_values = normalized[:, groups == positive_group]
        negative_values = normalized[:, groups == negative_group]
        n_pos, mean_pos, variance_pos = group_stats(positive_values)
        n_neg, mean_neg, variance_neg = group_stats(negative_values)
        beta = mean_pos - mean_neg
        se = np.sqrt(
            np.divide(
                variance_pos, n_pos,
                out=np.full(len(n_pos), np.nan), where=n_pos > 0,
            )
            + np.divide(
                variance_neg, n_neg,
                out=np.full(len(n_neg), np.nan), where=n_neg > 0,
            )
        )
        prevalence_pos = n_pos / positive_values.shape[1]
        prevalence_neg = n_neg / negative_values.shape[1]
        eligible = (
            (n_pos >= minimum_units)
            & (n_neg >= minimum_units)
            & (prevalence_pos >= minimum_group_prevalence)
            & (prevalence_neg >= minimum_group_prevalence)
            & np.isfinite(beta)
            & np.isfinite(se)
            & (se > 0)
        )
        output = metadata[["feature_id", "raw_entity_id", "mz", "rt_sec"]].copy()
        output.insert(0, "contrast_id", contrast_id)
        output["positive_group"] = positive_group
        output["negative_group"] = negative_group
        output["beta"] = beta
        output["se"] = se
        output["n_positive"] = n_pos
        output["n_negative"] = n_neg
        output["prevalence_positive"] = prevalence_pos
        output["prevalence_negative"] = prevalence_neg
        output["qc_pass"] = eligible
        output["identity_status"] = "unpartitioned"
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def link_representative_ms2(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    precursor_ppm: float,
    rt_window_sec: float,
    *,
    require_unique_candidate: bool = False,
    minimum_ms2_peaks: int = 3,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]], pd.DataFrame]:
    order = np.argsort(metadata.mz.to_numpy(float))
    sorted_mz = metadata.mz.to_numpy(float)[order]
    feature_rt = metadata.rt_sec.to_numpy(float)
    representatives: dict[int, dict[str, object]] = {}
    links: list[dict[str, object]] = []
    file_audit: list[dict[str, object]] = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        source = Path(row.local_path)
        experiment = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        loader.load(str(source), experiment)
        linked = 0
        ambiguous = 0
        low_peak_count = 0
        for spectrum in experiment:
            precursors = spectrum.getPrecursors()
            if not precursors:
                continue
            precursor_mz = float(precursors[0].getMZ())
            precursor_charge = int(precursors[0].getCharge())
            rt = float(spectrum.getRT())
            tolerance = precursor_mz * precursor_ppm * 1e-6
            left = int(np.searchsorted(sorted_mz, precursor_mz - tolerance, side="left"))
            right = int(np.searchsorted(sorted_mz, precursor_mz + tolerance, side="right"))
            candidates = order[left:right]
            if not len(candidates):
                continue
            candidates = candidates[np.abs(feature_rt[candidates] - rt) <= rt_window_sec]
            if not len(candidates):
                continue
            if require_unique_candidate and len(candidates) != 1:
                ambiguous += 1
                continue
            ppm_error = (metadata.mz.to_numpy(float)[candidates] - precursor_mz) / precursor_mz * 1e6
            rt_error = feature_rt[candidates] - rt
            distance = np.abs(ppm_error) / precursor_ppm + np.abs(rt_error) / rt_window_sec
            selected = int(candidates[int(np.argmin(distance))])
            feature_id = int(metadata.iloc[selected].feature_id)
            mzs, intensities = spectrum.get_peaks()
            mzs = np.asarray(mzs, dtype=float)
            intensities = np.asarray(intensities, dtype=float)
            if len(mzs) < minimum_ms2_peaks or not np.isfinite(intensities).all():
                low_peak_count += 1
                continue
            record = {
                "feature_id": feature_id,
                "filename": row.filename,
                "native_id": spectrum.getNativeID(),
                "precursor_mz": precursor_mz,
                "precursor_charge": precursor_charge,
                "ms2_rt_sec": rt,
                "precursor_ppm_error": float(ppm_error[int(np.argmin(distance))]),
                "rt_error_sec": float(rt_error[int(np.argmin(distance))]),
                "n_peaks": int(len(mzs)),
                "tic": float(np.sum(intensities)),
            }
            links.append(record)
            linked += 1
            ranking = (
                int(len(mzs)),
                float(np.sum(intensities)),
                -float(distance[int(np.argmin(distance))]),
            )
            current = representatives.get(feature_id)
            if current is None or ranking > current["ranking"]:
                representatives[feature_id] = {
                    **record,
                    "mzs": mzs,
                    "intensities": intensities,
                    "ranking": ranking,
                }
        file_audit.append(
            {
                "filename": row.filename,
                "group": row.group,
                "ms2_spectra": int(experiment.size()),
                "linked_ms2_spectra": linked,
                "ambiguous_candidate_ms2_spectra": ambiguous,
                "low_peak_count_ms2_spectra": low_peak_count,
            }
        )
        print(
            f"[MS2 {number}/{len(manifest)}] {source.name}: "
            f"{linked}/{experiment.size()} linked",
            flush=True,
        )
    return pd.DataFrame(links), representatives, pd.DataFrame(file_audit)


def write_mgf(path: Path, metadata: pd.DataFrame, representatives: dict[int, dict[str, object]]) -> pd.DataFrame:
    lookup = metadata.set_index("feature_id")
    index_rows: list[dict[str, object]] = []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for scan, feature_id in enumerate(sorted(representatives), start=1):
            record = representatives[feature_id]
            entity = lookup.loc[feature_id]
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE={entity.raw_entity_id}\n")
            handle.write(f"SCANS={feature_id}\n")
            handle.write(f"PEPMASS={record['precursor_mz']:.8f}\n")
            handle.write(f"RTINSECONDS={record['ms2_rt_sec']:.4f}\n")
            if int(record["precursor_charge"]) != 0:
                sign = "+" if int(record["precursor_charge"]) > 0 else "-"
                handle.write(f"CHARGE={abs(int(record['precursor_charge']))}{sign}\n")
            for mz, intensity in zip(record["mzs"], record["intensities"]):
                handle.write(f"{float(mz):.6f} {float(intensity):.6f}\n")
            handle.write("END IONS\n\n")
            index_rows.append(
                {
                    "mgf_index": scan,
                    "feature_id": feature_id,
                    "raw_entity_id": entity.raw_entity_id,
                    "mz": float(entity.mz),
                    "rt_sec": float(entity.rt_sec),
                    "representative_file": record["filename"],
                    "representative_native_id": record["native_id"],
                    "representative_n_peaks": record["n_peaks"],
                }
            )
    return pd.DataFrame(index_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-manifest",
        type=Path,
        default=Path("data/mtbls1066/pos_rp_primary_mzml/staged_manifest.tsv"),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("data/mtbls1066/pos_rp_primary_work_v1")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_mtbls1066_kd_response_v1_20261022"),
    )
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    parser.add_argument("--consensus-mz-ppm", type=float, default=5.0)
    parser.add_argument("--consensus-rt-sec", type=float, default=15.0)
    parser.add_argument("--ms2-link-ppm", type=float, default=10.0)
    parser.add_argument("--ms2-link-rt-sec", type=float, default=15.0)
    parser.add_argument("--minimum-units", type=int, default=3)
    parser.add_argument("--minimum-group-prevalence", type=float, default=0.75)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.staged_manifest)
    frames, extraction_audit = extract_or_reuse_features(
        manifest,
        args.work_dir,
        noise_threshold=args.noise_threshold,
        mass_error_ppm=args.mass_error_ppm,
        min_trace_sec=args.min_trace_sec,
        max_trace_sec=args.max_trace_sec,
    )
    metadata, intensities = build_consensus(
        frames, manifest, args.consensus_mz_ppm, args.consensus_rt_sec
    )
    responses = estimate_responses(
        intensities,
        metadata,
        manifest,
        args.minimum_units,
        args.minimum_group_prevalence,
    )
    links, representatives, ms2_file_audit = link_representative_ms2(
        metadata, manifest, args.ms2_link_ppm, args.ms2_link_rt_sec
    )

    args.output_dir.mkdir(parents=True)
    metadata["has_representative_ms2"] = metadata.feature_id.isin(representatives)
    metadata.to_csv(args.output_dir / "consensus_entities.tsv.gz", sep="\t", index=False)
    intensities.to_csv(args.output_dir / "discovery_intensity_matrix.tsv.gz", sep="\t", index=False)
    pd.DataFrame(extraction_audit).to_csv(
        args.output_dir / "feature_extraction_audit.tsv", sep="\t", index=False
    )
    links.to_csv(args.output_dir / "ms2_links.tsv.gz", sep="\t", index=False)
    ms2_file_audit.to_csv(args.output_dir / "ms2_file_audit.tsv", sep="\t", index=False)
    spectrum_index = write_mgf(
        args.output_dir / "representative_ms2.mgf", metadata, representatives
    )
    spectrum_index.to_csv(args.output_dir / "spectrum_index.tsv.gz", sep="\t", index=False)

    responses["has_representative_ms2"] = responses.feature_id.isin(representatives)
    responses.to_csv(args.output_dir / "all_response_effects.tsv.gz", sep="\t", index=False)
    eligible = responses[responses.qc_pass & responses.has_representative_ms2].copy()
    eligible.to_csv(args.output_dir / "ms2_linked_response_objects.tsv.gz", sep="\t", index=False)
    for contrast_id, subset in eligible.groupby("contrast_id", sort=True):
        subset.to_csv(args.output_dir / f"response__{contrast_id}.tsv.gz", sep="\t", index=False)

    response_counts = {
        contrast_id: {
            "eligible_all_consensus": int(
                responses.loc[responses.contrast_id.eq(contrast_id), "qc_pass"].sum()
            ),
            "eligible_ms2_linked": int((eligible.contrast_id == contrast_id).sum()),
        }
        for contrast_id in CONTRASTS
    }
    report = {
        "schema": "contrast_atlas_mtbls1066_ketogenic_response_v1",
        "status": "development_response_objects_complete",
        "evaluation_role": "development_consumed",
        "leakage_status": (
            "feature discovery, consensus construction and MS2 linkage were phenotype blind; "
            "group labels entered only during response estimation"
        ),
        "denominators": {
            "biological_units": len(manifest),
            "units_by_group": manifest.groupby("group").size().astype(int).to_dict(),
            "consensus_entities": len(metadata),
            "entities_with_representative_ms2": len(representatives),
            "raw_ms2_links": len(links),
            "responses": response_counts,
        },
        "parameters": {
            "noise_threshold": args.noise_threshold,
            "mass_error_ppm": args.mass_error_ppm,
            "min_trace_sec": args.min_trace_sec,
            "max_trace_sec": args.max_trace_sec,
            "consensus_mz_ppm": args.consensus_mz_ppm,
            "consensus_rt_sec": args.consensus_rt_sec,
            "ms2_link_ppm": args.ms2_link_ppm,
            "ms2_link_rt_sec": args.ms2_link_rt_sec,
            "minimum_units": args.minimum_units,
            "minimum_group_prevalence": args.minimum_group_prevalence,
        },
        "provenance": {
            "staged_manifest": str(args.staged_manifest),
            "staged_manifest_sha256": sha256(args.staged_manifest),
            "representative_mgf_sha256": sha256(args.output_dir / "representative_ms2.mgf"),
        },
        "claim_boundary": [
            "These are within-source directed response objects, not a successful cross-study retrieval.",
            "Precursor/RT-linked DDA spectra are chemical evidence, not unique molecular identities.",
            "Consensus LC-MS features and redundant adduct/isotope ions are not counted as independent metabolites; ion-family control is required before entity-level biological claims.",
            "identity_status remains unpartitioned; known-only or unknown-increment claims are forbidden until a complete frozen library search is applied.",
            "Discovery intensities retain missingness and have not undergone targeted EIC gap filling; confirmatory analyses must re-quantify frozen targets uniformly.",
            "The secondary contrast is conditional on xenograft and cyclophosphamide background and is not a pure diet effect.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
