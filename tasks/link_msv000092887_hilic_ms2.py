#!/usr/bin/env python3
"""Phenotype-blind raw-MS2 linkage for the 3,041 MSV000092887 features.

Only feature m/z, retention time, existing reference-spectrum evidence, and the
phenotype-blind acquisition manifest are loaded while links are built.  COVID
labels and response effects are joined only after the complete link table and
representative spectra have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


BIOLOGICAL_ASSAY = re.compile(r"^\d+[A-Za-z]+P$")
QC_ASSAY = re.compile(r"^Q\d+$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assay_role(name: str) -> str:
    if BIOLOGICAL_ASSAY.fullmatch(name):
        return "biological"
    if QC_ASSAY.fullmatch(name):
        return "pooled_qc"
    return "blank"


def load_feature_coordinates(path: Path) -> pd.DataFrame:
    columns = [
        "entity_id", "precursor_mz", "retention_time_seconds",
        "reference_spectrum_evidence",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=columns, dtype={"entity_id": str})
    frame["precursor_mz"] = pd.to_numeric(frame.precursor_mz, errors="coerce")
    frame["retention_time_seconds"] = pd.to_numeric(
        frame.retention_time_seconds, errors="coerce"
    )
    evidence = frame["reference_spectrum_evidence"]
    if not pd.api.types.is_bool_dtype(evidence):
        normalized = evidence.astype(str).str.strip().str.lower()
        unexpected = ~normalized.isin({"true", "false", "1", "0"})
        if unexpected.any():
            raise RuntimeError("reference_spectrum_evidence is not a strict boolean field")
        frame["reference_spectrum_evidence"] = normalized.isin({"true", "1"})
    if len(frame) != 3041 or frame.entity_id.duplicated().any():
        raise RuntimeError("expected 3,041 unique frozen SMF feature IDs")
    if not np.isfinite(frame[["precursor_mz", "retention_time_seconds"]]).all().all():
        raise RuntimeError("non-finite feature coordinates")
    return frame.reset_index(drop=True)


def load_staged_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"assay_name", "expected_bytes", "local_path", "sha256"}
    if missing := required - set(frame):
        raise ValueError(f"staged manifest lacks {sorted(missing)}")
    if len(frame) != 129 or frame.assay_name.duplicated().any():
        raise RuntimeError("expected 129 unique HILIC-positive assays")
    frame["role"] = frame.assay_name.map(assay_role)
    if frame.role.value_counts().to_dict() != {
        "biological": 78, "blank": 40, "pooled_qc": 11
    }:
        raise RuntimeError(f"unexpected assay-role topology: {frame.role.value_counts().to_dict()}")
    for row in frame.itertuples(index=False):
        local = Path(row.local_path)
        if not local.is_file() or local.stat().st_size != int(row.expected_bytes):
            raise FileNotFoundError(f"missing or wrong-size staged file: {local}")
    return frame.sort_values("assay_name").reset_index(drop=True)


def load_ms2(path: Path) -> oms.MSExperiment:
    experiment = oms.MSExperiment()
    loader = oms.MzXMLFile()
    options = loader.getOptions()
    options.setMSLevels([2])
    loader.setOptions(options)
    loader.load(str(path), experiment)
    return experiment


def freeze_links(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    mass_tolerance_da: float,
    rt_tolerance_seconds: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], pd.DataFrame]:
    feature_mz = features.precursor_mz.to_numpy(float)
    feature_rt = features.retention_time_seconds.to_numpy(float)
    order = np.argsort(feature_mz)
    sorted_mz = feature_mz[order]
    records: list[dict[str, object]] = []
    representatives: dict[str, dict[str, object]] = {}
    file_rows: list[dict[str, object]] = []

    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        experiment = load_ms2(Path(row.local_path))
        matched = unique = ambiguous = 0
        for spectrum in experiment:
            precursors = spectrum.getPrecursors()
            if not precursors:
                continue
            precursor = precursors[0]
            precursor_mz = float(precursor.getMZ())
            rt_sec = float(spectrum.getRT())
            left = int(np.searchsorted(
                sorted_mz, precursor_mz - mass_tolerance_da, side="left"
            ))
            right = int(np.searchsorted(
                sorted_mz, precursor_mz + mass_tolerance_da, side="right"
            ))
            candidates = order[left:right]
            candidates = candidates[
                np.abs(feature_rt[candidates] - rt_sec) <= rt_tolerance_seconds
            ]
            if not len(candidates):
                continue
            mz_error = feature_mz[candidates] - precursor_mz
            rt_error = feature_rt[candidates] - rt_sec
            distance = (
                np.abs(mz_error) / mass_tolerance_da
                + np.abs(rt_error) / rt_tolerance_seconds
            )
            selected_position = int(np.argmin(distance))
            selected = int(candidates[selected_position])
            selected_id = str(features.iloc[selected].entity_id)
            candidate_ids = [str(features.iloc[int(index)].entity_id) for index in candidates]
            mzs, intensities = spectrum.get_peaks()
            mzs = np.asarray(mzs, dtype=float)
            intensities = np.asarray(intensities, dtype=float)
            if len(mzs) < 3 or not np.isfinite(intensities).all():
                continue
            candidate_count = int(len(candidates))
            is_unique = candidate_count == 1
            matched += 1
            unique += int(is_unique)
            ambiguous += int(not is_unique)
            audit = {
                "assay_name": row.assay_name,
                "assay_role": row.role,
                "native_id": spectrum.getNativeID(),
                "precursor_mz": precursor_mz,
                "ms2_rt_sec": rt_sec,
                "candidate_count": candidate_count,
                "candidate_feature_ids": "|".join(candidate_ids),
                "nearest_feature_id": selected_id,
                "nearest_mz_error_da": float(mz_error[selected_position]),
                "nearest_rt_error_sec": float(rt_error[selected_position]),
                "unique_feature_link": is_unique,
                "n_fragment_peaks": int(len(mzs)),
                "fragment_tic": float(np.sum(intensities)),
            }
            records.append(audit)
            if not is_unique:
                continue
            role_priority = {"blank": 0, "pooled_qc": 1, "biological": 2}[row.role]
            ranking = (
                role_priority,
                int(len(mzs)),
                float(np.sum(intensities)),
                -float(distance[selected_position]),
            )
            current = representatives.get(selected_id)
            if current is None or ranking > current["ranking"]:
                representatives[selected_id] = {
                    **audit,
                    "precursor_charge": int(precursor.getCharge()),
                    "mzs": mzs,
                    "intensities": intensities,
                    "ranking": ranking,
                }
        file_rows.append(
            {
                "assay_name": row.assay_name,
                "assay_role": row.role,
                "ms2_spectra": int(experiment.size()),
                "matched_spectra": matched,
                "unique_link_spectra": unique,
                "ambiguous_link_spectra": ambiguous,
            }
        )
        print(
            f"[link {number}/{len(manifest)}] {row.assay_name}: "
            f"MS2={experiment.size()} matched={matched} unique={unique} ambiguous={ambiguous}",
            flush=True,
        )
    return pd.DataFrame(records), representatives, pd.DataFrame(file_rows)


def summarize_features(
    features: pd.DataFrame,
    links: pd.DataFrame,
    representatives: dict[str, dict[str, object]],
) -> pd.DataFrame:
    result = features.copy()
    if links.empty:
        raise RuntimeError("no raw MS2 spectrum matched the frozen feature universe")
    unique = links.loc[links.unique_feature_link].copy()
    for role in ("biological", "pooled_qc", "blank"):
        counts = unique.loc[unique.assay_role.eq(role)].groupby("nearest_feature_id").size()
        assays = unique.loc[unique.assay_role.eq(role)].groupby("nearest_feature_id")["assay_name"].nunique()
        result[f"unique_ms2_spectra__{role}"] = result.entity_id.map(counts).fillna(0).astype(int)
        result[f"unique_ms2_assays__{role}"] = result.entity_id.map(assays).fillna(0).astype(int)
    ambiguous = links.loc[~links.unique_feature_link].copy()
    ambiguous_members = Counter(
        feature_id
        for value in ambiguous.candidate_feature_ids.astype(str)
        for feature_id in value.split("|") if feature_id
    )
    result["ambiguous_candidate_spectra"] = result.entity_id.map(ambiguous_members).fillna(0).astype(int)
    result["has_unique_raw_ms2"] = result.entity_id.isin(representatives)
    result["has_usable_raw_ms2"] = (
        result["unique_ms2_spectra__biological"].gt(0)
        | result["unique_ms2_spectra__pooled_qc"].gt(0)
    )
    result["blank_only_ms2"] = (
        result["unique_ms2_spectra__blank"].gt(0)
        & ~result["has_usable_raw_ms2"]
    )
    result["source_local_ms2_partition"] = np.select(
        [
            result.has_usable_raw_ms2 & result.reference_spectrum_evidence.astype(bool),
            result.has_usable_raw_ms2 & ~result.reference_spectrum_evidence.astype(bool),
            result.blank_only_ms2,
        ],
        ["library_supported", "unresolved_source_local", "blank_only"],
        default="no_unique_raw_ms2",
    )
    return result


def write_mgf(
    path: Path,
    feature_summary: pd.DataFrame,
    representatives: dict[str, dict[str, object]],
) -> pd.DataFrame:
    usable = feature_summary.loc[feature_summary.has_usable_raw_ms2].set_index("entity_id")
    rows: list[dict[str, object]] = []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, entity_id in enumerate(sorted(usable.index.astype(str)), start=1):
            record = representatives[entity_id]
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE={entity_id}\n")
            handle.write(f"SCANS={entity_id}\n")
            handle.write(f"PEPMASS={record['precursor_mz']:.8f}\n")
            handle.write(f"RTINSECONDS={record['ms2_rt_sec']:.4f}\n")
            charge = int(record["precursor_charge"])
            if charge:
                handle.write(f"CHARGE={abs(charge)}{'+' if charge > 0 else '-'}\n")
            for mz, intensity in zip(record["mzs"], record["intensities"]):
                handle.write(f"{float(mz):.6f} {float(intensity):.6f}\n")
            handle.write("END IONS\n\n")
            rows.append(
                {
                    "mgf_index": index,
                    "entity_id": entity_id,
                    "representative_assay": record["assay_name"],
                    "representative_assay_role": record["assay_role"],
                    "representative_native_id": record["native_id"],
                    "representative_n_fragment_peaks": record["n_fragment_peaks"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response-object",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_msv000092887_paired_response_v1_20261019/"
            "paired_response_object.tsv.gz"
        ),
    )
    parser.add_argument(
        "--staged-manifest",
        type=Path,
        default=Path("data/msv000092887/hilic_positive_mzxml/staged_manifest.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_msv000092887_raw_ms2_link_v1_20261022"),
    )
    parser.add_argument("--mass-tolerance-da", type=float, default=0.005)
    parser.add_argument("--rt-tolerance-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.mass_tolerance_da <= 0 or args.rt_tolerance_seconds <= 0:
        raise ValueError("linkage tolerances must be positive")

    features = load_feature_coordinates(args.response_object)
    manifest = load_staged_manifest(args.staged_manifest)
    links, representatives, file_audit = freeze_links(
        features,
        manifest,
        mass_tolerance_da=args.mass_tolerance_da,
        rt_tolerance_seconds=args.rt_tolerance_seconds,
    )
    summary = summarize_features(features, links, representatives)

    args.output_dir.mkdir(parents=True)
    links_path = args.output_dir / "spectrum_link_audit.tsv.gz"
    links.to_csv(links_path, sep="\t", index=False, compression="gzip")
    file_audit.to_csv(args.output_dir / "file_ms2_audit.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "feature_ms2_membership.tsv.gz", sep="\t", index=False)
    mgf_path = args.output_dir / "representative_ms2.mgf"
    mgf_index = write_mgf(mgf_path, summary, representatives)
    mgf_index.to_csv(args.output_dir / "representative_ms2_index.tsv", sep="\t", index=False)

    # Outcome and response columns are opened only after the phenotype-blind
    # spectrum link artifacts above have been frozen.
    response = pd.read_csv(args.response_object, sep="\t", dtype={"entity_id": str})
    joined = response.merge(
        summary.drop(columns=["precursor_mz", "retention_time_seconds", "reference_spectrum_evidence"]),
        on="entity_id", how="left", validate="one_to_one",
    )
    joined.to_csv(
        args.output_dir / "paired_response_with_raw_ms2.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )

    usable = summary.has_usable_raw_ms2
    unresolved = summary.source_local_ms2_partition.eq("unresolved_source_local")
    reference = summary.reference_spectrum_evidence.astype(bool)
    report = {
        "schema": "contrast_atlas_msv000092887_raw_ms2_link_v1",
        "status": "PHENOTYPE_BLIND_RAW_MS2_LINK_COMPLETE",
        "evidence_role": "development_consumed",
        "leakage_status": (
            "feature links were frozen using only m/z, RT, existing reference-spectrum status, "
            "and a phenotype-blind acquisition manifest; response effects were joined afterward"
        ),
        "denominators": {
            "features": len(summary),
            "mzxml_files": len(manifest),
            "biological_files": int(manifest.role.eq("biological").sum()),
            "pooled_qc_files": int(manifest.role.eq("pooled_qc").sum()),
            "blank_files": int(manifest.role.eq("blank").sum()),
            "raw_ms2_spectra": int(file_audit.ms2_spectra.sum()),
            "matched_spectra": int(file_audit.matched_spectra.sum()),
            "unique_link_spectra": int(file_audit.unique_link_spectra.sum()),
            "ambiguous_link_spectra": int(file_audit.ambiguous_link_spectra.sum()),
            "features_with_any_unique_raw_ms2": int(summary.has_unique_raw_ms2.sum()),
            "features_with_usable_raw_ms2": int(usable.sum()),
            "features_blank_only": int(summary.blank_only_ms2.sum()),
            "reference_spectrum_features_with_usable_raw_ms2": int((reference & usable).sum()),
            "source_local_unresolved_features_with_usable_raw_ms2": int((unresolved & usable).sum()),
            "abundance_qc_pass_with_usable_raw_ms2": int(
                (joined.abundance_qc_pass.astype(bool) & joined.has_usable_raw_ms2.astype(bool)).sum()
            ),
            "detection_qc_pass_with_usable_raw_ms2": int(
                (joined.detection_qc_pass.astype(bool) & joined.has_usable_raw_ms2.astype(bool)).sum()
            ),
        },
        "published_aggregate_reference": {
            "reported_features_with_ms2": 1982,
            "reported_features_with_reference_spectrum": 175,
            "observed_usable_raw_ms2_minus_reported": int(usable.sum()) - 1982,
            "interpretation": (
                "The published count is an audit reference, not a target to tune toward; raw linkage "
                "may differ because this implementation requires unique 0.005-Da/3-s feature assignment."
            ),
        },
        "parameters": {
            "mass_tolerance_da": args.mass_tolerance_da,
            "rt_tolerance_seconds": args.rt_tolerance_seconds,
            "ambiguous_spectrum_policy": "not assigned to any feature",
            "usable_ms2": "at least one unique biological or pooled-QC link; blank-only excluded",
        },
        "provenance": {
            "response_object_sha256": sha256(args.response_object),
            "staged_manifest_sha256": sha256(args.staged_manifest),
            "link_table_sha256_before_response_join": sha256(links_path),
            "representative_mgf_sha256": sha256(mgf_path),
        },
        "claim_boundary": [
            "Raw precursor/RT linkage establishes feature-level MS2 evidence, not molecular identity.",
            "unresolved_source_local means no deposited reference-spectrum hit in this source; it is not yet a harmonized cross-source unknown partition.",
            "This result alone is not a replicated COVID response and not an unknown-information increment.",
            "Ambiguous multi-feature spectra and blank-only links are excluded from usable representative spectra.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
