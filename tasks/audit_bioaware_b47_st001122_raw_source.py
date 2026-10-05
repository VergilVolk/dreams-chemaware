#!/usr/bin/env python
"""Truth-blind readiness audit for the first B47 external raw source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile


EXPECTED_RAW_MD5 = "eedbf0071f945b4d6e01b4b836d3f5ca"
EXPECTED_EXTERNAL_SHA256 = (
    "2d6ed0ae51b385b99025398e62939d8902096ba798e603dc5bd0acc8c9b28898"
)
FEATURE_SUFFIX = "Oliver Fiehn/ST001122_urine/asari/preferred_Feature_table.tsv"


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def unique_suffix(archive: ZipFile, suffix: str) -> str:
    matches = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and info.filename.replace("\\", "/").casefold().endswith(suffix.casefold())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member ending {suffix!r}; observed {matches}")
    return matches[0]


def read_feature_samples(archive: ZipFile, member: str) -> tuple[list[str], int]:
    with archive.open(member) as handle:
        header = handle.readline(1024 * 1024).decode("utf-8-sig").rstrip("\r\n")
    columns = next(csv.reader([header], delimiter="\t"))
    fixed = {
        "id_number", "mz", "rtime", "rtime_left_base", "rtime_right_base",
        "parent_masstrack_id", "peak_area", "cSelectivity", "goodness_fitting",
        "snr", "detection_counts",
    }
    samples = [column for column in columns if column not in fixed]
    return samples, len(columns)


def count_mgf(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(line.strip().upper() == b"BEGIN IONS" for line in handle)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--external-archive", type=Path, required=True)
    parser.add_argument("--probe-raw", type=Path, required=True)
    parser.add_argument("--probe-mgf", type=Path, required=True)
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    paths = {
        key: value.resolve()
        for key, value in {
            "raw_archive": args.raw_archive,
            "external_archive": args.external_archive,
            "probe_raw": args.probe_raw,
            "probe_mgf": args.probe_mgf,
            "parser": args.parser,
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    raw_md5 = digest(paths["raw_archive"], "md5")
    if raw_md5 != EXPECTED_RAW_MD5:
        raise RuntimeError(f"ST001122 raw archive MD5 mismatch: {raw_md5}")
    external_sha256 = digest(paths["external_archive"])
    if external_sha256 != EXPECTED_EXTERNAL_SHA256:
        raise RuntimeError(f"external archive SHA256 mismatch: {external_sha256}")

    with ZipFile(paths["raw_archive"]) as raw_archive:
        raw_members = sorted(
            info.filename
            for info in raw_archive.infolist()
            if not info.is_dir() and info.filename.casefold().endswith(".raw")
        )
        raw_samples = [Path(member).stem for member in raw_members]
        probe_member = unique_suffix(raw_archive, paths["probe_raw"].name)
        probe_member_sha256 = hashlib.sha256(raw_archive.read(probe_member)).hexdigest()
    with ZipFile(paths["external_archive"]) as external_archive:
        feature_member = unique_suffix(external_archive, FEATURE_SUFFIX)
        feature_samples, feature_columns = read_feature_samples(external_archive, feature_member)

    missing_raw = sorted(set(feature_samples) - set(raw_samples))
    extra_raw = sorted(set(raw_samples) - set(feature_samples))
    duplicate_raw_basenames = len(raw_samples) - len(set(raw_samples))
    ms2_spectra = count_mgf(paths["probe_mgf"])
    gates = {
        "deposited_raw_md5_matches": raw_md5 == EXPECTED_RAW_MD5,
        "raw_file_count_matches_feature_sample_count": len(raw_samples) == len(feature_samples),
        "all_feature_samples_have_exact_raw_basename": not missing_raw,
        "no_extra_raw_samples": not extra_raw,
        "no_duplicate_raw_basenames": duplicate_raw_basenames == 0,
        "probe_raw_is_exact_archive_member": digest(paths["probe_raw"]) == probe_member_sha256,
        "probe_contains_ms2": ms2_spectra > 0,
        "truth_payloads_opened": False,
        "algorithm_output_payloads_opened": False,
    }
    scientific_gates = {
        key: value
        for key, value in gates.items()
        if key not in {"truth_payloads_opened", "algorithm_output_payloads_opened"}
    }
    scientific_gates["truth_payloads_not_opened"] = not gates["truth_payloads_opened"]
    scientific_gates["algorithm_output_payloads_not_opened"] = not gates[
        "algorithm_output_payloads_opened"
    ]
    report = {
        "status": "bioaware_b47_st001122_raw_source_readiness_complete",
        "formal": True,
        "study": "ST001122",
        "raw_files": len(raw_members),
        "feature_table_columns": feature_columns,
        "feature_table_sample_columns": len(feature_samples),
        "exact_raw_feature_sample_matches": len(set(raw_samples) & set(feature_samples)),
        "missing_raw_samples": missing_raw,
        "extra_raw_samples": extra_raw,
        "probe": {
            "raw_member": probe_member,
            "raw_sha256": digest(paths["probe_raw"]),
            "mgf_sha256": digest(paths["probe_mgf"]),
            "ms2_spectra": ms2_spectra,
            "parser_sha256": digest(paths["parser"]),
        },
        "gates": scientific_gates,
        "pass_to_full_truth_blind_ms2_conversion": all(scientific_gates.values()),
        "next_stage": (
            "Convert all 43 raw files to standardized MS2, join spectra to feature rows by "
            "sample plus precursor m/z plus RT, and report join coverage before sealed truth is opened."
        ),
        "claim_limit": (
            "This establishes raw-data and join-namespace feasibility. It does not establish "
            "validated query count, candidate ranking, reaction evidence, or BioAware gain."
        ),
        "provenance": {
            key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
