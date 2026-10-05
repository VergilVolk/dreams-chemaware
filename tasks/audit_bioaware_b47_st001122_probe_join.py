#!/usr/bin/env python
"""Truth-blind precursor/RT join audit for one ST001122 raw sample."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile


FEATURE_SUFFIX = "Oliver Fiehn/ST001122_urine/asari/preferred_Feature_table.tsv"
EXPECTED_EXTERNAL_SHA256 = (
    "2d6ed0ae51b385b99025398e62939d8902096ba798e603dc5bd0acc8c9b28898"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def unique_suffix(archive: ZipFile, suffix: str) -> str:
    matches = [
        item.filename
        for item in archive.infolist()
        if not item.is_dir() and item.filename.replace("\\", "/").casefold().endswith(suffix.casefold())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one feature-table member; observed {matches}")
    return matches[0]


def finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_features(archive: ZipFile, member: str, sample: str) -> list[dict]:
    with archive.open(member) as binary:
        rows = csv.DictReader(
            (line.decode("utf-8-sig") for line in binary), delimiter="\t"
        )
        if rows.fieldnames is None or sample not in rows.fieldnames:
            raise RuntimeError(f"sample {sample!r} absent from feature table")
        features = []
        for row in rows:
            mz = finite_float(row.get("mz", ""))
            rt = finite_float(row.get("rtime", ""))
            left = finite_float(row.get("rtime_left_base", ""))
            right = finite_float(row.get("rtime_right_base", ""))
            abundance = finite_float(row.get(sample, ""))
            if None in {mz, rt, left, right, abundance}:
                continue
            features.append(
                {
                    "id": row.get("id_number", ""),
                    "mz": float(mz),
                    "rt": float(rt),
                    "left": float(left),
                    "right": float(right),
                    "abundance": float(abundance),
                }
            )
    return features


def read_mgf_headers(path: Path) -> list[dict]:
    spectra = []
    current = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            upper = line.upper()
            if upper == "BEGIN IONS":
                current = {"scan": None, "rt": None, "precursor_mz": None}
            elif upper == "END IONS":
                if current is not None:
                    spectra.append(current)
                current = None
            elif current is not None and "=" in line:
                key, value = line.split("=", 1)
                key = key.upper()
                if key == "SCANS":
                    current["scan"] = value
                elif key == "RTINSECONDS":
                    current["rt"] = finite_float(value)
                elif key == "PEPMASS":
                    current["precursor_mz"] = finite_float(value.split()[0])
    return spectra


def join_summary(features: list[dict], spectra: list[dict], ppm: float) -> dict:
    detected = sorted((feature for feature in features if feature["abundance"] > 0), key=lambda x: x["mz"])
    masses = [feature["mz"] for feature in detected]
    complete = [
        spectrum
        for spectrum in spectra
        if spectrum["rt"] is not None and spectrum["precursor_mz"] is not None
    ]
    candidate_counts = []
    matched_ids = set()
    for spectrum in complete:
        precursor = spectrum["precursor_mz"]
        tolerance = precursor * ppm * 1e-6
        start = bisect_left(masses, precursor - tolerance)
        stop = bisect_right(masses, precursor + tolerance)
        matches = [
            feature
            for feature in detected[start:stop]
            if feature["left"] <= spectrum["rt"] <= feature["right"]
        ]
        candidate_counts.append(len(matches))
        matched_ids.update(feature["id"] for feature in matches)
    matched = sum(value > 0 for value in candidate_counts)
    unique = sum(value == 1 for value in candidate_counts)
    ambiguous = sum(value > 1 for value in candidate_counts)
    return {
        "ppm": ppm,
        "feature_peak_boundary_required": True,
        "sample_abundance_positive_required": True,
        "detected_features": len(detected),
        "ms2_spectra": len(spectra),
        "ms2_with_precursor_and_rt": len(complete),
        "matched_ms2": matched,
        "matched_fraction": matched / len(complete) if complete else 0.0,
        "uniquely_matched_ms2": unique,
        "unique_match_fraction": unique / len(complete) if complete else 0.0,
        "ambiguously_matched_ms2": ambiguous,
        "matched_feature_ids": len(matched_ids),
    }


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
    parser.add_argument("--external-archive", type=Path, required=True)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    external = args.external_archive.resolve()
    mgf = args.mgf.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    if digest(external) != EXPECTED_EXTERNAL_SHA256:
        raise RuntimeError("external archive SHA256 mismatch")
    with ZipFile(external) as archive:
        member = unique_suffix(archive, FEATURE_SUFFIX)
        features = read_features(archive, member, args.sample)
    spectra = read_mgf_headers(mgf)
    summaries = {str(ppm): join_summary(features, spectra, ppm) for ppm in (5.0, 10.0)}
    primary = summaries["10.0"]
    gates = {
        "ms2_spectra_ge_1000": primary["ms2_spectra"] >= 1000,
        "complete_precursor_rt_ge_99pct": (
            primary["ms2_with_precursor_and_rt"] >= 0.99 * primary["ms2_spectra"]
        ),
        "matched_ms2_ge_100": primary["matched_ms2"] >= 100,
        "matched_feature_ids_ge_50": primary["matched_feature_ids"] >= 50,
        "truth_payloads_opened": False,
        "algorithm_output_payloads_opened": False,
    }
    scientific = {
        key: value
        for key, value in gates.items()
        if key not in {"truth_payloads_opened", "algorithm_output_payloads_opened"}
    }
    scientific["truth_payloads_not_opened"] = not gates["truth_payloads_opened"]
    scientific["algorithm_output_payloads_not_opened"] = not gates[
        "algorithm_output_payloads_opened"
    ]
    report = {
        "status": "bioaware_b47_st001122_probe_join_complete",
        "formal": True,
        "sample": args.sample,
        "feature_rows_with_complete_coordinates": len(features),
        "join_sensitivity": summaries,
        "primary_join": "10 ppm plus empirical feature peak boundary plus positive sample abundance",
        "gates": scientific,
        "pass_to_full_source_conversion_and_join": all(scientific.values()),
        "next_stage": (
            "Run the frozen join on all 43 samples and quantify independent feature-level MS2 "
            "coverage before any validation identity is read."
        ),
        "claim_limit": (
            "A raw-to-feature namespace join feasibility result only; no truth or annotation "
            "performance is used."
        ),
        "provenance": {
            "external_archive_sha256": digest(external),
            "mgf_sha256": digest(mgf),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
