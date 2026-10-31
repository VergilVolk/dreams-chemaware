#!/usr/bin/env python3
"""Query a frozen public fecal-background MGF for PRISM ion-family evidence.

This is a spectrum-only validation.  A precursor hit or fragment similarity in
MSV000079651 is evidence that a related spectrum occurs in an independent
public fecal-background asset.  It is not standard confirmation, participant-
level response replication, or proof of a shared molecular identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
from pathlib import Path


TARGET_MASSES = (371.258, 389.269, 595.349, 597.362, 617.330, 817.581, 834.609, 839.563)
PAIR_TESTS = (
    ("mz371_389", 371.258, 389.269),
    ("mz595_617", 595.349, 617.330),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_mgf(content: bytes) -> list[dict]:
    spectra: list[dict] = []
    inside = False
    metadata: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    for raw in io.StringIO(content.decode("utf-8", errors="replace")):
        line = raw.strip()
        if line == "BEGIN IONS":
            inside, metadata, peaks = True, {}, []
        elif inside and line == "END IONS":
            if "PEPMASS" in metadata:
                spectra.append({"metadata": metadata.copy(), "peaks": peaks[:]})
            inside = False
        elif inside and "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
        elif inside and line:
            fields = line.split()
            if len(fields) >= 2:
                try:
                    peaks.append((float(fields[0]), float(fields[1])))
                except ValueError:
                    pass
    return spectra


def precursor_mz(spectrum: dict) -> float:
    return float(spectrum["metadata"]["PEPMASS"].split()[0])


def raw_cosine(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    tolerance: float,
) -> tuple[float, int]:
    candidates: list[tuple[float, int, int, float]] = []
    for left_index, (left_mz, left_intensity) in enumerate(left):
        for right_index, (right_mz, right_intensity) in enumerate(right):
            delta = abs(left_mz - right_mz)
            if delta <= tolerance:
                candidates.append(
                    (delta, left_index, right_index, left_intensity * right_intensity)
                )
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    dot = 0.0
    matched = 0
    for _, left_index, right_index, product in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        dot += product
        matched += 1
    left_norm = math.sqrt(sum(intensity * intensity for _, intensity in left))
    right_norm = math.sqrt(sum(intensity * intensity for _, intensity in right))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("zero-intensity public spectrum")
    return dot / (left_norm * right_norm), matched


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        args.mgf_url,
        headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0 public-evidence audit"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
        content = response.read()
        final_url = response.geturl()
    spectra = parse_mgf(content)
    if not spectra:
        raise ValueError("public MGF contained no spectra")

    hit_rows: list[dict] = []
    best_hits: dict[float, dict] = {}
    for target in TARGET_MASSES:
        matches: list[tuple[float, dict]] = []
        for spectrum in spectra:
            observed = precursor_mz(spectrum)
            ppm = 1e6 * (observed - target) / target
            if abs(ppm) <= args.precursor_tolerance_ppm:
                matches.append((abs(ppm), spectrum))
                hit_rows.append(
                    {
                        "target_mz": target,
                        "observed_precursor_mz": observed,
                        "signed_ppm_error": ppm,
                        "scan": spectrum["metadata"].get("SCANS", ""),
                        "charge": spectrum["metadata"].get("CHARGE", ""),
                        "peak_count": len(spectrum["peaks"]),
                        "claim_boundary": (
                            "public_precursor_and_msms_occurrence_not_identity_or_response"
                        ),
                    }
                )
        if matches:
            best_hits[target] = min(matches, key=lambda item: item[0])[1]

    pair_rows: list[dict] = []
    for group_id, left_target, right_target in PAIR_TESTS:
        if left_target not in best_hits or right_target not in best_hits:
            pair_rows.append(
                {
                    "group_id": group_id,
                    "left_target_mz": left_target,
                    "right_target_mz": right_target,
                    "left_scan": best_hits.get(left_target, {}).get("metadata", {}).get(
                        "SCANS", ""
                    ),
                    "right_scan": best_hits.get(right_target, {}).get("metadata", {}).get(
                        "SCANS", ""
                    ),
                    "matched_peaks": "",
                    "raw_cosine": "",
                    "evidence_class": "one_or_both_public_spectra_absent",
                    "claim_boundary": "no_family_inference",
                }
            )
            continue
        left = best_hits[left_target]
        right = best_hits[right_target]
        cosine, matched = raw_cosine(
            left["peaks"], right["peaks"], args.fragment_tolerance_da
        )
        evidence_class = (
            "strong_public_fragment_relation"
            if cosine >= 0.8 and matched >= 6
            else "public_spectra_present_but_relation_not_supported"
        )
        pair_rows.append(
            {
                "group_id": group_id,
                "left_target_mz": left_target,
                "right_target_mz": right_target,
                "left_scan": left["metadata"].get("SCANS", ""),
                "right_scan": right["metadata"].get("SCANS", ""),
                "matched_peaks": matched,
                "raw_cosine": cosine,
                "evidence_class": evidence_class,
                "claim_boundary": (
                    "public_fragment_relation_not_neutral_identity_or_standard_confirmation"
                ),
            }
        )

    hit_path = args.output_dir / "public_precursor_msms_hits.tsv"
    pair_path = args.output_dir / "public_fragment_pair_evidence.tsv"
    write_tsv(
        hit_path,
        hit_rows,
        [
            "target_mz",
            "observed_precursor_mz",
            "signed_ppm_error",
            "scan",
            "charge",
            "peak_count",
            "claim_boundary",
        ],
    )
    write_tsv(pair_path, pair_rows, list(pair_rows[0]))

    report = {
        "schema": "contrast_atlas_prism_public_mgf_family_evidence_v1",
        "data_role": "public_development_consumed_spectrum_only_evidence",
        "public_asset": {
            "requested_url": args.mgf_url,
            "resolved_url": final_url,
            "bytes": len(content),
            "sha256": sha256_bytes(content),
            "spectrum_count": len(spectra),
            "dataset": "MSV000079651",
            "context": "fecal-background concentration-series spectral asset",
        },
        "parameters": {
            "precursor_tolerance_ppm": args.precursor_tolerance_ppm,
            "fragment_tolerance_da": args.fragment_tolerance_da,
            "analog_search": False,
        },
        "targets": list(TARGET_MASSES),
        "targets_with_public_msms": sorted(best_hits),
        "target_hit_counts": {
            str(target): sum(float(row["target_mz"]) == target for row in hit_rows)
            for target in TARGET_MASSES
        },
        "pair_evidence": {row["group_id"]: row for row in pair_rows},
        "claim_boundary": (
            "This public spectrum-only asset has no linked IBD response for this analysis. "
            "Hits do not establish identity, MSI level 1, or biological replication."
        ),
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            "precursor_msms_hits": hit_path.name,
            "fragment_pair_evidence": pair_path.name,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths = [hit_path, pair_path, report_path]
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in output_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
