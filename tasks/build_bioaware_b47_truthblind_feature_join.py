#!/usr/bin/env python
"""Build a truth-blind MS2-to-MS1 feature join for one B47 external study."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from zipfile import ZipFile


EXPECTED_EXTERNAL_SHA256 = (
    "2d6ed0ae51b385b99025398e62939d8902096ba798e603dc5bd0acc8c9b28898"
)
STUDIES = {
    "ST001122": "Oliver Fiehn/ST001122_urine/asari/preferred_Feature_table.tsv",
    "ST003356": "CZ Biohub/ST003356_urine/asari/HILICPOS/preferred_Feature_table.tsv",
}
FEATURE_METADATA = {
    "id_number", "mz", "rtime", "rtime_left_base", "rtime_right_base",
    "parent_masstrack_id", "peak_area", "cSelectivity", "goodness_fitting",
    "snr", "detection_counts",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def finite(value: str | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def unique_suffix(archive: ZipFile, suffix: str) -> str:
    matches = [
        item.filename
        for item in archive.infolist()
        if not item.is_dir()
        and item.filename.replace("\\", "/").casefold().endswith(suffix.casefold())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member ending {suffix!r}: {matches}")
    return matches[0]


def read_detected_features(
    archive: ZipFile, member: str, samples: list[str]
) -> tuple[dict[str, list[dict]], int]:
    result = {sample: [] for sample in samples}
    complete_rows = 0
    with archive.open(member) as binary:
        rows = csv.DictReader(
            (line.decode("utf-8-sig") for line in binary), delimiter="\t"
        )
        if rows.fieldnames is None:
            raise RuntimeError("feature table has no header")
        missing = sorted(set(samples) - set(rows.fieldnames))
        unexpected = sorted(
            set(rows.fieldnames) - FEATURE_METADATA - set(samples)
        )
        if missing or unexpected:
            raise RuntimeError(
                f"feature sample namespace mismatch missing={missing} unexpected={unexpected}"
            )
        for row in rows:
            mz = finite(row.get("mz"))
            rt = finite(row.get("rtime"))
            left = finite(row.get("rtime_left_base"))
            right = finite(row.get("rtime_right_base"))
            feature_id = str(row.get("id_number", "")).strip()
            if None in {mz, rt, left, right} or not feature_id or left > right:
                continue
            complete_rows += 1
            base = {
                "feature_id": feature_id,
                "feature_mz": float(mz),
                "feature_rt": float(rt),
                "rt_left": float(left),
                "rt_right": float(right),
            }
            for sample in samples:
                abundance = finite(row.get(sample))
                if abundance is not None and abundance > 0:
                    result[sample].append({**base, "abundance": float(abundance)})
    for sample in samples:
        result[sample].sort(key=lambda item: item["feature_mz"])
    return result, complete_rows


def read_mgf(path: Path) -> list[dict]:
    spectra = []
    current: dict | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            upper = line.upper()
            if upper == "BEGIN IONS":
                if current is not None:
                    raise RuntimeError(f"nested BEGIN IONS in {path}")
                current = {
                    "spectrum_index": len(spectra), "title": "", "scan": "",
                    "precursor_mz": None, "rt_seconds": None, "peak_count": 0,
                    "peak_tic": 0.0,
                }
            elif upper == "END IONS":
                if current is None:
                    raise RuntimeError(f"END IONS without BEGIN IONS in {path}")
                spectra.append(current)
                current = None
            elif current is not None and "=" in line:
                key, value = line.split("=", 1)
                key = key.upper()
                if key == "TITLE":
                    current["title"] = value
                elif key == "SCANS":
                    current["scan"] = value
                elif key == "PEPMASS":
                    current["precursor_mz"] = finite(value.split()[0])
                elif key == "RTINSECONDS":
                    current["rt_seconds"] = finite(value)
            elif current is not None and line:
                fields = line.split()
                if len(fields) >= 2:
                    intensity = finite(fields[1])
                    if intensity is not None:
                        current["peak_count"] += 1
                        current["peak_tic"] += max(0.0, intensity)
    if current is not None:
        raise RuntimeError(f"unterminated MGF spectrum in {path}")
    return spectra


def match_spectrum(
    spectrum: dict, features: list[dict], masses: list[float], ppm: float
) -> list[dict]:
    precursor = spectrum["precursor_mz"]
    rt = spectrum["rt_seconds"]
    if precursor is None or rt is None:
        return []
    tolerance = precursor * ppm * 1e-6
    start = bisect_left(masses, precursor - tolerance)
    stop = bisect_right(masses, precursor + tolerance)
    return [feature for feature in features[start:stop] if feature["rt_left"] <= rt <= feature["rt_right"]]


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"minimum": 0, "median": 0.0, "p90": 0.0, "maximum": 0}
    ordered = sorted(values)
    return {
        "minimum": int(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p90": float(ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]),
        "maximum": int(ordered[-1]),
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
    parser.add_argument("--study", choices=sorted(STUDIES), required=True)
    parser.add_argument("--external-archive", type=Path, required=True)
    parser.add_argument("--conversion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-ppm", type=float, default=10.0)
    parser.add_argument("--sensitivity-ppm", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    external = args.external_archive.resolve()
    conversion = args.conversion_dir.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    if digest(external) != EXPECTED_EXTERNAL_SHA256:
        raise RuntimeError("external archive SHA256 mismatch")
    conversion_report_path = conversion / "report.json"
    conversion_manifest_path = conversion / "conversion_manifest.csv.gz"
    conversion_report = json.loads(conversion_report_path.read_text(encoding="utf-8"))
    if (
        conversion_report.get("study") != args.study
        or not conversion_report.get("pass_to_truthblind_feature_join")
        or conversion_report.get("conversion_manifest_sha256") != digest(conversion_manifest_path)
    ):
        raise RuntimeError("conversion provenance/gate mismatch")
    with gzip.open(conversion_manifest_path, "rt", encoding="utf-8", newline="") as handle:
        conversions = list(csv.DictReader(handle))
    samples = [item["sample"] for item in conversions]
    if len(samples) != len(set(samples)):
        raise RuntimeError("duplicate samples in conversion manifest")

    with ZipFile(external) as archive:
        member = unique_suffix(archive, STUDIES[args.study])
        by_sample, feature_rows = read_detected_features(archive, member, samples)

    spectrum_audit = []
    query_spectra = []
    sensitivity_counts = {"matched": 0, "unique": 0, "ambiguous": 0}
    for sample_index, sample in enumerate(samples, 1):
        mgf = conversion / "mgf" / f"{sample}.mgf"
        conversion_row = next(item for item in conversions if item["sample"] == sample)
        if not mgf.is_file() or digest(mgf) != conversion_row["mgf_sha256"]:
            raise RuntimeError(f"MGF provenance mismatch for {sample}")
        spectra = read_mgf(mgf)
        if len(spectra) != int(conversion_row["spectra"]):
            raise RuntimeError(f"MGF spectrum count mismatch for {sample}")
        features = by_sample[sample]
        masses = [item["feature_mz"] for item in features]
        for spectrum in spectra:
            primary = match_spectrum(spectrum, features, masses, args.primary_ppm)
            sensitivity = match_spectrum(spectrum, features, masses, args.sensitivity_ppm)
            sensitivity_counts["matched"] += len(sensitivity) > 0
            sensitivity_counts["unique"] += len(sensitivity) == 1
            sensitivity_counts["ambiguous"] += len(sensitivity) > 1
            status = "unmatched" if not primary else "unique" if len(primary) == 1 else "ambiguous"
            spectrum_audit.append({
                "study": args.study, "sample": sample,
                "spectrum_index": spectrum["spectrum_index"], "scan": spectrum["scan"],
                "title": spectrum["title"], "precursor_mz": spectrum["precursor_mz"],
                "rt_seconds": spectrum["rt_seconds"], "peak_count": spectrum["peak_count"],
                "peak_tic": spectrum["peak_tic"], "join_status": status,
                "match_count": len(primary),
                "matched_feature_ids": ";".join(item["feature_id"] for item in primary),
            })
            if len(primary) == 1:
                feature = primary[0]
                query_spectra.append({
                    "study": args.study, "sample": sample,
                    "event_id": f"{args.study}|{sample}|{feature['feature_id']}",
                    "feature_id": feature["feature_id"],
                    "feature_mz": feature["feature_mz"], "feature_rt": feature["feature_rt"],
                    "rt_left": feature["rt_left"], "rt_right": feature["rt_right"],
                    "abundance": feature["abundance"],
                    "spectrum_index": spectrum["spectrum_index"], "scan": spectrum["scan"],
                    "title": spectrum["title"], "precursor_mz": spectrum["precursor_mz"],
                    "rt_seconds": spectrum["rt_seconds"], "peak_count": spectrum["peak_count"],
                    "peak_tic": spectrum["peak_tic"],
                    "ppm_error": 1e6 * (spectrum["precursor_mz"] - feature["feature_mz"]) / feature["feature_mz"],
                    "rt_error_seconds": spectrum["rt_seconds"] - feature["feature_rt"],
                    "mgf_path": str(mgf),
                })
        print(
            f"[join {args.study}] {sample_index}/{len(samples)} {sample} "
            f"spectra={len(spectra):,}", flush=True,
        )

    if not spectrum_audit:
        raise RuntimeError("no MS2 spectra were read")
    if not query_spectra:
        raise RuntimeError("no unique truth-blind MS2-to-feature joins were produced")
    output.mkdir(parents=True, exist_ok=False)
    audit_path = output / "spectrum_join_audit.csv.gz"
    query_path = output / "query_spectra.csv.gz"
    event_path = output / "query_events.csv.gz"
    for path, rows in ((audit_path, spectrum_audit), (query_path, query_spectra)):
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, list[dict]] = {}
    for row in query_spectra:
        grouped.setdefault(row["event_id"], []).append(row)
    events = []
    for event_id, rows in sorted(grouped.items()):
        representative = max(
            rows, key=lambda item: (item["peak_tic"], item["peak_count"], -item["spectrum_index"])
        )
        events.append({
            "study": representative["study"], "sample": representative["sample"],
            "event_id": event_id, "feature_id": representative["feature_id"],
            "feature_mz": representative["feature_mz"],
            "feature_rt": representative["feature_rt"],
            "abundance": representative["abundance"], "n_linked_ms2": len(rows),
            "representative_rule": "maximum_peak_tic_then_peak_count_then_first_index",
            "representative_spectrum_index": representative["spectrum_index"],
            "representative_scan": representative["scan"],
            "representative_precursor_mz": representative["precursor_mz"],
            "representative_rt_seconds": representative["rt_seconds"],
            "representative_peak_count": representative["peak_count"],
            "representative_peak_tic": representative["peak_tic"],
            "mgf_path": representative["mgf_path"],
        })
    if not events:
        raise RuntimeError("no sample-feature query events were produced")
    with gzip.open(event_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)

    unique_matched = len(query_spectra)
    ambiguous = sum(item["join_status"] == "ambiguous" for item in spectrum_audit)
    any_matched = unique_matched + ambiguous
    complete = sum(
        item["precursor_mz"] is not None and item["rt_seconds"] is not None
        for item in spectrum_audit
    )
    distinct_features = {item["feature_id"] for item in events}
    probe_reproduction = {"applicable": args.study == "ST001122"}
    if args.study == "ST001122":
        probe_rows = [item for item in spectrum_audit if item["sample"] == "IC1_22"]
        probe_any = sum(item["join_status"] != "unmatched" for item in probe_rows)
        probe_unique = sum(item["join_status"] == "unique" for item in probe_rows)
        probe_ambiguous = sum(item["join_status"] == "ambiguous" for item in probe_rows)
        probe_reproduction.update({
            "sample": "IC1_22", "any_matched_spectra": probe_any,
            "uniquely_matched_spectra": probe_unique,
            "ambiguously_matched_spectra": probe_ambiguous,
            "expected": {"any_matched_spectra": 2762, "uniquely_matched_spectra": 2747,
                         "ambiguously_matched_spectra": 15},
            "pass": (probe_any, probe_unique, probe_ambiguous) == (2762, 2747, 15),
        })
    else:
        probe_reproduction["pass"] = True
    gates = {
        "conversion_gate_passed": True,
        "all_samples_processed": len(samples) == conversion_report["samples"],
        "complete_precursor_rt_ge_99pct": complete >= 0.99 * len(spectrum_audit),
        "unique_joined_spectra_ge_1000": unique_matched >= 1000,
        "sample_feature_events_ge_1000": len(events) >= 1000,
        "distinct_feature_ids_ge_500": len(distinct_features) >= 500,
        "st001122_probe_join_reproduced_when_applicable": probe_reproduction["pass"],
        "truth_payloads_not_opened": True,
        "algorithm_output_payloads_not_opened": True,
    }
    report = {
        "status": "bioaware_b47_truthblind_feature_join_complete",
        "formal": True, "study": args.study, "samples": len(samples),
        "feature_rows_with_complete_coordinates": feature_rows,
        "detected_feature_instances": sum(len(items) for items in by_sample.values()),
        "ms2_spectra": len(spectrum_audit), "ms2_with_precursor_and_rt": complete,
        "primary_join": {
            "ppm": args.primary_ppm, "requires_empirical_feature_peak_boundary": True,
            "requires_positive_sample_abundance": True,
            "any_matched_spectra": any_matched,
            "any_matched_fraction": any_matched / complete if complete else 0.0,
            "uniquely_matched_spectra": unique_matched,
            "unique_match_fraction": unique_matched / complete if complete else 0.0,
            "ambiguous_spectra": ambiguous,
            "ambiguous_fraction": ambiguous / complete if complete else 0.0,
            "sample_feature_events": len(events),
            "distinct_feature_ids": len(distinct_features),
            "linked_ms2_per_event": quantiles([item["n_linked_ms2"] for item in events]),
        },
        "sensitivity_join": {
            "ppm": args.sensitivity_ppm, **sensitivity_counts,
            "matched_fraction": sensitivity_counts["matched"] / complete if complete else 0.0,
        },
        "probe_reproduction": probe_reproduction,
        "gates": gates, "pass_to_truthblind_candidate_graph": all(gates.values()),
        "contracts": {
            "evaluation_unit": "sample-feature event",
            "primary_query_spectrum": "highest peak TIC, then peak count, then first index",
            "ambiguous_feature_links": "retained in audit but excluded from query events",
            "truth_opened": False, "algorithm_outputs_opened": False,
        },
        "provenance": {
            "external_archive_sha256": digest(external),
            "conversion_report_sha256": digest(conversion_report_path),
            "conversion_manifest_sha256": digest(conversion_manifest_path),
            "spectrum_join_audit_sha256": digest(audit_path),
            "query_spectra_sha256": digest(query_path),
            "query_events_sha256": digest(event_path),
        },
        "claim_limit": (
            "This freezes observable sample-feature-MS2 events without truth. It does not "
            "establish candidate identities, DreaMS errors, BioAware gain, or embedding gain."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
