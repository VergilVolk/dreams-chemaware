#!/usr/bin/env python
"""Resumable truth-blind RAW-to-MGF conversion for B47 external studies.

Only raw files whose basenames occur in the frozen observable feature table are
extracted. Validation/truth and algorithm-output payloads are never opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zlib
from zipfile import ZipFile, ZipInfo


EXPECTED_EXTERNAL_SHA256 = (
    "2d6ed0ae51b385b99025398e62939d8902096ba798e603dc5bd0acc8c9b28898"
)
STUDIES = {
    "ST001122": {
        "raw_md5": "eedbf0071f945b4d6e01b4b836d3f5ca",
        "feature_suffix": "Oliver Fiehn/ST001122_urine/asari/preferred_Feature_table.tsv",
        "raw_selector": lambda name: name.casefold().endswith(".raw"),
    },
    "ST003356": {
        "raw_md5": "2498d32330e1dbdfd7f06fd793319215",
        "feature_suffix": "CZ Biohub/ST003356_urine/asari/HILICPOS/preferred_Feature_table.tsv",
        "raw_selector": lambda name: (
            name.casefold().endswith(".raw") and "_pos_" in name.casefold()
        ),
    },
}
FEATURE_METADATA = {
    "id_number", "mz", "rtime", "rtime_left_base", "rtime_right_base",
    "parent_masstrack_id", "peak_area", "cSelectivity", "goodness_fitting",
    "snr", "detection_counts",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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


def unique_suffix(archive: ZipFile, suffix: str) -> str:
    matches = [
        item.filename
        for item in archive.infolist()
        if not item.is_dir()
        and item.filename.replace("\\", "/").casefold().endswith(suffix.casefold())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one archive member ending {suffix!r}: {matches}")
    return matches[0]


def feature_samples(archive: ZipFile, suffix: str) -> tuple[str, list[str]]:
    member = unique_suffix(archive, suffix)
    with archive.open(member) as handle:
        header = handle.readline(1024 * 1024).decode("utf-8-sig").rstrip("\r\n")
    columns = next(csv.reader([header], delimiter="\t"))
    samples = [column for column in columns if column not in FEATURE_METADATA]
    if len(samples) != len(set(samples)):
        raise RuntimeError("duplicate sample columns in feature table")
    return member, samples


def file_crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def extract_exact(archive: ZipFile, info: ZipInfo, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == info.file_size:
        if file_crc32(destination) == info.CRC:
            return digest(destination)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with archive.open(info) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if temporary.stat().st_size != info.file_size or file_crc32(temporary) != info.CRC:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"raw extraction checksum mismatch: {info.filename}")
    os.replace(temporary, destination)
    return digest(destination)


def inspect_mgf(path: Path) -> dict[str, int]:
    spectra = precursor = retention = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip().upper()
            spectra += line == "BEGIN IONS"
            precursor += line.startswith("PEPMASS=")
            retention += line.startswith("RTINSECONDS=")
    return {"spectra": spectra, "precursor_headers": precursor, "rt_headers": retention}


def convert_one(
    sample: str,
    raw: Path,
    raw_sha256: str,
    mgf_dir: Path,
    checkpoint_dir: Path,
    parser: Path,
    parser_sha256: str,
) -> dict:
    mgf = mgf_dir / f"{sample}.mgf"
    checkpoint = checkpoint_dir / f"{sample}.json"
    if checkpoint.is_file() and mgf.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            prior.get("raw_sha256") == raw_sha256
            and prior.get("parser_sha256") == parser_sha256
            and prior.get("mgf_sha256") == digest(mgf)
            and prior.get("spectra", 0) > 0
            and prior.get("precursor_headers") == prior.get("spectra")
            and prior.get("rt_headers") == prior.get("spectra")
        ):
            return prior

    mgf_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{sample}.", dir=mgf_dir) as temporary_name:
        temporary_dir = Path(temporary_name)
        command = [
            str(parser), f"-i={raw}", f"-o={temporary_dir}", "-f=0", "-m=2",
            "-L=2", "-l=3",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"ThermoRawFileParser failed for {sample}: {completed.stderr[-2000:]}"
            )
        produced = temporary_dir / f"{sample}.mgf"
        if not produced.is_file():
            raise RuntimeError(f"parser did not produce MGF for {sample}")
        counts = inspect_mgf(produced)
        if (
            counts["spectra"] <= 0
            or counts["precursor_headers"] != counts["spectra"]
            or counts["rt_headers"] != counts["spectra"]
        ):
            raise RuntimeError(f"incomplete MGF metadata for {sample}: {counts}")
        os.replace(produced, mgf)

    result = {
        "sample": sample,
        "raw_path": str(raw.resolve()),
        "raw_sha256": raw_sha256,
        "mgf_path": str(mgf.resolve()),
        "mgf_sha256": digest(mgf),
        "parser_sha256": parser_sha256,
        **inspect_mgf(mgf),
    }
    atomic_json(checkpoint, result)
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=sorted(STUDIES), required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--external-archive", type=Path, required=True)
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.jobs < 1 or args.jobs > 8:
        raise ValueError("--jobs must be between 1 and 8")
    config = STUDIES[args.study]
    raw_archive = args.raw_archive.resolve()
    external_archive = args.external_archive.resolve()
    parser = args.parser.resolve()
    output = args.output.resolve()
    for path in (raw_archive, external_archive, parser):
        if not path.is_file():
            raise FileNotFoundError(path)
    final_report = output / "report.json"
    if final_report.is_file():
        print(final_report.read_text(encoding="utf-8"))
        return
    if digest(raw_archive, "md5") != config["raw_md5"]:
        raise RuntimeError(f"{args.study} raw archive MD5 mismatch")
    if digest(external_archive) != EXPECTED_EXTERNAL_SHA256:
        raise RuntimeError("external archive SHA256 mismatch")

    with ZipFile(external_archive) as archive:
        feature_member, samples = feature_samples(archive, config["feature_suffix"])
    with ZipFile(raw_archive) as archive:
        selected = [
            item for item in archive.infolist()
            if not item.is_dir() and config["raw_selector"](item.filename)
        ]
        by_sample: dict[str, ZipInfo] = {}
        for item in selected:
            sample = Path(item.filename).stem
            if sample in by_sample:
                raise RuntimeError(f"duplicate raw basename: {sample}")
            by_sample[sample] = item
        if set(by_sample) != set(samples):
            raise RuntimeError(
                f"raw/feature sample mismatch missing={sorted(set(samples)-set(by_sample))} "
                f"extra={sorted(set(by_sample)-set(samples))}"
            )
        raw_dir = output / "raw"
        raw_hashes = {}
        for index, sample in enumerate(samples, 1):
            raw_hashes[sample] = extract_exact(
                archive, by_sample[sample], raw_dir / f"{sample}.raw"
            )
            print(f"[extract {args.study}] {index}/{len(samples)} {sample}", flush=True)

    parser_sha256 = digest(parser)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                convert_one, sample, output / "raw" / f"{sample}.raw",
                raw_hashes[sample], output / "mgf", output / "checkpoints", parser,
                parser_sha256,
            ): sample
            for sample in samples
        }
        for index, future in enumerate(as_completed(futures), 1):
            sample = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[convert {args.study}] {index}/{len(samples)} {sample} "
                f"spectra={result['spectra']:,}", flush=True,
            )
    results.sort(key=lambda item: item["sample"])
    manifest = output / "conversion_manifest.csv.gz"
    with gzip.open(manifest, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    gates = {
        "raw_archive_md5_matches": True,
        "external_archive_sha256_matches": True,
        "raw_feature_samples_match_exactly": len(results) == len(samples),
        "every_sample_contains_ms2": all(item["spectra"] > 0 for item in results),
        "every_ms2_has_precursor": all(
            item["precursor_headers"] == item["spectra"] for item in results
        ),
        "every_ms2_has_rt": all(item["rt_headers"] == item["spectra"] for item in results),
        "truth_payloads_not_opened": True,
        "algorithm_output_payloads_not_opened": True,
    }
    report = {
        "status": "bioaware_b47_truthblind_ms2_conversion_complete",
        "formal": True,
        "study": args.study,
        "samples": len(samples),
        "ms2_spectra": sum(item["spectra"] for item in results),
        "minimum_ms2_per_sample": min(item["spectra"] for item in results),
        "maximum_ms2_per_sample": max(item["spectra"] for item in results),
        "feature_table_member": feature_member,
        "conversion_manifest_sha256": digest(manifest),
        "gates": gates,
        "pass_to_truthblind_feature_join": all(gates.values()),
        "contracts": {
            "selected_raw_files": "exact observable feature-table sample namespace only",
            "ms_level": 2,
            "output_format": "MGF",
            "truth_opened": False,
            "algorithm_outputs_opened": False,
        },
        "provenance": {
            "raw_archive": {"path": str(raw_archive), "sha256": digest(raw_archive)},
            "external_archive": {
                "path": str(external_archive), "sha256": digest(external_archive),
            },
            "parser": {"path": str(parser), "sha256": parser_sha256},
        },
        "claim_limit": (
            "This is a truth-blind conversion result. It contains no candidate ranking, "
            "annotation truth, BioAware effect, or embedding claim."
        ),
    }
    atomic_json(final_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
