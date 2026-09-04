#!/usr/bin/env python
"""Extract a small, preregistered TIMS-Bench subset from the remote Zenodo ZIP.

The archive is about 9.1 GB, but Zenodo supports byte-range requests.  This
script deliberately extracts one member per fresh connection, verifies the
uncompressed byte count and CRC32, and only then atomically publishes it.  A
failed member therefore never corrupts already completed work.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
import zipfile
import zlib
from hashlib import sha256
from pathlib import Path

import fsspec
import requests


ARCHIVE_URL = (
    "https://zenodo.org/records/20816379/files/"
    "data_clean_2026-06-23.zip?download=1"
)

# Only files needed to determine polarity, truth coverage and whether an
# independent DreaMS/BioAware confirmation protocol is constructible.
MEMBER_SUFFIXES = (
    "groundtruth_dataset/NIST_SRM/harmonized/NIST_SRM_mzmine_harmonized.parquet",
    "groundtruth_dataset/plant_spikein/harmonized/plant_spikein_mzmine_harmonized.parquet",
    "groundtruth_dataset/MSV000098263/confidently_annotated_inchikeys.csv",
    "groundtruth_dataset/MSV000098263/confidently_annotated_inchikey.csv",
    "groundtruth_dataset/MSV000098263/harmonized/MSV000098263_mzmine_harmonized.parquet",
    "library_spectra/reframe_ms2s_with_ccs.parquet",
    "library_spectra/reframe_smiles_list.csv",
    "library_spectra/reframe_spikein_lib.pq",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_remote():
    # A small block cache prevents downloading large neighboring members while
    # still amortising central-directory and compressed-member reads.
    return fsspec.open(
        ARCHIVE_URL,
        mode="rb",
        block_size=2 * 1024 * 1024,
        cache_type="readahead",
        timeout=120,
    ).open()


def read_http_range(start: int, end: int, retries: int) -> bytes:
    """Read an exact inclusive HTTP byte range without accepting a full ZIP."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                ARCHIVE_URL,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                timeout=(30, 120),
                stream=True,
            )
            if response.status_code != 206:
                response.close()
                raise RuntimeError(f"server ignored byte range: HTTP {response.status_code}")
            expected = end - start + 1
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    payload.extend(chunk)
                    if len(payload) > expected:
                        raise RuntimeError("range response exceeded requested size")
            response.close()
            if len(payload) != expected:
                raise RuntimeError(f"short range response: {len(payload)} != {expected}")
            return bytes(payload)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(3 * attempt, 15))
    raise RuntimeError(f"failed HTTP range {start}-{end}") from last_error


def compressed_data_offset(info: zipfile.ZipInfo, retries: int) -> int:
    header = read_http_range(info.header_offset, info.header_offset + 29, retries)
    signature, = struct.unpack_from("<I", header, 0)
    if signature != 0x04034B50:
        raise RuntimeError(f"invalid local ZIP header at {info.header_offset}")
    filename_length, extra_length = struct.unpack_from("<HH", header, 26)
    return info.header_offset + 30 + filename_length + extra_length


def resolve_members() -> dict[str, zipfile.ZipInfo]:
    with open_remote() as remote, zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        resolved: dict[str, zipfile.ZipInfo] = {}
        for suffix in MEMBER_SUFFIXES:
            matches = [info for info in infos if info.filename.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one archive member ending in {suffix!r}; "
                    f"found {len(matches)}"
                )
            resolved[suffix] = matches[0]
        return resolved


def extract_member(info: zipfile.ZipInfo, destination: Path, retries: int) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != info.file_size:
            raise RuntimeError(
                f"refusing mismatched existing file: {destination} "
                f"({destination.stat().st_size} != {info.file_size})"
            )
        return {
            "member": info.filename,
            "path": str(destination),
            "bytes": info.file_size,
            "crc32": f"{info.CRC:08x}",
            "sha256": file_sha256(destination),
            "status": "reused",
        }

    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    compressed = destination.with_suffix(destination.suffix + ".compressed.partial")
    if compressed.exists():
        compressed.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        crc = 0
        written = 0
        try:
            data_offset = compressed_data_offset(info, retries)
            range_chunk = 8 * 1024 * 1024
            with compressed.open("wb") as sink:
                for relative_start in range(0, info.compress_size, range_chunk):
                    relative_end = min(relative_start + range_chunk, info.compress_size) - 1
                    payload = read_http_range(
                        data_offset + relative_start,
                        data_offset + relative_end,
                        retries,
                    )
                    sink.write(payload)
                    print(
                        f"  [range] {relative_end + 1:,}/{info.compress_size:,} compressed bytes",
                        flush=True,
                    )
                sink.flush()
                os.fsync(sink.fileno())

            if compressed.stat().st_size != info.compress_size:
                raise RuntimeError("compressed member size mismatch")

            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            with compressed.open("rb") as source, temporary.open("wb") as sink:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    decoded = decompressor.decompress(chunk)
                    if decoded:
                        sink.write(decoded)
                        written += len(decoded)
                        crc = zlib.crc32(decoded, crc)
                decoded = decompressor.flush()
                if decoded:
                    sink.write(decoded)
                    written += len(decoded)
                    crc = zlib.crc32(decoded, crc)
                sink.flush()
                os.fsync(sink.fileno())
            crc &= 0xFFFFFFFF
            if written != info.file_size:
                raise RuntimeError(f"size mismatch: {written} != {info.file_size}")
            if crc != info.CRC:
                raise RuntimeError(f"CRC mismatch: {crc:08x} != {info.CRC:08x}")
            temporary.replace(destination)
            compressed.unlink()
            return {
                "member": info.filename,
                "path": str(destination),
                "bytes": written,
                "crc32": f"{crc:08x}",
                "sha256": file_sha256(destination),
                "attempt": attempt,
                "status": "extracted",
            }
        except Exception as exc:  # network failures are expected and retried
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if compressed.exists():
                compressed.unlink()
            if attempt < retries:
                time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"failed to extract {info.filename!r} after {retries} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference/timsbench_groundtruth_probe_20260901"),
    )
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    resolved = resolve_members()
    records = []
    for suffix, info in resolved.items():
        destination = args.output_dir / Path(suffix).name
        print(f"[extract] {info.filename} ({info.file_size:,} bytes)", flush=True)
        record = extract_member(info, destination, args.retries)
        records.append(record)
        print(f"[extract] {record['status']}: {destination}", flush=True)

    report = {
        "status": "timsbench_minimal_groundtruth_extracted",
        "archive_url": ARCHIVE_URL,
        "members": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
