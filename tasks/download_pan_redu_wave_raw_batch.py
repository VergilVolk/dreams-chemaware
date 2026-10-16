#!/usr/bin/env python3
"""Resume-safe, byte-verified batch downloader for the systematic raw wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.parse
from pathlib import Path

import pandas as pd

from download_pan_redu_wave_design_assets import safe_target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_envelope(path: Path) -> bool:
    if path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        head = handle.read(4096).lower()
        handle.seek(max(0, path.stat().st_size - 8192))
        tail = handle.read().lower()
    if path.suffix.lower() == ".mzml":
        return b"mzml" in head and (b"</mzml>" in tail or b"</indexedmzml>" in tail)
    if path.suffix.lower() == ".mzxml":
        return b"mzxml" in head and b"</mzxml>" in tail
    return False


def download(row: dict[str, object], version: str, root: Path) -> dict[str, object]:
    accession = str(row["dataset_accession"])
    descriptor = str(row["remote_descriptor"])
    prefix = f"f.{accession}/"
    if not descriptor.startswith(prefix):
        raise ValueError(f"remote descriptor/accession mismatch: {descriptor}")
    relative = descriptor[len(prefix):]
    expected = int(row["deposited_bytes"])
    target = safe_target(root, accession, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected and valid_envelope(target):
        status = "reused_valid"
    else:
        part = target.with_suffix(target.suffix + ".part")
        encoded = urllib.parse.quote(relative, safe="/")
        url = f"ftp://massive-ftp.ucsd.edu/{version}/{accession}/{encoded}"
        subprocess.run(
            [
                "curl", "--silent", "--show-error", "--ssl-reqd", "--ftp-pasv",
                "--continue-at", "-", "--retry", "12", "--retry-delay", "10",
                "--retry-all-errors", "--user", "anonymous:anonymous@",
                "--output", str(part), url,
            ], check=True,
        )
        if part.stat().st_size != expected:
            raise IOError(f"byte mismatch {accession}/{relative}: {part.stat().st_size} != {expected}")
        if not valid_envelope(part):
            raise IOError(f"invalid mzML/mzXML envelope: {accession}/{relative}")
        part.replace(target)
        status = "downloaded_valid"
    return {
        "dataset_accession": accession,
        "remote_descriptor": descriptor,
        "local_path": str(target),
        "expected_bytes": expected,
        "observed_bytes": target.stat().st_size,
        "sha256": sha256(target),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--version-map", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    versions = json.loads(args.version_map.read_text(encoding="utf-8"))
    start = args.batch_index * args.batch_size
    batch = frame.iloc[start:start + args.batch_size]
    if batch.empty:
        raise ValueError(f"empty batch {args.batch_index}; manifest has {len(frame)} rows")
    records = []
    for row in batch.to_dict("records"):
        records.append(download(row, versions[row["dataset_accession"]], args.root))
        print(json.dumps(records[-1]), flush=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "pan_redu_systematic_wave_download_batch_v1",
        "batch_index": args.batch_index,
        "batch_size": len(records),
        "verified_bytes": sum(record["observed_bytes"] for record in records),
        "records": records,
    }
    (args.report_dir / f"batch_{args.batch_index:04d}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("batch_index", "batch_size", "verified_bytes")}, indent=2))


if __name__ == "__main__":
    main()
