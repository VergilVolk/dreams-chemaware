#!/usr/bin/env python
"""Download a small, checksummed Metabolomics Workbench metadata snapshot."""

from __future__ import annotations

import argparse
import json
import os
import time
from hashlib import sha256
from pathlib import Path

import requests


ENDPOINTS = ("summary", "analysis", "mwtab", "files", "metabolites", "data", "factors")


def checksum(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(url: str, destination: Path, retries: int = 4) -> dict:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=(30, 180))
            response.raise_for_status()
            payload = response.content
            if not payload:
                raise RuntimeError("empty response")
            temporary.write_bytes(payload)
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            temporary.replace(destination)
            return {
                "url": response.url,
                "path": str(destination),
                "bytes": len(payload),
                "sha256": checksum(destination),
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference/bioaware_public_cohort_probe_20260901"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    study = args.study_id.upper()
    if not study.startswith("ST"):
        raise ValueError("study_id must be a Workbench ST accession")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for endpoint in ENDPOINTS:
        destination = args.output_dir / f"{study}__{endpoint}__json"
        if destination.exists() and not args.overwrite:
            records[endpoint] = {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": checksum(destination),
                "status": "reused",
            }
            continue
        url = f"https://www.metabolomicsworkbench.org/rest/study/study_id/{study}/{endpoint}"
        record = fetch(url, destination)
        record["status"] = "downloaded"
        records[endpoint] = record
        print(f"[{endpoint}] {record['bytes']:,} bytes", flush=True)

    report = {"status": "workbench_metadata_snapshot_complete", "study_id": study, "files": records}
    report_path = args.output_dir / f"{study}__snapshot_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
