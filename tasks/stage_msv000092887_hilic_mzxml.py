#!/usr/bin/env python3
"""Resume-safe staging of the frozen MSV000092887 HILIC-positive mzXML universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_candidates(url: str) -> list[str]:
    """Return equivalent official ProteoSAFe front ends without changing the file key."""
    parsed = urlsplit(url)
    hosts = [parsed.netloc, "gnps.ucsd.edu", "proteomics.ucsd.edu"]
    candidates: list[str] = []
    for host in hosts:
        candidate = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def stage_one(row: dict[str, str], output: Path) -> dict[str, object]:
    expected = int(row["expected_bytes"])
    target = output / Path(row["filepath"]).name
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if target.stat().st_size != expected:
            raise RuntimeError(f"wrong-size completed file: {target}")
        return {
            **row,
            "status": "reused",
            "retrieval_host": "reused_local",
            "observed_bytes": expected,
            "sha256": sha256(target),
            "local_path": str(target),
        }
    if partial.exists() and partial.stat().st_size > expected:
        raise RuntimeError(f"partial exceeds expected size: {partial}")

    error: Exception | None = None
    candidates = download_candidates(row["download_url"])
    retrieval_host = ""
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            break
        headers = {"User-Agent": "DreaMS-Contrast-Atlas-stager/1.0"}
        mode = "wb"
        if offset:
            headers["Range"] = f"bytes={offset}-"
            mode = "ab"
        candidate_url = candidates[attempt % len(candidates)]
        request = urllib.request.Request(candidate_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(getattr(response, "status", 200))
                if offset and status != 206:
                    mode = "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(4 << 20)
                        if not block:
                            break
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception as exc:
            error = exc
            time.sleep(min(2 ** attempt, 8))
            continue
        actual = partial.stat().st_size if partial.exists() else 0
        if actual == expected:
            retrieval_host = urlsplit(candidate_url).netloc
            break
        error = RuntimeError(f"short response: {actual} of {expected} bytes")
        time.sleep(min(2 ** attempt, 8))

    actual = partial.stat().st_size if partial.exists() else 0
    if actual != expected:
        raise RuntimeError(
            f"incomplete after eight attempts: {actual} != {expected} for {target.name}"
        ) from error
    partial.replace(target)
    return {
        **row,
        "status": "downloaded",
        "retrieval_host": retrieval_host,
        "observed_bytes": expected,
        "sha256": sha256(target),
        "local_path": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/validation/msv000092887_hilic_positive_mzxml_manifest_v2_20261019/"
            "mzxml_manifest.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/msv000092887/hilic_positive_mzxml")
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "dataset", "filepath", "assay_name", "expected_bytes", "download_url",
        "spectrum_linkage_visibility",
    }
    if not rows or required - set(rows[0]):
        raise ValueError("invalid phenotype-blind mzXML manifest")
    if len(rows) != 129 or sum(int(row["expected_bytes"]) for row in rows) != 838_574_158:
        raise RuntimeError("frozen MSV000092887 acquisition universe changed")
    if any(row["spectrum_linkage_visibility"] != "phenotype_blind" for row in rows):
        raise RuntimeError("linkage manifest exposes an unexpected visibility state")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage_one, row, args.output_dir): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                json.dumps(
                    {
                        "assay_name": result["assay_name"],
                        "status": result["status"],
                        "observed_bytes": result["observed_bytes"],
                    }
                ),
                flush=True,
            )

    completed.sort(key=lambda row: str(row["assay_name"]))
    staged_path = args.output_dir / "staged_manifest.tsv"
    with staged_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(completed)
    report = {
        "schema": "msv000092887_hilic_positive_mzxml_staging_v1",
        "status": "complete",
        "files": len(completed),
        "bytes": sum(int(row["observed_bytes"]) for row in completed),
        "retrieval_hosts": {
            host: sum(row["retrieval_host"] == host for row in completed)
            for host in sorted({str(row["retrieval_host"]) for row in completed})
        },
        "manifest": str(staged_path),
        "leakage_status": "phenotype-blind acquisition manifest only",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
