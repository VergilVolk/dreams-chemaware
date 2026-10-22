#!/usr/bin/env python3
"""Checkpointed staging of the 15 claim-bearing MTBLS1066 POS-RP files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "DreaMS-Contrast-Atlas-stager/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return int(response.headers["Content-Length"])
        except Exception as exc:  # public repository occasionally drops TLS handshakes
            error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"HEAD failed after five attempts: {url}") from error


def stage_one(row: dict[str, str], output: Path) -> dict[str, object]:
    url = row["official_url"]
    target = output / row["filename"]
    partial = target.with_suffix(target.suffix + ".part")
    expected = remote_size(url)
    if target.exists() and target.stat().st_size == expected:
        return {
            **row,
            "status": "reused",
            "bytes": expected,
            "sha256": sha256(target),
            "local_path": str(target),
        }
    if target.exists():
        raise RuntimeError(f"existing file has wrong size: {target}")

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise RuntimeError(f"partial file exceeds remote size: {partial}")
    if offset == expected:
        partial.replace(target)
        return {
            **row,
            "status": "recovered_complete_partial",
            "bytes": expected,
            "sha256": sha256(target),
            "local_path": str(target),
        }
    error: Exception | None = None
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            break
        headers = {"User-Agent": "DreaMS-Contrast-Atlas-stager/1.0"}
        mode = "wb"
        if offset:
            headers["Range"] = f"bytes={offset}-"
            mode = "ab"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(getattr(response, "status", 200))
                if offset and status != 206:
                    # A server that ignores Range must not be appended to a partial file.
                    offset = 0
                    mode = "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(8 << 20)
                        if not block:
                            break
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception as exc:
            error = exc
            time.sleep(min(2 ** attempt, 8))
            continue
        if partial.stat().st_size == expected:
            break
        error = RuntimeError(
            f"short response: {partial.stat().st_size} of {expected} bytes"
        )
        time.sleep(min(2 ** attempt, 8))
    actual = partial.stat().st_size if partial.exists() else 0
    if actual != expected:
        raise RuntimeError(
            f"incomplete download after eight attempts for {row['filename']}: "
            f"{actual} != {expected}"
        ) from error
    partial.replace(target)
    return {
        **row,
        "status": "downloaded",
        "bytes": expected,
        "sha256": sha256(target),
        "local_path": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_mtbls1066_kd_qualification_v1_20261022/"
            "primary_pos_rp_manifest.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mtbls1066/pos_rp_primary_mzml")
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 15 or any(row["panel"] != "POS_RP" for row in rows):
        raise RuntimeError("the frozen manifest is not the 15-file POS-RP primary panel")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage_one, row, args.output_dir): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps({
                "filename": result["filename"],
                "status": result["status"],
                "bytes": result["bytes"],
            }), flush=True)

    completed.sort(key=lambda row: str(row["filename"]))
    manifest_path = args.output_dir / "staged_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(completed)
    report = {
        "schema": "mtbls1066_primary_pos_rp_staging_v1",
        "status": "complete",
        "files": len(completed),
        "bytes": sum(int(row["bytes"]) for row in completed),
        "groups": {
            group: sum(row["group"] == group for row in completed)
            for group in sorted({str(row["group"]) for row in completed})
        },
        "manifest": str(manifest_path),
        "claim_boundary": "Staging opens no abundance effects and creates no biological result.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
