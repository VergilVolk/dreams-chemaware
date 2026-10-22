#!/usr/bin/env python3
"""Resume-safe staging of the frozen 109-file MTBLS1866 positive universe."""

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


def stage_one(row: dict[str, str], output: Path) -> dict[str, object]:
    expected = int(row["expected_bytes"])
    target = output / Path(row["relative_path"]).name
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if target.stat().st_size != expected:
            raise RuntimeError(f"wrong-size completed file: {target}")
        return {**row, "status": "reused", "observed_bytes": expected, "sha256": sha256(target), "local_path": str(target)}
    if partial.exists() and partial.stat().st_size > expected:
        raise RuntimeError(f"partial exceeds expected size: {partial}")

    error: Exception | None = None
    for attempt in range(10):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            break
        headers = {"User-Agent": "DreaMS-Contrast-Atlas-stager/1.0"}
        mode = "wb"
        if offset:
            headers["Range"] = f"bytes={offset}-"
            mode = "ab"
        request = urllib.request.Request(row["official_url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if offset and int(getattr(response, "status", 200)) != 206:
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
            time.sleep(min(2**attempt, 12))
            continue
        if partial.stat().st_size == expected:
            break
        error = RuntimeError(f"short response: {partial.stat().st_size} != {expected}")
    actual = partial.stat().st_size if partial.exists() else 0
    if actual != expected:
        raise RuntimeError(f"incomplete staged file: {target.name}, {actual} != {expected}") from error
    partial.replace(target)
    return {**row, "status": "downloaded", "observed_bytes": expected, "sha256": sha256(target), "local_path": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/validation/mtbls1866_positive_manifest_v1_20261022/"
            "phenotype_blind_acquisition_manifest.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mtbls1866/positive_primary_mzml")
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between one and four")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 109 or any(row["linkage_visibility"] != "phenotype_blind" for row in rows):
        raise RuntimeError("manifest is not the frozen 109-file phenotype-blind universe")
    if len({row["assay_name"] for row in rows}) != 109 or len({row["relative_path"] for row in rows}) != 109:
        raise RuntimeError("duplicate assay or mzML path in manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage_one, row, args.output_dir): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps({
                "assay_name": result["assay_name"],
                "status": result["status"],
                "observed_bytes": result["observed_bytes"],
            }), flush=True)
    completed.sort(key=lambda row: str(row["assay_name"]))
    manifest_path = args.output_dir / "staged_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(completed)
    report = {
        "schema": "mtbls1866_positive_primary_staging_v1",
        "status": "complete",
        "files": len(completed),
        "bytes": sum(int(row["observed_bytes"]) for row in completed),
        "manifest": str(manifest_path),
        "leakage_status": "phenotype-blind acquisition manifest only",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
