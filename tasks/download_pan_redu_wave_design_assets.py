#!/usr/bin/env python3
"""Download byte-verified small design assets for a Pan-ReDU compute wave."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_target(root: Path, accession: str, remote: str) -> Path:
    parts = []
    for part in PurePosixPath(remote).parts:
        if part in {"", ".", ".."}:
            continue
        # Preserve a reversible, filesystem-safe local name on Windows.
        parts.append(re.sub(r'[<>:"|?*]', lambda match: f"_{ord(match.group(0)):02x}", part))
    target = (root / accession / Path(*parts)).resolve()
    expected_parent = (root / accession).resolve()
    if expected_parent not in target.parents:
        raise ValueError(f"unsafe remote path: {remote}")
    return target


def fetch(url: str, target: Path, expected: int) -> str:
    if target.exists() and target.stat().st_size == expected:
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 11):
        part.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, part.open("wb") as handle:
                while block := response.read(1 << 20):
                    handle.write(block)
            if part.stat().st_size != expected:
                raise IOError(f"byte mismatch: {part.stat().st_size} != {expected}")
            part.replace(target)
            return "downloaded"
        except urllib.error.HTTPError as error:
            part.unlink(missing_ok=True)
            if attempt == 10 or error.code != 429:
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else min(60, 10 * attempt)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError):
            part.unlink(missing_ok=True)
            if attempt == 10:
                raise
            time.sleep(min(60, attempt * 5))
    raise AssertionError("unreachable")


def resolve_ftps_versions(accessions: set[str]) -> dict[str, str]:
    unresolved = set(accessions)
    result: dict[str, str] = {}
    for version_number in range(1, 16):
        version = f"v{version_number:02d}"
        for attempt in range(1, 6):
            completed = subprocess.run(
                [
                    "curl.exe", "--silent", "--show-error", "--ssl-reqd", "--ftp-pasv",
                    "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
                    "--user", "anonymous:anonymous@", "--list-only",
                    f"ftp://massive-ftp.ucsd.edu/{version}/",
                ],
                capture_output=True, text=True,
            )
            if completed.returncode == 0:
                break
            if attempt == 5:
                raise RuntimeError(
                    f"failed to list MassIVE {version}: {completed.stderr.strip()}"
                )
            time.sleep(attempt * 2)
        available = set(completed.stdout.splitlines()) & unresolved
        for accession in available:
            result[accession] = version
        unresolved -= available
        if not unresolved:
            break
    if unresolved:
        raise RuntimeError(f"MassIVE FTPS version unresolved: {sorted(unresolved)}")
    return result


def fetch_ftps(accession: str, filepath: str, version: str, target: Path, expected: int) -> str:
    if target.exists() and target.stat().st_size == expected:
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    encoded = urllib.parse.quote(filepath, safe="/")
    url = f"ftp://massive-ftp.ucsd.edu/{version}/{accession}/{encoded}"
    subprocess.run(
        [
            "curl.exe", "--silent", "--show-error", "--ssl-reqd", "--ftp-pasv",
            "--retry", "8", "--retry-delay", "3", "--retry-all-errors",
            "--user", "anonymous:anonymous@", "--output", str(part), url,
        ], check=True,
    )
    if part.stat().st_size != expected:
        raise IOError(f"FTPS byte mismatch for {accession}/{filepath}: {part.stat().st_size} != {expected}")
    part.replace(target)
    return "downloaded_ftps"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--request-delay", type=float, default=0.4)
    parser.add_argument("--ftps", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    inventory = pd.read_csv(args.inventory, sep="\t", dtype=str, keep_default_na=False)
    versions = (
        resolve_ftps_versions(set(inventory["dataset_accession"])) if args.ftps else {}
    )
    if versions:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "ftps_version_map.json").write_text(
            json.dumps(versions, indent=2) + "\n", encoding="utf-8"
        )
    def process(row: dict[str, str]) -> dict[str, object]:
        target = safe_target(args.output_dir, row["dataset_accession"], row["filepath"])
        expected = int(row["bytes"])
        status = (
            fetch_ftps(
                row["dataset_accession"], row["filepath"],
                versions[row["dataset_accession"]], target, expected,
            )
            if args.ftps else fetch(row["download_url"], target, expected)
        )
        return {
            **row, "local_path": str(target), "status": status,
            "observed_bytes": target.stat().st_size, "sha256": sha256(target),
        }

    records = []
    input_rows = inventory.to_dict("records")
    if args.workers == 1:
        for index, row in enumerate(input_rows, start=1):
            record = process(row)
            records.append(record)
            print(
                f"[{index}/{len(inventory)}] {record['status']} "
                f"{row['dataset_accession']} {row['filepath']}", flush=True,
            )
            time.sleep(args.request_delay)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, row): row for row in input_rows}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                record = future.result()
                records.append(record)
                print(
                    f"[{index}/{len(inventory)}] {record['status']} "
                    f"{row['dataset_accession']} {row['filepath']}", flush=True,
                )
    checks = pd.DataFrame(records)
    checks.to_csv(args.output_dir / "download_checksums.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_wave_design_asset_download_v1",
        "files": len(checks),
        "bytes": int(checks["observed_bytes"].sum()),
        "dataset_accessions": int(checks["dataset_accession"].nunique()),
        "all_byte_verified": bool((checks["bytes"].astype(int) == checks["observed_bytes"]).all()),
        "leakage_status": "assets_acquired_not_used_for_model_or_retrieval_tuning",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
