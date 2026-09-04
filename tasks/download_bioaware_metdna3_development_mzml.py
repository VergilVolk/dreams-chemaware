#!/usr/bin/env python
"""Download and byte-verify a frozen MetDNA3 mzML manifest.

The default remains the consumed development panel.  Internal/external panels
must opt in with an explicit scope so that reports cannot be mistaken for one
another.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(row: dict, destination: Path, retries: int = 8) -> None:
    expected = int(row["bytes"])
    if destination.exists():
        if destination.stat().st_size != expected:
            raise RuntimeError(f"existing file size mismatch: {destination}")
        print(f"[reuse] {destination.name}", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise RuntimeError(f"partial file exceeds expected size: {partial}")
        headers = {"User-Agent": "DreaMS-BioAware/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(row["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            break
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code == 429 or exc.code >= 500
            if not retryable or attempt >= retries:
                raise
            retry_after = None
            if isinstance(exc, urllib.error.HTTPError):
                retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else min(5 * 2**attempt, 60)
            except ValueError:
                delay = min(5 * 2**attempt, 60)
            print(
                f"[retry {attempt + 1}/{retries}] {destination.name}: "
                f"{type(exc).__name__}; resume={offset:,}; wait={delay:g}s",
                flush=True,
            )
            time.sleep(delay)
    if partial.stat().st_size != expected:
        raise RuntimeError(
            f"download size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected}"
        )
    # Windows antivirus/indexing can briefly retain a handle after the writer
    # closes.  Retry only the atomic commit; never redownload or truncate a
    # byte-complete partial file because of a transient sharing violation.
    for commit_attempt in range(8):
        try:
            os.replace(partial, destination)
            break
        except PermissionError:
            if commit_attempt == 7:
                raise
            delay = min(0.5 * 2**commit_attempt, 8.0)
            print(
                f"[commit retry {commit_attempt + 1}/7] {destination.name}; "
                f"wait={delay:g}s",
                flush=True,
            )
            time.sleep(delay)
    if not destination.exists() or destination.stat().st_size != expected:
        raise RuntimeError(f"atomic download commit failed: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1/download_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/metdna3_2025/mzml/development"),
    )
    parser.add_argument(
        "--scope",
        choices=("development", "internal_rplc", "external"),
        default="development",
    )
    parser.add_argument("--expected-files", type=int, default=16)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = payload["files"]
    if len(files) != args.expected_files:
        raise RuntimeError(
            f"{args.scope} download manifest must contain exactly "
            f"{args.expected_files} files, got {len(files)}"
        )
    names = [str(row["local_name"]) for row in files]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate local names in {args.scope} manifest")
    if any(Path(name).name != name for name in names):
        raise RuntimeError("local_name must be a basename, not a path")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    installed: list[dict] = []
    for index, row in enumerate(files, start=1):
        destination = output / row["local_name"]
        print(f"[download {index}/{len(files)}] {destination.name}", flush=True)
        download(row, destination)
        installed.append(
            {
                "name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "source_path": row["filepath"],
            }
        )
    report = {
        "status": f"bioaware_metdna3_{args.scope}_mzml_complete",
        "scope": args.scope,
        "files": installed,
        "total_bytes": sum(row["bytes"] for row in installed),
        "manifest_sha256": sha256(args.manifest),
    }
    report_path = output / "download_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError("existing download report differs")
    else:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
