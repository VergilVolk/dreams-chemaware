#!/usr/bin/env python3
"""Resume-safe, byte- and envelope-verified MetaboLights HFD downloader."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_mzml(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        head = handle.read(4096).lower()
        handle.seek(max(0, path.stat().st_size - 8192))
        tail = handle.read().lower()
    return b"mzml" in head and (b"</mzml>" in tail or b"</indexedmzml>" in tail)


def download(row: dict[str, str]) -> dict[str, object]:
    target = Path(row["local_path"])
    expected = int(row["deposited_bytes"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected and valid_mzml(target):
        status = "reused_valid"
    else:
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run([
            "curl", "--location", "--fail", "--silent", "--show-error",
            "--continue-at", "-", "--retry", "12", "--retry-delay", "10",
            "--retry-all-errors", "--output", str(partial), row["download_url"],
        ], check=True)
        if partial.stat().st_size != expected:
            raise IOError(f"byte mismatch for {target}: {partial.stat().st_size} != {expected}")
        if not valid_mzml(partial):
            raise IOError(f"invalid mzML envelope: {target}")
        partial.replace(target)
        status = "downloaded_valid"
    return {
        "panel_id": row["panel_id"], "dataset_accession": row["dataset_accession"],
        "sample_id": row["sample_id"], "local_path": target.as_posix(),
        "expected_bytes": expected, "observed_bytes": target.stat().st_size,
        "sha256": sha256(target), "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--batch-index", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    start = args.batch_index * args.batch_size
    block = frame.iloc[start:start + args.batch_size]
    if block.empty:
        raise ValueError(f"empty batch {args.batch_index}; manifest contains {len(frame)} files")
    records = []
    for row in block.to_dict("records"):
        records.append(download(row))
        print(json.dumps(records[-1]), flush=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "factorial_hfd_download_batch_v1",
        "batch_index": args.batch_index,
        "files": len(records),
        "verified_bytes": sum(int(record["observed_bytes"]) for record in records),
        "records": records,
    }
    (args.report_dir / f"batch_{args.batch_index:03d}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("batch_index", "files", "verified_bytes")}, indent=2))


if __name__ == "__main__":
    main()
