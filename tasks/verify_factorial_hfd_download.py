#!/usr/bin/env python3
"""Verify that every frozen factorial-HFD mzML was downloaded exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    required = {"panel_id", "dataset_accession", "sample_id", "local_path", "deposited_bytes"}
    if missing := sorted(required - set(manifest)):
        raise ValueError(f"manifest lacks {missing}")
    if manifest["local_path"].duplicated().any():
        raise ValueError("manifest contains duplicate local paths")

    reported: dict[str, dict[str, object]] = {}
    report_hashes: dict[str, str] = {}
    for path in sorted(args.report_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report_hashes[path.name] = sha256(path)
        for record in payload.get("records", []):
            key = str(record["local_path"])
            if key in reported:
                raise RuntimeError(f"download path reported more than once: {key}")
            reported[key] = record

    expected_paths = set(manifest["local_path"].astype(str))
    if set(reported) != expected_paths:
        missing = sorted(expected_paths - set(reported))
        extra = sorted(set(reported) - expected_paths)
        raise RuntimeError(f"download reports differ from manifest: missing={missing[:5]} extra={extra[:5]}")

    rows: list[dict[str, object]] = []
    for row in manifest.to_dict("records"):
        path = Path(row["local_path"])
        expected_bytes = int(row["deposited_bytes"])
        record = reported[path.as_posix()]
        if not path.is_file() or path.stat().st_size != expected_bytes or not valid_mzml(path):
            raise RuntimeError(f"invalid downloaded mzML: {path}")
        observed_hash = sha256(path)
        if int(record["observed_bytes"]) != expected_bytes or str(record["sha256"]) != observed_hash:
            raise RuntimeError(f"download report mismatch: {path}")
        rows.append({
            "panel_id": row["panel_id"],
            "dataset_accession": row["dataset_accession"],
            "sample_id": row["sample_id"],
            "local_path": path.as_posix(),
            "bytes": expected_bytes,
            "sha256": observed_hash,
        })

    verified = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True)
    verified.to_csv(args.output_dir / "verified_files.tsv.gz", sep="\t", index=False, compression="gzip")
    report = {
        "schema": "factorial_hfd_download_verification_v1",
        "status": "FROZEN_DOWNLOAD_COMPLETE",
        "data_role": "development_consumed_raw_unopened",
        "denominators": {
            "studies": int(verified.dataset_accession.nunique()),
            "panels": int(verified.panel_id.nunique()),
            "sample_files": len(verified),
            "verified_bytes": int(verified.bytes.sum()),
        },
        "leakage_status": "byte and envelope verification only; genotype, diet, abundance, spectra and response truth were not read",
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "batch_report_sha256": report_hashes,
            "verified_files_sha256": sha256(args.output_dir / "verified_files.tsv.gz"),
            "implementation_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
