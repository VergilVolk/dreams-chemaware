#!/usr/bin/env python
"""Physically isolate CHDWB observable inputs from the mixed MSMICA archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile


EXPECTED_ARCHIVE_SHA256 = (
    "67887553720068e18394fb6fb011f8365bc6714f5f3ce99c693fd13dc742dafa"
)
SELECTED = {
    "feature_table.csv": "METDNA2/Input/CHDWB_HILICPOS_feature_table.csv",
    "sample_information.csv": "METDNA2/Input/CHDWB_HILICPOS_sample_information.csv",
    "query_spectra.mgf": "METDNA2/Input/CHDWB_hilic_pos.mgf",
}


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def member_digest(archive: ZipFile, member: str) -> str:
    hasher = hashlib.sha256()
    with archive.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def find_unique_member(archive: ZipFile, suffix: str) -> str:
    matches = [
        info.filename.replace("\\", "/")
        for info in archive.infolist()
        if not info.is_dir() and info.filename.replace("\\", "/").casefold().endswith(suffix.casefold())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member ending {suffix!r}; observed {matches}")
    return matches[0]


def extract_selected(archive_path: Path, output: Path) -> dict:
    records = {}
    with ZipFile(archive_path) as archive:
        for target_name, suffix in SELECTED.items():
            member = find_unique_member(archive, suffix)
            expected_member_sha256 = member_digest(archive, member)
            target = output / target_name
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            observed_member_sha256 = file_digest(target)
            if observed_member_sha256 != expected_member_sha256:
                raise RuntimeError(f"extracted member hash mismatch: {member}")
            records[target_name] = {
                "archive_member": member,
                "bytes": int(target.stat().st_size),
                "sha256": observed_member_sha256,
            }
    return records


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--namespace-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    archive_path = args.archive.resolve()
    namespace_report_path = args.namespace_report.resolve()
    output = args.output.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not namespace_report_path.is_file():
        raise FileNotFoundError(namespace_report_path)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    archive_sha256 = file_digest(archive_path)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"head-to-head archive SHA256 mismatch: {archive_sha256}")
    namespace_report = json.loads(namespace_report_path.read_text(encoding="utf-8"))
    if namespace_report.get("truth_values_read") is not False:
        raise RuntimeError("namespace report does not attest truth_values_read=false")
    if not namespace_report.get("head_to_head", {}).get(
        "pass_to_truth_separated_development_construction", False
    ):
        raise RuntimeError("namespace report did not pass CHDWB development extraction gate")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        records = extract_selected(archive_path, staging)
        if set(path.name for path in staging.iterdir()) != set(SELECTED):
            raise RuntimeError("unexpected file in observable staging namespace")
        manifest = {
            "status": "bioaware_b47_chdwb_observable_namespace_sealed",
            "formal": True,
            "dataset": "CHDWB_HILIC_POS",
            "source_archive": str(archive_path),
            "source_archive_sha256": archive_sha256,
            "namespace_report": str(namespace_report_path),
            "namespace_report_sha256": file_digest(namespace_report_path),
            "files": records,
            "file_count": len(records),
            "sealed_truth_payloads_opened": False,
            "algorithm_output_payloads_opened": False,
            "contract": (
                "This directory contains observable inputs only. Validation and algorithm-output "
                "members remain inside the mixed source archive and are inaccessible by path here."
            ),
            "claim_limit": (
                "Physical separation permits development construction; it does not establish "
                "candidate validity, seed independence, or performance."
            ),
        }
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "files": manifest["file_count"],
                "bytes": sum(item["bytes"] for item in records.values()),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
