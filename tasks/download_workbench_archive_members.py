#!/usr/bin/env python
"""Download named members from a Workbench study ZIP without the full archive."""

from __future__ import annotations

import argparse
import json
import os
import time
from hashlib import sha256
from pathlib import Path

import requests


ENDPOINT = "https://www.metabolomicsworkbench.org/data/file_extract.php"


def checksum(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(files_json: Path) -> dict[str, int]:
    payload = json.loads(files_json.read_text(encoding="utf-8-sig"))
    answer: dict[str, int] = {}
    for rows in payload.get("compressed_file_content", {}).values():
        for row in rows:
            answer[str(row["name"])] = int(row.get("size", 0) or 0)
    return answer


def download_member(archive: str, member: str, destination: Path, expected: int) -> dict:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with requests.post(
                ENDPOINT,
                data={"A": archive, "F": member},
                timeout=(30, 600),
                stream=True,
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                with temporary.open("wb") as handle:
                    written = 0
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                            written += len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if expected and written != expected:
                preview = temporary.read_bytes()[:200]
                raise RuntimeError(
                    f"member size mismatch for {member}: {written} != {expected}; "
                    f"content_type={content_type!r}; prefix={preview!r}"
                )
            temporary.replace(destination)
            return {
                "member": member,
                "path": str(destination),
                "bytes": written,
                "sha256": checksum(destination),
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < 4:
                time.sleep(5 * attempt)
    raise RuntimeError(f"failed to download archive member {member}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", default="ST001154")
    parser.add_argument("--archive", default="ST001154_rawdata.zip")
    parser.add_argument(
        "--files-json",
        type=Path,
        default=Path(
            "data/reference/bioaware_public_cohort_probe_20260901/"
            "ST001154__files__json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference/ST001154_negative_pilot_20260901"),
    )
    parser.add_argument("members", nargs="+")
    args = parser.parse_args()

    known = inventory(args.files_json)
    unknown = [member for member in args.members if member not in known]
    if unknown:
        raise RuntimeError(f"requested members are absent from the frozen inventory: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for member in args.members:
        destination = args.output_dir / Path(member).name
        expected = known[member]
        if destination.exists():
            if destination.stat().st_size != expected:
                raise RuntimeError(f"refusing mismatched existing member: {destination}")
            record = {
                "member": member,
                "path": str(destination),
                "bytes": expected,
                "sha256": checksum(destination),
                "status": "reused",
            }
        else:
            print(f"[download] {member} ({expected:,} bytes)", flush=True)
            record = download_member(args.archive, member, destination, expected)
            record["status"] = "downloaded"
        records.append(record)
        print(f"[download] {record['status']}: {destination}", flush=True)
    report_path = args.output_dir / "download_report.json"
    existing_records: dict[str, dict] = {}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("study_id") != args.study_id or previous.get("archive") != args.archive:
            raise RuntimeError(
                "refusing to merge a download ledger from a different study/archive"
            )
        for previous_record in previous.get("members", []):
            previous_member = str(previous_record["member"])
            if previous_member in existing_records:
                raise RuntimeError(f"duplicate member in existing ledger: {previous_member}")
            existing_records[previous_member] = previous_record
    for record in records:
        member = str(record["member"])
        previous = existing_records.get(member)
        if previous is not None and (
            int(previous.get("bytes", -1)) != int(record["bytes"])
            or str(previous.get("sha256", "")) != str(record["sha256"])
        ):
            raise RuntimeError(f"download ledger conflict for member: {member}")
        existing_records[member] = record
    report = {
        "status": "workbench_archive_members_downloaded",
        "study_id": args.study_id,
        "archive": args.archive,
        "members": [existing_records[key] for key in sorted(existing_records)],
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".partial")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
