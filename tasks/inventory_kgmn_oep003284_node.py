#!/usr/bin/env python3
"""Freeze the public NODE inventory for KGMN OEP003284 without downloading data.

NODE exposes the raw files through a public metadata API, while files larger
than 200 MiB must be downloaded through the authenticated SFTP service.  This
script records the exact public file set, checks the expected 46STD layout,
and optionally verifies an already downloaded local directory by MD5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path


PROJECT_ALIAS = "OEP003284"
PROJECT_NUMBER = "OEP00003284"
API_URL = "https://www.biosino.org/node/api/app/browseDetail/getDataList"
SFTP_HOST = "fms.biosino.org"
SFTP_PORT = 44398
RUN_NUMBER = "OER00253320"
EXPECTED_NAME = re.compile(r"^(g[124])_46std_(pos|neg)_([1-4])\.mzXML$")


def md5(path: Path) -> str:
    hasher = hashlib.md5()  # noqa: S324 - repository publishes MD5 checksums
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sftp_run_path(run_number: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", run_number)
    if match is None:
        raise RuntimeError(f"invalid NODE run number: {run_number}")
    prefix, digits = match.groups()
    pieces = [prefix + digits[:end] for end in range(2, len(digits) + 1, 2)]
    if len(digits) % 2:
        pieces.append(prefix + digits)
    return "/Public/byRun/" + "/".join(pieces)


def fetch_inventory() -> list[dict]:
    payload = json.dumps(
        {
            "type": "project",
            "typeNo": PROJECT_NUMBER,
            "pageNum": 1,
            "pageSize": 100,
            "sortKey": "name",
            "sortType": "asc",
            "searchId": "",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "DreaMS-KGMN-audit/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS URL
        body = json.load(response)
    if body.get("code") != 200:
        raise RuntimeError(f"NODE API failed: {body}")
    data = body.get("data") or {}
    rows = data.get("list") or []
    if int(data.get("total", -1)) != len(rows):
        raise RuntimeError("NODE API returned a paginated or incomplete file list")
    return rows


def validate_remote(rows: list[dict]) -> dict:
    problems: list[str] = []
    if len(rows) != 24:
        problems.append(f"expected 24 public files, found {len(rows)}")
    names = [str(row.get("name", "")) for row in rows]
    if len(set(names)) != len(names):
        problems.append("duplicate public filenames")
    cells: dict[str, int] = {}
    for row in rows:
        name = str(row.get("name", ""))
        match = EXPECTED_NAME.fullmatch(name)
        if match is None:
            problems.append(f"unexpected filename: {name}")
            continue
        cell = f"{match.group(1)}_{match.group(2)}"
        cells[cell] = cells.get(cell, 0) + 1
        if row.get("security") != "Public" or row.get("accessible") is not True:
            problems.append(f"file is not publicly accessible: {name}")
        if row.get("runNo") != RUN_NUMBER:
            problems.append(f"unexpected run number for {name}: {row.get('runNo')}")
        if not re.fullmatch(r"[0-9a-f]{32}", str(row.get("md5", ""))):
            problems.append(f"invalid MD5 for {name}")
        if int(row.get("fileSize", 0)) <= 200 * 1024 * 1024:
            problems.append(f"unexpectedly small raw file: {name}")
    expected_cells = {f"g{group}_{polarity}": 4 for group in (1, 2, 4) for polarity in ("pos", "neg")}
    if cells != expected_cells:
        problems.append(f"raw-cell layout mismatch: {cells}")
    return {"problems": problems, "cells": cells, "pass": not problems}


def validate_local(rows: list[dict], local_root: Path | None) -> dict:
    if local_root is None:
        return {"checked": False, "ready": False, "reason": "local root not requested"}
    missing: list[str] = []
    size_mismatch: list[str] = []
    md5_mismatch: list[str] = []
    for row in rows:
        path = local_root / str(row["name"])
        if not path.is_file():
            missing.append(path.name)
            continue
        if path.stat().st_size != int(row["fileSize"]):
            size_mismatch.append(path.name)
            continue
        if md5(path).lower() != str(row["md5"]).lower():
            md5_mismatch.append(path.name)
    extra = sorted(
        path.name for path in local_root.glob("*.mzXML")
        if path.name not in {str(row["name"]) for row in rows}
    ) if local_root.is_dir() else []
    return {
        "checked": True,
        "local_root": str(local_root.resolve()) if local_root.exists() else str(local_root),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "md5_mismatch": md5_mismatch,
        "extra_mzxml": extra,
        "ready": not missing and not size_mismatch and not md5_mismatch and not extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-raw-root", type=Path)
    parser.add_argument("--require-local", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty inventory: {args.output_dir}")
    rows = fetch_inventory()
    remote = validate_remote(rows)
    if not remote["pass"]:
        raise RuntimeError("NODE public inventory failed: " + "; ".join(remote["problems"]))
    local = validate_local(rows, args.local_raw_root)
    if args.require_local and not local["ready"]:
        raise RuntimeError(f"local OEP003284 raw set is incomplete: {local}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sftp_path = sftp_run_path(RUN_NUMBER)
    compact_rows = []
    for row in sorted(rows, key=lambda item: str(item["name"])):
        compact_rows.append(
            {
                "name": row["name"],
                "dat_no": row["datNo"],
                "run_no": row["runNo"],
                "bytes": int(row["fileSize"]),
                "md5": row["md5"],
                "sftp_path": f"{sftp_path}/{row['name']}",
            }
        )
    csv_path = args.output_dir / "node_files.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0]))
        writer.writeheader()
        writer.writerows(compact_rows)

    report = {
        "status": "kgmn_oep003284_node_inventory_complete",
        "formal": True,
        "project_alias": PROJECT_ALIAS,
        "project_number": PROJECT_NUMBER,
        "run_number": RUN_NUMBER,
        "files": len(compact_rows),
        "bytes": sum(row["bytes"] for row in compact_rows),
        "raw_cells": remote["cells"],
        "download": {
            "transport": "SFTP required because every raw file exceeds NODE's 200 MiB HTTP limit",
            "host": SFTP_HOST,
            "port": SFTP_PORT,
            "path": sftp_path,
            "authentication": "user's NODE account; credentials are never stored by this script",
        },
        "local_validation": local,
        "files_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "claim_limit": "Outcome-free public raw-data inventory; no KGMN or DreaMS performance result.",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
