#!/usr/bin/env python
"""Fail-closed audit: NetID yeast is an MS1/formula, not DreaMS-MS2, benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "MSV000087434"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_rows(sql: str) -> list[list]:
    url = (
        "https://datasetcache.gnps2.org/datasette/database.json?sql="
        + urllib.parse.quote(sql)
    )
    request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-BioAware/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["rows"]


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("data/validation/bioaware_netid_external_contract_v1/contract.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/bioaware_netid_dreams_feasibility.json"),
    )
    args = parser.parse_args()
    if not args.contract.exists():
        raise FileNotFoundError(args.contract)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("status") != "bioaware_netid_external_contract_frozen":
        raise RuntimeError("unexpected NetID contract")
    rows = query_rows(
        "select filepath,collection,size from filename "
        f"where dataset='{DATASET}' order by filepath"
    )
    yeast = [row for row in rows if "yeast" in (row[0] or "").lower()]
    yeast_open = [
        row for row in yeast if (row[0] or "").lower().endswith((".mzml", ".mzxml"))
    ]
    explicit_ms2 = [
        row
        for row in yeast_open
        if "ms2" in (row[0] or "").lower() or "target" in (row[0] or "").lower()
    ]
    liver_targeted = [
        row
        for row in rows
        if "liver-targeted-ms2" in (row[1] or "").lower()
        or "liver-targeted-ms2" in (row[0] or "").lower()
    ]
    report = {
        "status": "bioaware_netid_dreams_feasibility_blocked",
        "formal": True,
        "dataset": DATASET,
        "registry_files": len(rows),
        "yeast_open_files": len(yeast_open),
        "yeast_explicit_ms2_files": len(explicit_ms2),
        "liver_targeted_ms2_files": len(liver_targeted),
        "netid_task": "MS1 peak/formula/ion-relation global annotation",
        "dreams_task": "MS2 structure-level spectral retrieval",
        "decision": (
            "Do not use the 314 yeast manual-curation records as a DreaMS/BioAware "
            "structure-ranking benchmark. Retain NetID for the separate MS1 formula "
            "and feature-graph task."
        ),
        "reason": (
            "The public yeast collection is explicitly MS1 and contains no named "
            "yeast targeted-MS2 files; the dataset's targeted-MS2 collection belongs "
            "to liver. A formula-level manual curation cannot be silently converted "
            "into an MS2 structure-retrieval endpoint."
        ),
        "manual_outcome_opened": False,
        "contract_sha256": sha256(args.contract),
        "registry_query": "GNPS2 datasetcache filename table, metadata only",
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError(f"fail-closed: existing report differs: {args.output}")
        print(f"[reuse] verified NetID feasibility stop: {args.output}", flush=True)
    else:
        write_atomic(args.output, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
