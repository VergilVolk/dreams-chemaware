#!/usr/bin/env python3
"""Inventory small deposited design/metadata assets for a systematic wave."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


API = "https://datasetcache.gnps2.org/datasette/database.json?sql="
BASE = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
ALLOWED = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
DESIGN_RE = re.compile(r"metadata|sample|design|phenotype|subject|cohort|clinical", re.I)
EXCLUDE_RE = re.compile(
    r"summary\.tsv$|parameter|workflow|log|readme|"
    r"(?:^|/)(?:raw|peak)/|lcparms\.txt$|nugenesistemplate\.txt$",
    re.I,
)


def query(accession: str) -> list[tuple[str, str, int]]:
    sql = (
        "select filepath,collection,size from filename "
        f"where dataset='{accession}' and size <= 20000000 and ("
        "lower(filepath) like '%metadata%' or lower(filepath) like '%sample%' or "
        "lower(filepath) like '%design%' or lower(filepath) like '%phenotype%' or "
        "lower(filepath) like '%subject%' or lower(filepath) like '%cohort%' or "
        "lower(filepath) like '%clinical%') order by filepath limit 5000"
    )
    url = API + urllib.parse.quote(sql)
    for attempt in range(1, 6):
        request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(attempt * 2)
    return [(str(row[0]), str(row[1]), int(row[2])) for row in payload.get("rows", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--request-delay", type=float, default=0.25)
    args = parser.parse_args()
    wave = pd.read_csv(args.wave, sep="\t", dtype=str, keep_default_na=False)
    accessions = sorted(wave["dataset_accession"].unique())
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, accession in enumerate(accessions, start=1):
        try:
            inventory = query(accession)
        except Exception as error:
            failures.append({"dataset_accession": accession, "error": repr(error)})
            continue
        for filepath, collection, size in inventory:
            suffix = Path(filepath).suffix.lower()
            if suffix not in ALLOWED or not DESIGN_RE.search(filepath) or EXCLUDE_RE.search(filepath):
                continue
            descriptor = f"f.{accession}/{filepath}"
            rows.append({
                "dataset_accession": accession,
                "filepath": filepath,
                "collection": collection,
                "bytes": size,
                "remote_descriptor": descriptor,
                "download_url": BASE + urllib.parse.urlencode({"file": descriptor}),
                "selection_role": "candidate_design_asset_not_yet_adjudicated",
            })
        print(f"[{index}/{len(accessions)}] {accession}: {len(inventory)} small files scanned", flush=True)
        time.sleep(args.request_delay)

    assets = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets.to_csv(args.output_dir / "design_asset_inventory.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "inventory_failures.tsv", sep="\t", index=False)
    with_assets = set(assets["dataset_accession"]) if len(assets) else set()
    report = {
        "schema": "pan_redu_wave_design_asset_inventory_v1",
        "denominators": {
            "wave_accessions": len(accessions),
            "accessions_with_candidate_design_assets": len(with_assets),
            "accessions_without_candidate_design_assets": len(set(accessions) - with_assets),
            "candidate_assets": len(assets),
            "candidate_asset_bytes": int(assets["bytes"].sum()) if len(assets) else 0,
            "inventory_failures": len(failures),
        },
        "accessions_without_candidate_assets": sorted(set(accessions) - with_assets),
        "leakage_status": "inventory_paths_only_asset_contents_unopened",
        "claim_boundary": "Filename selection does not establish that an asset defines independent biological units.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
