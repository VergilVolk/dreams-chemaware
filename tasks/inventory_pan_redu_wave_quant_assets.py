#!/usr/bin/env python3
"""Inventory deposited quantitative feature tables and MS2 exports for a wave."""

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
QUANT_RE = re.compile(r"quant|feature.?table|peak.?table|bucket|mzmine|xcms|area.?table", re.I)
EXCLUDE_RE = re.compile(r"(?:^|/)(?:raw|peak)/.*\.(?:raw|d|mzml|mzxml)$|log|parameter", re.I)


def query(accession: str) -> list[list[object]]:
    sql = (
        "select filepath,collection,size from filename "
        f"where dataset='{accession}' and size > 0 and size <= 2000000000 and ("
        "lower(filepath) like '%quant%' or lower(filepath) like '%feature%table%' or "
        "lower(filepath) like '%peak%table%' or lower(filepath) like '%bucket%' or "
        "lower(filepath) like '%mzmine%' or lower(filepath) like '%xcms%' or "
        "lower(filepath) like '%.mgf') order by filepath limit 5000"
    )
    url = API + urllib.parse.quote(sql)
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response).get("rows", [])
        except Exception:
            if attempt == 5:
                raise
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    wave = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    failures = []
    for index, accession in enumerate(sorted(wave["dataset_accession"].unique()), start=1):
        try:
            inventory = query(accession)
        except Exception as error:
            failures.append({"dataset_accession": accession, "error": repr(error)})
            continue
        for filepath, collection, size in inventory:
            filepath = str(filepath)
            if EXCLUDE_RE.search(filepath):
                continue
            suffix = Path(filepath).suffix.lower()
            kind = "ms2_export" if suffix == ".mgf" else "quantitative_candidate" if QUANT_RE.search(filepath) else ""
            if not kind or suffix not in {".mgf", ".csv", ".tsv", ".txt", ".xlsx", ".xls"}:
                continue
            descriptor = f"f.{accession}/{filepath}"
            rows.append({
                "dataset_accession": accession,
                "asset_kind": kind,
                "filepath": filepath,
                "collection": str(collection),
                "bytes": int(size),
                "remote_descriptor": descriptor,
                "download_url": (
                    "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
                    + urllib.parse.urlencode({"file": descriptor})
                ),
                "qualification_status": "path_candidate_contents_unopened",
            })
        print(f"[{index}/{wave['dataset_accession'].nunique()}] {accession}", flush=True)
    assets = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets.to_csv(args.output_dir / "quant_asset_inventory.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "inventory_failures.tsv", sep="\t", index=False)
    paired = 0
    if len(assets):
        counts = assets.groupby(["dataset_accession", "asset_kind"]).size().unstack(fill_value=0)
        paired = int(((counts.get("quantitative_candidate", 0) > 0) & (counts.get("ms2_export", 0) > 0)).sum())
    report = {
        "schema": "pan_redu_wave_quant_asset_inventory_v1",
        "denominators": {
            "accessions": int(wave["dataset_accession"].nunique()),
            "candidate_assets": len(assets),
            "candidate_bytes": int(assets["bytes"].sum()) if len(assets) else 0,
            "accessions_with_quant_candidate": int(
                assets.loc[assets["asset_kind"].eq("quantitative_candidate"), "dataset_accession"].nunique()
            ) if len(assets) else 0,
            "accessions_with_mgf": int(
                assets.loc[assets["asset_kind"].eq("ms2_export"), "dataset_accession"].nunique()
            ) if len(assets) else 0,
            "accessions_with_both_path_candidates": paired,
            "inventory_failures": len(failures),
        },
        "claim_boundary": (
            "Path candidates are not qualified quantitative-MS2 pairs until schemas, sample coverage, "
            "feature-to-spectrum keys, processing leakage, and biological design are audited."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
