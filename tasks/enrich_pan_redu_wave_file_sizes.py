#!/usr/bin/env python3
"""Attach deposited byte sizes to the frozen systematic-wave file manifest."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


API = "https://datasetcache.gnps2.org/datasette/database.json?sql="


def query(accession: str) -> list[list[object]]:
    sql = (
        "select filepath,collection,size from filename "
        f"where dataset='{accession}' and (lower(filepath) like '%.mzml' "
        "or lower(filepath) like '%.mzxml') order by filepath limit 10000"
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
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    observed: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for accession, block in manifest.groupby("dataset_accession", sort=True):
        prefix = f"f.{accession}/"
        relative = [
            value[len(prefix):] if value.startswith(prefix) else value
            for value in block["remote_descriptor"]
        ]
        for filepath, collection, size in query(accession):
            observed.setdefault((accession, str(filepath)), []).append((str(collection), int(size)))
        print(f"{accession}: requested={len(set(relative))} matched={sum((accession, p) in observed for p in set(relative))}", flush=True)

    output = manifest.copy()
    deposited_bytes = []
    collection_values = []
    match_counts = []
    for row in output.to_dict("records"):
        accession = row["dataset_accession"]
        prefix = f"f.{accession}/"
        relative = row["remote_descriptor"][len(prefix):]
        matches = observed.get((accession, relative), [])
        match_counts.append(len(matches))
        if matches:
            # Identical logical paths in update collections should have the
            # same bytes. A disagreement is retained as ambiguity, not hidden.
            sizes = {size for _, size in matches}
            deposited_bytes.append(next(iter(sizes)) if len(sizes) == 1 else -1)
            collection_values.append(";".join(sorted({collection for collection, _ in matches})))
        else:
            deposited_bytes.append(0)
            collection_values.append("")
    output["inventory_match_count"] = match_counts
    output["deposited_bytes"] = deposited_bytes
    output["inventory_collections"] = collection_values
    output["byte_status"] = [
        "verified_inventory" if count > 0 and size > 0 else
        "ambiguous_inventory" if size < 0 else "missing_inventory"
        for count, size in zip(match_counts, deposited_bytes)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "unique_file_manifest_with_sizes.tsv", sep="\t", index=False)
    valid = output["deposited_bytes"] > 0
    report = {
        "schema": "pan_redu_systematic_wave_file_sizes_v1",
        "denominators": {
            "manifest_files": len(output),
            "files_with_unambiguous_size": int(valid.sum()),
            "files_missing_from_inventory": int((output["deposited_bytes"] == 0).sum()),
            "files_with_ambiguous_size": int((output["deposited_bytes"] < 0).sum()),
            "selected_file_bytes": int(output.loc[valid, "deposited_bytes"].sum()),
            "selected_file_gib": float(output.loc[valid, "deposited_bytes"].sum() / 2**30),
        },
        "leakage_status": "acquisition_inventory_only_no_abundance_or_truth_read",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
