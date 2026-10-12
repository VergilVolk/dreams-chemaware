#!/usr/bin/env python
"""Truth-blind readiness audit for the Arabidopsis flavonoid-mutant LC-MS/MS study."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd


DATASET = "MSV000096214"
BASE = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


_LAST_REQUEST = 0.0


def fetch(remote: str) -> bytes:
    global _LAST_REQUEST
    url = BASE + urllib.parse.urlencode({"file": f"f.{DATASET}/{remote}"})
    for attempt in range(6):
        elapsed = time.monotonic() - _LAST_REQUEST
        if elapsed < 0.75:
            time.sleep(0.75 - elapsed)
        request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-biology/1.0"})
        try:
            _LAST_REQUEST = time.monotonic()
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * (attempt + 1)
            time.sleep(min(delay, 30.0))
    raise RuntimeError(f"unreachable fetch failure: {remote}")


def query_inventory() -> list[dict[str, object]]:
    sql = f"select filepath,collection,size from filename where dataset='{DATASET}' order by filepath"
    url = "https://datasetcache.gnps2.org/datasette/database.json?sql=" + urllib.parse.quote(sql)
    with urllib.request.urlopen(url, timeout=180) as response:
        rows = json.load(response)["rows"]
    return [{"filepath": str(row[0]), "collection": str(row[1]), "bytes": int(row[2])}
            for row in rows]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def column_number(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - 64
    return number


def read_sample_rows(workbook_bytes: bytes) -> dict[int, dict[int, str]]:
    bundle = zipfile.ZipFile(io.BytesIO(workbook_bytes))
    shared_root = ET.fromstring(bundle.read("xl/sharedStrings.xml"))
    shared = ["".join(t.text or "" for t in item.findall(".//m:t", NS))
              for item in shared_root.findall("m:si", NS)]
    root = ET.fromstring(bundle.read("xl/worksheets/sheet1.xml"))
    selected: dict[int, dict[int, str]] = {}
    for row in root.findall(".//m:sheetData/m:row", NS):
        index = int(row.attrib["r"])
        if index not in {12, 13, 14, 16}:
            continue
        values = {}
        for cell in row.findall("m:c", NS):
            value = cell.find("m:v", NS)
            if value is None:
                continue
            text = shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
            values[column_number(cell.attrib["r"])] = text
        selected[index] = values
    return selected


def normalize_genotype(value: str) -> str:
    value = value.strip()
    if value == "Col-0":
        return "WT_Col0"
    if value.startswith("115 (EMS"):
        return "tt7_115_EMS"
    if value.startswith("tt7-8 tt4-11"):
        return "tt7_8__tt4_11"
    if value.startswith("tt7-9 tt4-11"):
        return "tt7_9__tt4_11"
    if value.startswith("115 tt4-11"):
        return "tt7_115__tt4_11"
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    inventory = query_inventory()
    workbook = fetch("metadata/FAIR-Metadata_TT7.xlsx")
    params = fetch("ccms_parameters/params.xml")
    rows = read_sample_rows(workbook)
    records = []
    for column in range(2, 56):  # B:BC; 50 samples, 3 pooled QCs, 1 blank.
        tube = rows[14].get(column, "")
        genotype_raw = rows[16].get(column, "")
        role = "study" if str(tube).isdigit() else "pooled_qc" if str(tube).startswith("CQ") else "blank"
        for polarity, row_number in (("positive", 12), ("negative", 13)):
            filename = rows[row_number].get(column, "")
            if not filename:
                continue
            remote = f"peak/{filename}"
            scan_remote = f"ccms_metadata/ccms_peak/{filename}.scans"
            records.append({
                "tube": tube,
                "role": role,
                "genotype_raw": genotype_raw if role == "study" else role,
                "genotype": normalize_genotype(genotype_raw) if role == "study" else role,
                "polarity": polarity,
                "filename": filename,
                "remote_path": remote,
                "scan_metadata_path": scan_remote,
            })
    frame = pd.DataFrame(records)
    inventory_map = {row["filepath"]: row for row in inventory}
    frame["deposited"] = frame["remote_path"].isin(inventory_map)
    frame["bytes"] = [int(inventory_map[path]["bytes"]) if path in inventory_map else 0
                      for path in frame["remote_path"]]
    frame["total_scans"] = 0
    frame["ms2_scans"] = 0
    frame["scan_metadata_sha256"] = ""
    reusable = {}
    if args.reuse_manifest:
        old = pd.read_csv(args.reuse_manifest)
        reusable = {
            str(row.scan_metadata_path): row
            for row in old.itertuples(index=False)
            if bool(row.deposited) and isinstance(row.scan_metadata_sha256, str)
            and len(row.scan_metadata_sha256) == 64
        }
    for index, row in frame.loc[frame["deposited"]].iterrows():
        if str(row["scan_metadata_path"]) in reusable:
            cached = reusable[str(row["scan_metadata_path"])]
            frame.loc[index, "total_scans"] = int(cached.total_scans)
            frame.loc[index, "ms2_scans"] = int(cached.ms2_scans)
            frame.loc[index, "scan_metadata_sha256"] = str(cached.scan_metadata_sha256)
            continue
        payload = fetch(str(row["scan_metadata_path"]))
        levels = [int(line.split("\t")[1]) for line in payload.decode("utf-8").splitlines() if line]
        frame.loc[index, "total_scans"] = len(levels)
        frame.loc[index, "ms2_scans"] = sum(level == 2 for level in levels)
        frame.loc[index, "scan_metadata_sha256"] = sha256_bytes(payload)
    frame.to_csv(args.output / "sample_manifest.csv.gz", index=False, compression="gzip")

    study = frame.loc[frame["role"] == "study"]
    deposited_study = study.loc[study["deposited"]]
    genotype_files = deposited_study.groupby(["genotype", "polarity"]).size().unstack(fill_value=0)
    ms2_by_genotype = deposited_study.groupby(["genotype", "polarity"])["ms2_scans"].sum().unstack(fill_value=0)
    missing = frame.loc[~frame["deposited"], ["tube", "role", "genotype", "polarity", "filename"]]
    report = {
        "status": "MSV000096214_PERTURBATION_READINESS_AUDITED",
        "formal": True,
        "dataset": DATASET,
        "data_role": "development/consumed",
        "sealed": False,
        "effect_or_annotation_results_opened": False,
        "repository_inventory": {
            "files": len(inventory),
            "bytes": sum(int(row["bytes"]) for row in inventory),
            "peak_files": sum(row["collection"] == "peak" for row in inventory),
            "peak_bytes": sum(int(row["bytes"]) for row in inventory if row["collection"] == "peak"),
        },
        "design": {
            "registered_biological_samples": int(study["tube"].nunique()),
            "registered_study_files_two_polarities": int(len(study)),
            "deposited_study_files": int(len(deposited_study)),
            "paired_polarity_samples_complete": int(
                deposited_study.groupby("tube")["polarity"].nunique().eq(2).sum()
            ),
            "genotypes": int(study["genotype"].nunique()),
            "genotype_files_by_polarity": genotype_files.to_dict(orient="index"),
            "pooled_qc_files": int((frame["role"] == "pooled_qc").sum()),
            "blank_files": int((frame["role"] == "blank").sum()),
            "missing_registered_files": missing.to_dict(orient="records"),
        },
        "scan_census": {
            "deposited_study_total_scans": int(deposited_study["total_scans"].sum()),
            "deposited_study_ms2_scans": int(deposited_study["ms2_scans"].sum()),
            "pooled_qc_ms2_scans": int(frame.loc[(frame["role"] == "pooled_qc") & frame["deposited"], "ms2_scans"].sum()),
            "ms2_by_genotype_and_polarity": ms2_by_genotype.to_dict(orient="index"),
            "study_files_with_ms2": int((deposited_study["ms2_scans"] > 0).sum()),
        },
        "gates": {
            "at_least_three_independent_tt7_lines": all(
                genotype in set(study["genotype"])
                for genotype in ("tt7_115_EMS", "tt7_8_SALK_039417", "tt7_9_SALK_seq_064306")
            ),
            "at_least_four_replicates_per_genotype_and_polarity": bool((genotype_files >= 4).all().all()),
            "paired_positive_negative_for_at_least_49_samples": int(
                deposited_study.groupby("tube")["polarity"].nunique().eq(2).sum()
            ) >= 49,
            "all_deposited_study_files_contain_ms2": bool((deposited_study["ms2_scans"] > 0).all()),
            "pooled_qc_contains_ms2": int(frame.loc[(frame["role"] == "pooled_qc") & frame["deposited"], "ms2_scans"].sum()) > 0,
        },
        "provenance": {
            "metadata_xlsx_sha256": sha256_bytes(workbook),
            "params_xml_sha256": sha256_bytes(params),
            "sample_manifest_sha256": hashlib.sha256((args.output / "sample_manifest.csv.gz").read_bytes()).hexdigest(),
        },
        "claim_limit": (
            "Repository, sample-design and scan-level readiness only. No abundance, differential feature, "
            "molecular-network, annotation or published biological outcome was used."
        ),
    }
    report["pass_to_truth_blind_feature_table_audit"] = all(report["gates"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
