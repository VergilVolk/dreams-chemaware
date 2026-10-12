#!/usr/bin/env python
"""Extract an annotation-isolated abundance matrix from MSV000096214 Table S1.

The primary output deliberately contains no structure, molecular-family, author
network, or annotation-level fields.  Those fields are written to a separate
sidecar whose hash is recorded but whose values are not summarized.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


DATASET = "MSV000096214"
PMC_ID = "PMC12189940"
SUPPLEMENT_URL = (
    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMC_ID}/supplementaryFiles"
)
WORKBOOK_NAME = "TPJ-122-0-s002.xlsx"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def column_number(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(reference)
    number = 0
    for letter in letters.group(0):
        number = number * 26 + ord(letter) - 64
    return number


def parse_sheet(workbook: bytes) -> dict[int, dict[int, str]]:
    bundle = zipfile.ZipFile(io.BytesIO(workbook))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in bundle.namelist():
        root = ET.fromstring(bundle.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//m:t", NS))
            for item in root.findall("m:si", NS)
        ]
    root = ET.fromstring(bundle.read("xl/worksheets/sheet1.xml"))
    rows: dict[int, dict[int, str]] = {}
    for row in root.findall(".//m:sheetData/m:row", NS):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            index = column_number(cell.attrib["r"])
            if cell.attrib.get("t") == "inlineStr":
                values[index] = "".join(
                    node.text or "" for node in cell.findall(".//m:t", NS)
                )
                continue
            value = cell.find("m:v", NS)
            if value is None:
                values[index] = ""
            elif cell.attrib.get("t") == "s":
                values[index] = shared[int(value.text)]
            else:
                values[index] = value.text or ""
        rows[int(row.attrib["r"])] = values
    return rows


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


def write_deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    request = urllib.request.Request(
        SUPPLEMENT_URL, headers={"User-Agent": "DreaMS-biology/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        archive = response.read()
    supplement = zipfile.ZipFile(io.BytesIO(archive))
    workbook = supplement.read(WORKBOOK_NAME)
    rows = parse_sheet(workbook)

    # Frozen schema: B:E are structure-free feature fields; M:BI are the 49
    # actually deposited biological samples. J:L are pooled QCs. F:I and all
    # fields after BI are excluded from the primary table.
    header = rows[6]
    sample_columns = list(range(13, 62))  # M:BI, exactly 49.
    if len(sample_columns) != 49 or any(column not in header for column in sample_columns):
        raise ValueError("Table S1 no longer has the frozen M:BI 49-sample schema")

    sample_records = []
    renamed: dict[int, str] = {}
    for position, column in enumerate(sample_columns, start=1):
        source_name = header[column]
        genotype_raw = rows[5].get(column, "").strip()
        genotype = normalize_genotype(genotype_raw)
        if not genotype:
            raise ValueError(f"missing genotype for source column {source_name}")
        sample_id = f"sample_{position:03d}"
        renamed[column] = sample_id
        sample_records.append({
            "sample_id": sample_id,
            "source_column": source_name,
            "genotype": genotype,
            "genotype_raw": genotype_raw,
            "author_replicate": rows[4].get(column, ""),
            "deposited_raw_pair_expected": True,
        })
    sample_manifest = pd.DataFrame(sample_records)

    primary_records = []
    annotation_records = []
    qc_records = []
    for row_number in range(7, 927):
        row = rows.get(row_number, {})
        feature_id = row.get(2, "").strip()
        if not feature_id:
            continue
        primary = {
            "feature_id": feature_id,
            "ion_mode": row.get(3, "").strip(),
            "mz": row.get(4, ""),
            "rt_min": row.get(5, ""),
        }
        for column in sample_columns:
            primary[renamed[column]] = row.get(column, "")
        primary_records.append(primary)
        qc_records.append({
            "feature_id": feature_id,
            "pooled_qc_1": row.get(10, ""),
            "pooled_qc_2": row.get(11, ""),
            "pooled_qc_3": row.get(12, ""),
        })
        annotation_records.append({
            "feature_id": feature_id,
            "annotation": row.get(6, ""),
            "metabolite_family": row.get(7, ""),
            "annotation_level": row.get(8, ""),
            "author_molecular_network_cluster": row.get(9, ""),
        })

    primary_frame = pd.DataFrame(primary_records)
    qc_frame = pd.DataFrame(qc_records)
    annotation_frame = pd.DataFrame(annotation_records)
    if len(primary_frame) != 920 or not primary_frame["feature_id"].is_unique:
        raise ValueError("expected 920 unique Table S1 features")
    numeric_columns = ["mz", "rt_min", *sample_manifest["sample_id"].tolist()]
    converted = primary_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    nonnumeric = int(converted.isna().sum().sum())
    primary_frame[numeric_columns] = converted

    write_deterministic_csv_gz(primary_frame, args.output / "structure_hidden_abundance.csv.gz")
    write_deterministic_csv_gz(qc_frame, args.output / "pooled_qc_abundance.csv.gz")
    write_deterministic_csv_gz(annotation_frame, args.output / "SEALED_annotations.csv.gz")
    write_deterministic_csv_gz(sample_manifest, args.output / "sample_manifest.csv.gz")

    genotypes = sample_manifest.groupby("genotype").size().sort_index().to_dict()
    ion_modes = primary_frame.groupby("ion_mode").size().sort_index().to_dict()
    report = {
        "status": "MSV000096214_STRUCTURE_HIDDEN_TABLE_EXTRACTED",
        "formal": True,
        "dataset": DATASET,
        "data_role": "development/consumed",
        "sealed": False,
        "annotation_values_used_for_analysis": False,
        "source": {
            "pmc_id": PMC_ID,
            "url": SUPPLEMENT_URL,
            "workbook": WORKBOOK_NAME,
            "supplement_archive_sha256": sha256(archive),
            "workbook_sha256": sha256(workbook),
        },
        "matrix": {
            "features": int(len(primary_frame)),
            "samples": int(len(sample_manifest)),
            "genotype_sample_counts": {str(k): int(v) for k, v in genotypes.items()},
            "ion_mode_feature_counts": {str(k): int(v) for k, v in ion_modes.items()},
            "nonnumeric_or_missing_primary_cells": nonnumeric,
            "primary_columns": ["feature_id", "ion_mode", "mz", "rt_min", "49 sample abundances"],
            "excluded_from_primary": [
                "annotation", "metabolite_family", "annotation_level",
                "author_molecular_network_cluster", "author summary/formula fields",
            ],
        },
        "output_sha256": {
            path.name: sha256(path.read_bytes())
            for path in sorted(args.output.iterdir()) if path.is_file()
        },
        "claim_limit": (
            "Schema and annotation-isolated abundance extraction only. No genotype effect, "
            "published differential result, annotation stratum, or algorithm performance was computed."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
