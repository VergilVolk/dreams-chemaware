"""Phenotype-blind QC/blank/dilution headroom gate for LCNEC HSST3n.

This second rejection gate never reads the tumor/control labels.  It clusters
DDA precursor events from pooled QC, method blanks, and the frozen QC dilution
series, then asks how many precursor/RT families are reproducible, blank-clean,
and concentration responsive.  Passing only justifies an author-overlap and
annotation audit; it is not evidence for disease biology or a new metabolite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN


MS_LEVEL = "MS:1000511"
SCAN_START_TIME = "MS:1000016"
SELECTED_ION_MZ = "MS:1000744"
SELECTED_ION_INTENSITY = "MS:1000042"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def cv_params(element: ET.Element) -> dict[str, tuple[str | None, str | None]]:
    values: dict[str, tuple[str | None, str | None]] = {}
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] != "cvParam":
            continue
        accession = child.attrib.get("accession")
        value = child.attrib.get("value")
        if accession and value not in (None, ""):
            values[accession] = (value, child.attrib.get("unitName"))
    return values


def stream_ms2(handle: BinaryIO, file_id: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for _event, element in ET.iterparse(handle, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "spectrum":
            continue
        params = cv_params(element)
        if number(params.get(MS_LEVEL, (None, None))[0]) != 2:
            element.clear()
            continue
        mz = number(params.get(SELECTED_ION_MZ, (None, None))[0])
        intensity = number(params.get(SELECTED_ION_INTENSITY, (None, None))[0])
        rt_value, rt_unit = params.get(SCAN_START_TIME, (None, None))
        rt = number(rt_value)
        if rt_unit and "minute" in rt_unit.lower():
            rt *= 60.0
        if math.isfinite(mz) and math.isfinite(rt):
            rows.append({"file_id": file_id, "mz": mz, "rt_sec": rt, "precursor_intensity": intensity})
        element.clear()
    return rows


def injection_class(note: str) -> str:
    if note == "QC sample":
        return "pooled_qc"
    if note == "Method blank":
        return "blank"
    if note.startswith("Serial dilution"):
        return "qc_dilution"
    return "study"


def dilution_fraction(sample_id: str) -> float:
    return {
        "SD-A-0": 0.0,
        "SD-A-1-16": 1 / 16,
        "SD-A-1-8": 1 / 8,
        "SD-A-1-4": 1 / 4,
        "SD-A-1-2": 1 / 2,
        "SD-A-1": 1.0,
    }.get(sample_id, math.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_HSST3n_mzML_public.zip"),
    )
    parser.add_argument(
        "--overview",
        type=Path,
        default=Path("data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/validation/lcnec_hsst3n_qc_headroom_gate")
    )
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    args = parser.parse_args()

    with args.overview.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))
    controls = [row for row in ledger if injection_class(row["NOTE"]) != "study"]
    if len(controls) != 17:
        raise RuntimeError(f"expected 17 phenotype-blind control injections, found {len(controls)}")

    records: list[dict[str, float | int]] = []
    file_meta: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if info.filename.lower().endswith(".mzml")}
        for file_id, row in enumerate(controls):
            name = row["mzML_FILE_NAME"]
            if name not in members:
                raise RuntimeError(f"control mzML missing from ZIP: {name}")
            meta = {
                "file_id": file_id,
                "name": name,
                "sample_id": row["SAMPLE_ID"],
                "class": injection_class(row["NOTE"]),
                "dilution": dilution_fraction(row["SAMPLE_ID"]),
            }
            file_meta.append(meta)
            with archive.open(members[name]) as handle:
                current = stream_ms2(handle, file_id)
            records.extend(current)
            print(f"[QC headroom] {file_id + 1}/17 {row['SAMPLE_ID']} MS2={len(current):,}", flush=True)

    mz = np.asarray([float(row["mz"]) for row in records])
    rt = np.asarray([float(row["rt_sec"]) for row in records])
    scaled = np.column_stack((np.log(mz) / (args.ppm * 1e-6), rt / args.rt_sec))
    labels = DBSCAN(eps=1.0, min_samples=1, metric="chebyshev", n_jobs=-1).fit_predict(scaled)
    for row, label in zip(records, labels, strict=True):
        row["family_id"] = int(label)

    meta_by_id = {int(row["file_id"]): row for row in file_meta}
    family_rows: list[dict[str, object]] = []
    for family_id in sorted(set(int(value) for value in labels)):
        subset = [row for row in records if int(row["family_id"]) == family_id]
        per_file: dict[int, float] = {}
        for row in subset:
            value = float(row["precursor_intensity"])
            if not math.isfinite(value):
                value = 0.0
            file_id = int(row["file_id"])
            per_file[file_id] = max(per_file.get(file_id, 0.0), value)

        qc_ids = [int(row["file_id"]) for row in file_meta if row["class"] == "pooled_qc"]
        blank_ids = [int(row["file_id"]) for row in file_meta if row["class"] == "blank"]
        dilution_ids = [int(row["file_id"]) for row in file_meta if row["class"] == "qc_dilution"]
        qc_values = [per_file.get(file_id, 0.0) for file_id in qc_ids]
        blank_values = [per_file.get(file_id, 0.0) for file_id in blank_ids]
        dilution_values = [per_file.get(file_id, 0.0) for file_id in dilution_ids]
        dilution_levels = [float(meta_by_id[file_id]["dilution"]) for file_id in dilution_ids]
        rho = float(spearmanr(dilution_levels, np.log1p(dilution_values)).statistic)
        if not math.isfinite(rho):
            rho = 0.0
        qc_detected = sum(value > 0 for value in qc_values)
        blank_detected = sum(value > 0 for value in blank_values)
        qc_median = statistics.median([value for value in qc_values if value > 0]) if qc_detected else 0.0
        blank_max = max(blank_values, default=0.0)
        blank_ratio = blank_max / qc_median if qc_median > 0 else math.inf
        reproducible = qc_detected >= 6
        blank_clean = blank_detected == 0 or blank_ratio <= 0.20
        dilution_responsive = rho >= 0.70
        family_rows.append(
            {
                "family_id": family_id,
                "mz_median": statistics.median(float(row["mz"]) for row in subset),
                "rt_median_sec": statistics.median(float(row["rt_sec"]) for row in subset),
                "ms2_events": len(subset),
                "files_detected": len(per_file),
                "qc_detected": qc_detected,
                "blank_detected": blank_detected,
                "dilution_detected": sum(value > 0 for value in dilution_values),
                "qc_median_precursor_intensity": qc_median,
                "blank_to_qc_ratio": blank_ratio,
                "dilution_spearman_rho": rho,
                "qc_reproducible": reproducible,
                "blank_clean": blank_clean,
                "dilution_responsive": dilution_responsive,
                "passes_all": reproducible and blank_clean and dilution_responsive,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "precursor_family_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)

    passing = [row for row in family_rows if row["passes_all"]]
    report = {
        "status": "lcnec_hsst3n_qc_headroom_complete",
        "formal": True,
        "phenotype_labels_read": False,
        "control_files": len(controls),
        "ms2_events": len(records),
        "precursor_rt_families": len(family_rows),
        "qc_reproducible_families": sum(bool(row["qc_reproducible"]) for row in family_rows),
        "blank_clean_families": sum(bool(row["blank_clean"]) for row in family_rows),
        "dilution_responsive_families": sum(bool(row["dilution_responsive"]) for row in family_rows),
        "all_three_qualified_families": len(passing),
        "gates": {
            "qualified_families_ge_100": len(passing) >= 100,
            "qualified_families_ge_250": len(passing) >= 250,
        },
        "pass_to_author_overlap_audit": len(passing) >= 100,
        "parameters": {"ppm": args.ppm, "rt_sec": args.rt_sec, "qc_min_files": 6, "blank_ratio_max": 0.20, "dilution_rho_min": 0.70},
        "provenance": {"zip_sha256": sha256(args.zip), "overview_sha256": sha256(args.overview)},
        "claim_limit": "Phenotype-blind acquisition-qualified MS2 families only; not author-novel, annotated, differential, or biological.",
    }
    (args.output_dir / "qc_headroom_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
