"""Audit the frozen LCNEC HSST3n mzML acquisition without extracting features.

This is the first rejection gate for the LCNEC biology pivot.  It streams mzML
members directly from the public ZIP, verifies the 68-study/9-QC/2-blank/
6-dilution injection ledger, and measures whether sample and QC files contain
real DDA precursor coverage.  It does not annotate metabolites or inspect the
phenotype effect.
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
from collections import Counter
from pathlib import Path
from typing import BinaryIO


MS_LEVEL = "MS:1000511"
SCAN_START_TIME = "MS:1000016"
SELECTED_ION_MZ = "MS:1000744"
ISOLATION_TARGET_MZ = "MS:1000827"
COLLISION_ENERGY = "MS:1000045"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: str | None) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def cv_params(element: ET.Element) -> dict[str, tuple[str | None, str | None]]:
    values: dict[str, tuple[str | None, str | None]] = {}
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] != "cvParam":
            continue
        accession = child.attrib.get("accession")
        if accession:
            values[accession] = (child.attrib.get("value"), child.attrib.get("unitName"))
    return values


def stream_mzml(handle: BinaryIO) -> dict[str, object]:
    ms_levels: Counter[int] = Counter()
    precursor_values: list[float] = []
    collision_values: list[float] = []
    rt_values: list[float] = []
    missing_precursor = 0

    for _event, element in ET.iterparse(handle, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "spectrum":
            continue
        params = cv_params(element)
        level_value = number(params.get(MS_LEVEL, (None, None))[0])
        if math.isnan(level_value):
            element.clear()
            continue
        level = int(level_value)
        ms_levels[level] += 1

        rt, rt_unit = params.get(SCAN_START_TIME, (None, None))
        rt_float = number(rt)
        if not math.isnan(rt_float):
            if rt_unit and "minute" in rt_unit.lower():
                rt_float *= 60.0
            rt_values.append(rt_float)

        if level == 2:
            precursor = number(params.get(SELECTED_ION_MZ, (None, None))[0])
            if math.isnan(precursor):
                precursor = number(params.get(ISOLATION_TARGET_MZ, (None, None))[0])
            if math.isnan(precursor):
                missing_precursor += 1
            else:
                precursor_values.append(precursor)
            collision = number(params.get(COLLISION_ENERGY, (None, None))[0])
            if not math.isnan(collision):
                collision_values.append(collision)
        element.clear()

    unique_precursors = len({round(value, 4) for value in precursor_values})
    return {
        "ms1": ms_levels[1],
        "ms2": ms_levels[2],
        "other_ms": sum(count for level, count in ms_levels.items() if level not in (1, 2)),
        "ms2_with_precursor": len(precursor_values),
        "ms2_missing_precursor": missing_precursor,
        "unique_precursors_4dp": unique_precursors,
        "precursor_min": min(precursor_values) if precursor_values else math.nan,
        "precursor_max": max(precursor_values) if precursor_values else math.nan,
        "collision_energy_median": statistics.median(collision_values) if collision_values else math.nan,
        "rt_max_sec": max(rt_values) if rt_values else math.nan,
    }


def classify(note: str) -> str:
    if note == "Study sample":
        return "study"
    if note == "QC sample":
        return "pooled_qc"
    if note == "Method blank":
        return "blank"
    if note.startswith("Serial dilution"):
        return "qc_dilution"
    return "unknown"


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
        "--output-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_acquisition_gate"),
    )
    parser.add_argument("--maximum-files", type=int, default=0)
    args = parser.parse_args()

    if not args.zip.is_file() or not args.overview.is_file():
        raise FileNotFoundError("LCNEC HSST3n ZIP or overview is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.overview.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))
    if len(ledger) != 85:
        raise RuntimeError(f"expected 85 HSST3n ledger rows, found {len(ledger)}")

    ledger_by_name = {row["mzML_FILE_NAME"]: row for row in ledger}
    expected_class_counts = Counter(classify(row["NOTE"]) for row in ledger)
    if expected_class_counts != Counter({"study": 68, "pooled_qc": 9, "blank": 2, "qc_dilution": 6}):
        raise RuntimeError(f"unexpected injection ledger: {dict(expected_class_counts)}")

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        members = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if info.filename.lower().endswith(".mzml") and not info.is_dir()
        }
        missing = sorted(set(ledger_by_name) - set(members))
        extra = sorted(set(members) - set(ledger_by_name))
        if missing or extra:
            raise RuntimeError(f"ZIP/ledger mismatch: missing={missing[:5]} extra={extra[:5]}")

        selected = ledger if args.maximum_files <= 0 else ledger[: args.maximum_files]
        for index, ledger_row in enumerate(selected, 1):
            name = ledger_row["mzML_FILE_NAME"]
            with archive.open(members[name]) as mzml_handle:
                metrics = stream_mzml(mzml_handle)
            row = {
                "mzml_file": name,
                "injection_class": classify(ledger_row["NOTE"]),
                "sample_id": ledger_row["SAMPLE_ID"],
                "sample_code": ledger_row["SAMPLE_CODE"],
                "group_code": ledger_row["GROUP_CODE"],
                "note": ledger_row["NOTE"],
                **metrics,
            }
            rows.append(row)
            print(
                f"[LCNEC acquisition] {index}/{len(selected)} {name} "
                f"MS1={metrics['ms1']} MS2={metrics['ms2']} "
                f"precursors={metrics['unique_precursors_4dp']}",
                flush=True,
            )

    file_csv = args.output_dir / "file_acquisition_audit.csv"
    with file_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, dict[str, object]] = {}
    for injection_class in ("study", "pooled_qc", "blank", "qc_dilution"):
        subset = [row for row in rows if row["injection_class"] == injection_class]
        by_class[injection_class] = {
            "files": len(subset),
            "ms1": sum(int(row["ms1"]) for row in subset),
            "ms2": sum(int(row["ms2"]) for row in subset),
            "median_ms2_per_file": statistics.median(int(row["ms2"]) for row in subset) if subset else 0,
            "median_unique_precursors_per_file": statistics.median(
                int(row["unique_precursors_4dp"]) for row in subset
            ) if subset else 0,
        }

    formal = args.maximum_files <= 0
    gates = {
        "all_85_files_audited": len(rows) == 85,
        "study_has_ms2": by_class["study"]["ms2"] > 0,
        "qc_has_ms2": by_class["pooled_qc"]["ms2"] > 0,
        "study_median_unique_precursors_ge_100": by_class["study"]["median_unique_precursors_per_file"] >= 100,
        "qc_median_unique_precursors_ge_100": by_class["pooled_qc"]["median_unique_precursors_per_file"] >= 100,
    }
    report = {
        "status": "lcnec_hsst3n_acquisition_gate_complete",
        "formal": formal,
        "files_audited": len(rows),
        "injection_ledger": dict(expected_class_counts),
        "by_class": by_class,
        "gates": gates,
        "pass_to_feature_headroom": formal and all(gates.values()),
        "provenance": {
            "zip_sha256": sha256(args.zip),
            "overview_sha256": sha256(args.overview),
        },
        "claim_limit": "Acquisition and DDA coverage only; no feature, identity, phenotype, or biology result.",
    }
    (args.output_dir / "acquisition_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
