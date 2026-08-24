#!/usr/bin/env python
"""Role-aware B0 audit for the MTBLS1905 positive-HILIC raw files.

The audit fixes the input denominator before feature extraction.  It counts
MS1/MS2 scans, checks that mzML files are readable, and records acquisition
ranges for patient, pooled-QC and blank files.  It performs no annotation or
differential statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pyteomics import mzml


def scan_rt(spec: dict[str, Any]) -> float | None:
    try:
        return float(spec["scanList"]["scan"][0]["scan start time"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def audit_file(path: Path) -> dict[str, Any]:
    n_spectra = n_ms1 = n_ms2 = n_other = 0
    n_ms1_empty = 0
    rt_first = rt_last = None
    ms1_mz_min = ms1_mz_max = None
    with mzml.read(str(path), use_index=False) as reader:
        for spec in reader:
            n_spectra += 1
            level = int(spec.get("ms level", 0) or 0)
            rt = scan_rt(spec)
            if rt is not None:
                rt_first = rt if rt_first is None else min(rt_first, rt)
                rt_last = rt if rt_last is None else max(rt_last, rt)
            if level == 1:
                n_ms1 += 1
                mzs = spec.get("m/z array", [])
                if len(mzs) == 0:
                    n_ms1_empty += 1
                else:
                    local_min = float(mzs[0])
                    local_max = float(mzs[-1])
                    ms1_mz_min = local_min if ms1_mz_min is None else min(ms1_mz_min, local_min)
                    ms1_mz_max = local_max if ms1_mz_max is None else max(ms1_mz_max, local_max)
            elif level == 2:
                n_ms2 += 1
            else:
                n_other += 1
    return {
        "n_spectra": n_spectra,
        "n_ms1": n_ms1,
        "n_ms2": n_ms2,
        "n_other": n_other,
        "n_ms1_empty": n_ms1_empty,
        "rt_first": rt_first,
        "rt_last": rt_last,
        "ms1_mz_min": ms1_mz_min,
        "ms1_mz_max": ms1_mz_max,
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, parameters: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["sample_name", "status"]
    with (output_dir / "positive_hilic_file_audit.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    status_counts: dict[str, int] = {}
    role_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row["status"])
        role = str(row["sample_role"])
        status_counts[status] = status_counts.get(status, 0) + 1
        role_counts.setdefault(role, {})[status] = role_counts.setdefault(role, {}).get(status, 0) + 1
    report = {
        "study": "MTBLS1905",
        "stage": "B0 positive-HILIC file audit",
        "purpose": "input completeness and acquisition audit; not annotation evidence",
        "parameters": parameters,
        "n_manifest_rows": len(rows),
        "status_counts": status_counts,
        "role_status_counts": role_counts,
        "ready_for_b1": bool(rows) and all(row["status"] == "complete" for row in rows),
    }
    (output_dir / "positive_hilic_file_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/external/MTBLS1905/metadata/positive_ms1_processing_manifest.tsv"))
    parser.add_argument("--mzml-dir", type=Path, default=Path("data/external/MTBLS1905/positive_patients"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/MTBLS1905/ms1_b0_audit"))
    parser.add_argument("--roles", nargs="+", default=["patient", "pooled_qc", "blank"])
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N selected manifest rows for a smoke test.")
    parser.add_argument("--require-complete", action="store_true", help="Exit non-zero unless every selected manifest row is readable.")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = [row for row in csv.DictReader(handle, delimiter="\t") if row["sample_role"] in args.roles]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    if not manifest:
        raise ValueError("No manifest rows selected")

    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(manifest, 1):
        path = args.mzml_dir / item["file_name"]
        base = {
            "sample_name": item["sample_name"],
            "sample_role": item["sample_role"],
            "tissue_type": item.get("tissue_type", ""),
            "file_name": item["file_name"],
            "size_mib": round(path.stat().st_size / 2**20, 3) if path.is_file() else None,
            "status": None,
            "error": "",
            "n_spectra": None,
            "n_ms1": None,
            "n_ms2": None,
            "n_other": None,
            "n_ms1_empty": None,
            "rt_first": None,
            "rt_last": None,
            "ms1_mz_min": None,
            "ms1_mz_max": None,
        }
        if not path.is_file() or path.stat().st_size <= 1024:
            row = {**base, "status": "missing", "error": "file absent or <=1 KiB"}
        else:
            try:
                summary = audit_file(path)
                status = "complete" if summary["n_ms1"] > 0 else "invalid_no_ms1"
                row = {**base, "status": status, "error": "", **summary}
            except Exception as error:  # keep the full manifest denominator even on corrupt files
                row = {**base, "status": "unreadable", "error": repr(error)}
        rows.append(row)
        write_outputs(rows, args.output_dir, {"roles": args.roles, "limit": args.limit})
        print(f"[{ordinal}/{len(manifest)}] {item['file_name']}: {row['status']}", flush=True)

    report = json.loads((args.output_dir / "positive_hilic_file_audit.json").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2), flush=True)
    if args.require_complete and not report["ready_for_b1"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
