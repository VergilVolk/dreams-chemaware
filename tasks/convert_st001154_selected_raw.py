#!/usr/bin/env python
"""Convert one frozen ST001154 selection from Thermo RAW to mzML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_st001154_bioaware_external_readiness import checksum  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_extension_selection_v1"),
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("data/reference/ST001154_negative_pilot_20260901"),
    )
    parser.add_argument(
        "--exe", type=Path,
        default=Path("tools/external/ThermoRawFileParser-v2dev/app/ThermoRawFileParser.exe"),
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_extension_conversion_v1"),
    )
    args = parser.parse_args()
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise RuntimeError(f"fail-closed: report directory is non-empty: {args.report_dir}")
    selection_path = args.selection_dir / "samples.csv"
    selection_report_path = args.selection_dir / "report.json"
    for path in (selection_path, selection_report_path, args.exe):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    if checksum(selection_path) != selection_report["provenance"]["samples_sha256"]:
        raise RuntimeError("frozen selection hash mismatch")
    samples = pd.read_csv(selection_path)
    output_dir = args.raw_dir / "mzml"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(samples.itertuples(index=False), start=1):
        raw = args.raw_dir / str(row.FileName)
        mzml = output_dir / (raw.stem + ".mzML")
        if not raw.is_file() or raw.stat().st_size != int(row.archive_bytes):
            raise RuntimeError(f"missing or size-mismatched frozen RAW: {raw}")
        if mzml.exists():
            if mzml.stat().st_size == 0:
                raise RuntimeError(f"empty existing mzML: {mzml}")
            status = "reused"
        else:
            print(f"[convert {position}/{len(samples)}] {raw.name}", flush=True)
            completed = subprocess.run(
                [str(args.exe), f"-i={raw}", f"-o={output_dir}", "-f=1"],
                check=False, capture_output=True, text=True,
            )
            if completed.returncode != 0 or not mzml.is_file() or mzml.stat().st_size == 0:
                raise RuntimeError(
                    f"Thermo conversion failed for {raw.name}: rc={completed.returncode}; "
                    f"stdout={completed.stdout[-500:]!r}; stderr={completed.stderr[-500:]!r}"
                )
            status = "converted"
        records.append({
            "raw": str(raw), "raw_bytes": int(raw.stat().st_size),
            "raw_sha256": checksum(raw), "mzml": str(mzml),
            "mzml_bytes": int(mzml.stat().st_size), "mzml_sha256": checksum(mzml),
            "status": status,
        })
        print(f"[convert {position}/{len(samples)}] {status}: {mzml.name}", flush=True)
    args.report_dir.mkdir(parents=True)
    report = {
        "status": "st001154_selected_raw_conversion_complete",
        "samples": int(len(records)),
        "converted": int(sum(record["status"] == "converted" for record in records)),
        "reused": int(sum(record["status"] == "reused" for record in records)),
        "records": records,
        "provenance": {
            "selection_samples_sha256": checksum(selection_path),
            "selection_report_sha256": checksum(selection_report_path),
            "converter_sha256": checksum(args.exe),
            "script_sha256": checksum(Path(__file__)),
        },
    }
    (args.report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "samples", "converted", "reused")}, indent=2))


if __name__ == "__main__":
    main()
