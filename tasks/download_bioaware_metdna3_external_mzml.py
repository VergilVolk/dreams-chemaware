#!/usr/bin/env python
"""Low-concurrency, resumable downloader for the frozen 8-unit external panel."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from download_bioaware_metdna3_development_mzml import download, sha256


def fetch_unit(unit_dir: Path, output_root: Path) -> dict:
    manifest = unit_dir / "download_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    unit = str(payload["unit_id"])
    files = payload["files"]
    expected = 15 if unit == "Mouse_brain__rplc" else 16
    if len(files) != expected:
        raise RuntimeError(f"{unit}: expected {expected} files, got {len(files)}")
    destination_dir = output_root / unit
    destination_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    for index, row in enumerate(files, 1):
        destination = destination_dir / str(row["local_name"])
        print(f"[{unit} {index}/{len(files)}] {destination.name}", flush=True)
        download(row, destination)
        installed.append({
            "name": destination.name, "bytes": destination.stat().st_size,
            "sha256": sha256(destination), "source_path": row["filepath"],
        })
    report = {
        "status": "bioaware_metdna3_external_unit_mzml_complete",
        "scope": "external", "unit_id": unit, "files": installed,
        "total_bytes": int(sum(row["bytes"] for row in installed)),
        "manifest_sha256": sha256(manifest),
    }
    report_path = destination_dir / "download_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError(f"existing report differs: {unit}")
    else:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path(
        "data/validation/bioaware_metdna3_external_manifest_v1"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "data/external/metdna3_2025/mzml/external"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise RuntimeError("MassIVE rate-limit contract allows only 1-2 download workers")
    root_report = json.loads((args.manifest_root / "report.json").read_text(encoding="utf-8"))
    if root_report.get("status") != "bioaware_metdna3_external_manifest_frozen":
        raise RuntimeError("external manifest is not frozen")
    units = [args.manifest_root / name for name in sorted(root_report["units"])]
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_unit, unit, args.output_root): unit.name for unit in units}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(f"[complete] {report['unit_id']} {report['total_bytes']:,} bytes", flush=True)
    total_files = sum(len(report["files"]) for report in reports)
    if total_files != 127:
        raise RuntimeError(f"external download incomplete: {total_files}/127")
    final = {
        "status": "bioaware_metdna3_external_mzml_complete",
        "units": {report["unit_id"]: report for report in sorted(reports, key=lambda row: row["unit_id"])},
        "files": total_files,
        "total_bytes": int(sum(report["total_bytes"] for report in reports)),
        "manifest_report_sha256": sha256(args.manifest_root / "report.json"),
    }
    final_path = args.output_root / "download_report.json"
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: final[key] for key in ("status", "files", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
