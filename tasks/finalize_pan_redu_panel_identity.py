#!/usr/bin/env python3
"""Combine complete per-panel strict-library partitions without opening effects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jobs", required=True, type=Path)
    parser.add_argument("--partitions-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    jobs = pd.read_csv(args.panel_jobs, sep="\t", dtype=str)
    tables = []
    sources = []
    for panel_id in jobs.sort_values("job_index", key=lambda value: value.astype(int)).panel_id:
        root = args.partitions_root / panel_id
        report_path = root / "report.json"
        table_path = root / "identity_partition.tsv.gz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        allowed_status = {
            "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITION_FROZEN",
            "PHENOTYPE_BLIND_POLARITY_MATCHED_LIBRARY_PARTITION_FROZEN",
        }
        if report.get("status") not in allowed_status:
            raise RuntimeError(f"inadmissible identity partition: {panel_id}")
        if report.get("panel_id") != panel_id:
            raise RuntimeError(f"identity report panel mismatch: {panel_id}")
        panel_report_sha256 = str(report.get("provenance", {}).get("panel_report_sha256", ""))
        if not panel_report_sha256:
            raise RuntimeError(f"identity report lacks panel-object provenance: {panel_id}")
        actual_partition_sha256 = sha256(table_path)
        if report.get("provenance", {}).get("partition_sha256") != actual_partition_sha256:
            raise RuntimeError(f"identity partition checksum drift: {panel_id}")
        table = pd.read_csv(table_path, sep="\t", dtype=str)
        if not table.panel_id.eq(panel_id).all():
            raise RuntimeError(f"partition panel mismatch: {panel_id}")
        if int(report.get("denominators", {}).get("panel_anchors", -1)) != len(table):
            raise RuntimeError(f"identity partition row-count drift: {panel_id}")
        reported_supported = int(report.get("denominators", {}).get("library_supported", -1))
        observed_supported = int(table.identity_status.eq("library_supported").sum())
        if reported_supported != observed_supported:
            raise RuntimeError(f"identity-status denominator drift: {panel_id}")
        tables.append(table)
        sources.append({
            "panel_id": panel_id,
            "report_sha256": sha256(report_path),
            "partition_sha256": actual_partition_sha256,
            "panel_report_sha256": panel_report_sha256,
        })
    combined = pd.concat(tables, ignore_index=True)
    if combined.duplicated(["panel_id", "anchor_id"]).any():
        raise RuntimeError("combined identity table has duplicate panel/anchor rows")
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "identity_partition.tsv.gz"
    combined.to_csv(output, sep="\t", index=False, compression="gzip")
    counts = combined.identity_status.value_counts().astype(int).to_dict()
    report = {
        "schema": "pan_redu_strict_library_partition_v1",
        "status": "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITIONS_COMPLETE",
        "denominators": {"panels": int(combined.panel_id.nunique()), "anchors": len(combined), "identity_status": counts},
        "leakage_status": "mechanical phenotype-blind partition assembly; no effects or retrieval truth read",
        "provenance": {"panel_jobs_sha256": sha256(args.panel_jobs), "combined_partition_sha256": sha256(output), "sources": sources},
        "claim_boundary": "Unresolved means no qualifying frozen-library hit, not chemical novelty or absence from all libraries.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
