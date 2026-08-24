#!/usr/bin/env python
"""Checkpointed OpenMS MS1 feature detection for the MTBLS1905 HNSCC cohort.

This is the detection stage only.  It does not convert missing peaks to zero
and it does not test C/E/N differences; those operations are deferred until
retention-time alignment, consensus grouping, QC/blank filtering and targeted
gap filling have been audited.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pilot_mtbls13729_openms_features import detect_features, load_ms1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/external/MTBLS1905/metadata/positive_ms1_processing_manifest.tsv"))
    parser.add_argument("--mzml-dir", type=Path, default=Path("data/external/MTBLS1905/positive_patients"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/MTBLS1905/ms1_openms_features"))
    parser.add_argument("--roles", nargs="+", default=["patient"])
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N eligible existing files for a parameter pilot.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    manifest = manifest[manifest["sample_role"].isin(args.roles)].copy()
    manifest = manifest.sort_values(["sample_role", "sample_name"], kind="stable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    eligible = []
    for row in manifest.itertuples(index=False):
        path = args.mzml_dir / row.file_name
        if path.is_file() and path.stat().st_size > 1024:
            eligible.append((row, path))
        else:
            records.append({"sample_name": row.sample_name, "sample_role": row.sample_role, "source": str(path), "status": "pending_download"})
    if args.limit is not None:
        eligible = eligible[:args.limit]

    print(f"Eligible existing files: {len(eligible)}; pending download: {len(records)}", flush=True)
    for ordinal, (row, path) in enumerate(eligible, 1):
        output = feature_dir / f"{row.sample_name}__noise_{args.noise_threshold:g}.csv.gz"
        summary_path = feature_dir / f"{row.sample_name}__summary.json"
        base = {
            "sample_name": row.sample_name,
            "sample_role": row.sample_role,
            "tissue_type": getattr(row, "tissue_type", None),
            "source": str(path),
            "output": str(output),
        }
        if output.exists() and summary_path.exists() and not args.force:
            record = json.loads(summary_path.read_text(encoding="utf-8"))
            record["status"] = "reused"
            records.append(record)
            print(f"[{ordinal}/{len(eligible)}] Reused {row.sample_name}", flush=True)
            continue
        print(f"[{ordinal}/{len(eligible)}] Detecting {row.sample_name}", flush=True)
        started = time.time()
        try:
            experiment, load_summary = load_ms1(path)
            features, detection = detect_features(
                experiment,
                noise_threshold=args.noise_threshold,
                mass_error_ppm=args.mass_error_ppm,
                min_trace_length=args.min_trace_sec,
                max_trace_length=args.max_trace_sec,
            )
            features.insert(0, "sample_name", row.sample_name)
            features.insert(1, "sample_role", row.sample_role)
            features.to_csv(output, index=False)
            record = {**base, "status": "complete", "elapsed_seconds": time.time() - started, **load_summary, **detection}
            summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(json.dumps({"sample": row.sample_name, "features": len(features), "seconds": round(record["elapsed_seconds"], 1)}), flush=True)
        except Exception as error:
            record = {**base, "status": "failed", "elapsed_seconds": time.time() - started, "error": repr(error)}
            print(json.dumps(record), flush=True)
        records.append(record)
        pd.DataFrame(records).to_csv(args.output_dir / "extraction_manifest.partial.tsv", sep="\t", index=False)

    result = pd.DataFrame(records)
    result.to_csv(args.output_dir / "extraction_manifest.tsv", sep="\t", index=False)
    report = {
        "study": "MTBLS1905",
        "stage": "MS1 feature detection only",
        "parameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "counts": result.status.value_counts().to_dict() if len(result) else {},
        "warning": "Feature absence is not zero abundance. Do not conduct regional statistics until consensus grouping and gap filling are complete.",
    }
    (args.output_dir / "extraction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
