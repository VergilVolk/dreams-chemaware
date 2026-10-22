#!/usr/bin/env python3
"""Resume-safe disjoint feature-cache prefill for MTBLS1866 measurement builds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_mtbls1066_ketogenic_response import extract_or_reuse_features
from build_mtbls1866_phenotype_blind_measurement import (
    EXPECTED_BYTES,
    EXPECTED_FILES,
    EXTENSION_BYTES,
    EXTENSION_FILES,
    read_staged_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-manifest", required=True, type=Path)
    parser.add_argument("--extension-staged-manifest", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--noise-threshold", type=float, default=10_000.0)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be inside [0, shard count)")
    primary = read_staged_manifest(
        args.staged_manifest, expected_files=EXPECTED_FILES, expected_bytes=EXPECTED_BYTES,
    )
    frames = [primary]
    if args.extension_staged_manifest is not None:
        frames.append(read_staged_manifest(
            args.extension_staged_manifest,
            expected_files=EXTENSION_FILES,
            expected_bytes=EXTENSION_BYTES,
        ))
    manifest = pd.concat(frames, ignore_index=True).sort_values(
        ["assay_name", "local_path"], kind="mergesort",
    ).reset_index(drop=True)
    stems = manifest["local_path"].map(lambda value: Path(value).stem)
    if stems.duplicated().any():
        raise RuntimeError("cache filename stems overlap between staged assays")
    positions = [
        index for index in range(len(manifest)) if index % args.shard_count == args.shard_index
    ]
    shard = manifest.iloc[positions].reset_index(drop=True)
    _, audit = extract_or_reuse_features(
        shard,
        args.work_dir,
        noise_threshold=args.noise_threshold,
        mass_error_ppm=args.mass_error_ppm,
        min_trace_sec=args.min_trace_sec,
        max_trace_sec=args.max_trace_sec,
    )
    report = {
        "schema": "mtbls1866_feature_cache_prefill_shard_v1",
        "status": "PHENOTYPE_BLIND_FEATURE_CACHE_SHARD_COMPLETE",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "total_staged_files": len(manifest),
        "shard_files": len(shard),
        "complete": int(sum(row["status"] == "complete" for row in audit)),
        "reused": int(sum(row["status"] == "reused" for row in audit)),
        "leakage_status": "phenotype-blind staged manifests only; no design ledger or outcome read",
    }
    report_path = args.work_dir / f"feature_prefill_{args.shard_count}_{args.shard_index}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
