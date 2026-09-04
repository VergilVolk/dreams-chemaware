#!/usr/bin/env python
"""Validate MTBLS13729 consensus targets required by the MS1-MS2 bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"feature_id", "mz", "rt_sec", "keep_for_requantification"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/mtbls13729/ms1_features_full/features"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--minimum-samples", type=int, default=50)
    args = parser.parse_args()

    report = {"status": "mtbls13729_ms1_targets_validation_passed", "panels": {}}
    for panel in args.panels:
        feature_files = sorted(args.feature_dir.glob(f"{panel}__*__noise_10000.csv.gz"))
        if len(feature_files) < args.minimum_samples:
            raise RuntimeError(
                f"{panel}: only {len(feature_files)} extracted sample files; expected at least {args.minimum_samples}"
            )
        target_path = args.consensus_dir / f"{panel}__requantification_targets.csv.gz"
        metadata_path = args.consensus_dir / f"{panel}__consensus_metadata.csv.gz"
        samples_path = args.consensus_dir / f"{panel}__samples.csv"
        for path in (target_path, metadata_path, samples_path):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"{panel}: missing/empty consensus artifact: {path}")
        targets = pd.read_csv(target_path)
        missing = sorted(REQUIRED - set(targets.columns))
        if missing:
            raise RuntimeError(f"{panel}: targets missing columns {missing}")
        if targets.empty or targets.feature_id.duplicated().any():
            raise RuntimeError(f"{panel}: targets are empty or feature_id is duplicated")
        if not targets.keep_for_requantification.astype(bool).all():
            raise RuntimeError(f"{panel}: target table contains rows not marked for requantification")
        if (~np.isfinite(pd.to_numeric(targets.mz, errors="coerce"))).any() or (
            ~np.isfinite(pd.to_numeric(targets.rt_sec, errors="coerce"))
        ).any():
            raise RuntimeError(f"{panel}: non-finite m/z or RT in targets")
        samples = pd.read_csv(samples_path)
        if len(samples) != len(feature_files):
            raise RuntimeError(
                f"{panel}: sample table has {len(samples)} rows but {len(feature_files)} feature files"
            )
        report["panels"][panel] = {
            "extracted_samples": len(feature_files),
            "requantification_targets": int(len(targets)),
            "mz_min": float(targets.mz.min()),
            "mz_max": float(targets.mz.max()),
            "rt_min_sec": float(targets.rt_sec.min()),
            "rt_max_sec": float(targets.rt_sec.max()),
        }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
