#!/usr/bin/env python
"""Fail-closed validation for deployable BioAware automatic seed artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    report_path = args.input_dir / "report.json"
    seed_path = args.input_dir / "seeds_auto.csv"
    if not report_path.exists() or not seed_path.exists():
        raise FileNotFoundError("BioAware seed contract requires report.json and seeds_auto.csv")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parameters = report.get("auto_seed_parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("seed contract has no frozen auto_seed_parameters")
    required_parameters = {"minimum_score", "minimum_margin", "ppm_tolerance"}
    if required_parameters - set(parameters):
        raise RuntimeError("seed contract parameters are incomplete")
    artifacts = report.get("artifacts", {})
    recorded_hash = artifacts.get("auto_seeds_sha256")
    if not recorded_hash or recorded_hash != sha256(seed_path):
        raise RuntimeError("seeds_auto.csv hash does not match report")
    seeds = pd.read_csv(seed_path)
    required = {"seed_query_id", "seed_compound_id", "seed_score", "reference_kind"}
    missing = required - set(seeds)
    if missing:
        raise RuntimeError(f"automatic seeds missing columns: {sorted(missing)}")
    scores = pd.to_numeric(seeds["seed_score"], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all():
        raise RuntimeError("automatic seed scores are non-finite")
    minimum = float(parameters["minimum_score"])
    if len(scores) and np.any(scores + 1e-12 < minimum):
        raise RuntimeError("automatic seed artifact violates frozen minimum_score")
    if seeds.duplicated(["seed_query_id", "seed_compound_id"]).any():
        raise RuntimeError("automatic seed artifact contains duplicate query/identity rows")
    if len(seeds) != int(report.get("auto_seed_rows", -1)):
        raise RuntimeError("automatic seed row count does not match report")
    if seeds["seed_compound_id"].astype(str).nunique() != int(
        report.get("auto_seed_compounds", -1)
    ):
        raise RuntimeError("automatic seed identity count does not match report")
    print(
        json.dumps(
            {
                "status": "bioaware_seed_contract_validation_passed",
                "rows": int(len(seeds)),
                "identities": int(seeds["seed_compound_id"].astype(str).nunique()),
                "minimum_score": minimum,
                "seed_sha256": recorded_hash,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
