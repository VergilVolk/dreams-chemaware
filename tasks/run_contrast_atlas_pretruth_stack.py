#!/usr/bin/env python
"""Execute the complete phenotype-blind Contrast Atlas evaluation prefix.

The command stops before any relevance truth is opened.  It consumes already
frozen evidence-control worlds, preserves one candidate universe, runs paired
retrieval, and calibrates shared-pattern alignment by an inner permutation
null.  Downstream truth opening is intentionally a separate command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from calibrate_contrast_atlas_pattern_null import calibrate_retrievals
from run_contrast_atlas_control_retrieval import run_control_retrievals


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--universe-contrast-records", type=Path, required=True)
    parser.add_argument("--query-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-same-source", action="store_true")
    parser.add_argument("--min-entities", type=int, default=3)
    parser.add_argument("--min-patterns", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-match-probability", type=float, default=0.8)
    parser.add_argument("--association-threshold", type=float, default=0.3)
    parser.add_argument("--null-draws", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--fdr-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = {
        "min_entities": args.min_entities,
        "min_patterns": args.min_patterns,
        "min_coverage": args.min_coverage,
        "min_match_probability": args.min_match_probability,
        "association_threshold": args.association_threshold,
        "allow_same_source": args.allow_same_source,
    }
    registry = pd.read_csv(args.query_registry)
    universe = pd.read_csv(args.universe_contrast_records)
    retrieval, patterns, counts = run_control_retrievals(
        args.control_root, universe, registry, **config,
    )
    calibrated = calibrate_retrievals(
        retrieval, patterns,
        draws=args.null_draws,
        base_seed=args.seed,
        min_common=args.min_patterns,
        fdr_threshold=args.fdr_threshold,
    )
    args.output_dir.mkdir(parents=True)
    retrieval.to_csv(args.output_dir / "retrieval_results.csv.gz", index=False, compression="gzip")
    patterns.to_csv(args.output_dir / "pattern_responses.csv.gz", index=False, compression="gzip")
    calibrated.to_csv(args.output_dir / "calibrated_candidates.csv.gz", index=False, compression="gzip")
    report = {
        "status": "CONTRAST_ATLAS_PRETRUTH_STACK_COMPLETE",
        "formal": False,
        "truth_opened": False,
        "candidate_universe_invariant": True,
        "queries": int(registry["query_id"].nunique()),
        "candidate_rows": int(len(calibrated)),
        "calibrated_rows": int(calibrated["calibration_status"].eq(
            "calibrated_within_target_pattern_permutation"
        ).sum()),
        "world_counts": counts,
        "retrieval_configuration": config,
        "calibration_configuration": {
            "null_draws": args.null_draws,
            "seed": args.seed,
            "min_common_patterns": args.min_patterns,
            "development_fdr_threshold": args.fdr_threshold,
        },
        "input_sha256": {
            "universe_contrast_records": sha256(args.universe_contrast_records),
            "query_registry": sha256(args.query_registry),
            "control_world_report": sha256(args.control_root / "report.json"),
        },
        "next_gate": "Freeze outputs and thresholds before a separate truth-opening evaluation.",
        "claim_limit": "Phenotype-blind pretruth execution only; no biological correctness or unknown increment is established.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
