"""Fail-closed validation for the A4-B0 positive-evidence diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_a4b_positive_evidence")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    decision_path = args.output_dir / "decision.json"
    result_path = args.output_dir / "paired_results.csv.gz"
    for path in (decision_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "noise_v3_a4b_positive_evidence_complete":
        raise RuntimeError("unexpected B0 status")
    if not args.allow_smoke and not decision.get("formal"):
        raise RuntimeError("formal validation refuses a smoke result")
    frame = pd.read_csv(result_path)
    required = {
        "query_index", "query_ik14", "query_formula", "scan_kind", "alpha",
        "baseline_rank", "target_rank", "mean_random_accuracy", "corrected",
        "introduced", "new_correction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"B0 result is missing columns: {sorted(missing)}")
    expected_rows = (
        int(decision["integrity"]["aligned_queries"])
        * len(decision["parameters"]["alphas"])
    )
    if len(frame) != expected_rows:
        raise RuntimeError(f"B0 row count mismatch: {len(frame)} != {expected_rows}")
    if frame.duplicated(["query_index", "alpha"]).any():
        raise RuntimeError("duplicate B0 query/alpha rows")
    if frame[["query_index", "alpha", "baseline_rank", "target_rank"]].isna().any().any():
        raise RuntimeError("B0 contains missing core values")
    if (frame[["baseline_rank", "target_rank"]] < 1).any().any():
        raise RuntimeError("B0 contains invalid ranks")
    if not frame["mean_random_accuracy"].between(0, 1).all():
        raise RuntimeError("B0 random accuracy is outside [0,1]")
    if not args.allow_smoke:
        integrity = decision["integrity"]
        if integrity["source_queries"] != 4998:
            raise RuntimeError("formal B0 did not consume all A4 queries")
        if integrity["baseline_similarity_mismatch"] != 0:
            raise RuntimeError("formal B0 baseline integrity failed")
        if integrity["aligned_queries"] < 1000:
            raise RuntimeError("formal B0 has insufficient aligned queries")
    print("[validate_noise_v3_a4b_positive_evidence] PASS", flush=True)


if __name__ == "__main__":
    main()
