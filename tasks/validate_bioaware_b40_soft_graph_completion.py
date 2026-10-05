#!/usr/bin/env python
"""Independent validator for the outcome-blind B40-M0 cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = {"truth_candidate_id", "truth_formula", "is_positive", "spectral_score", "baseline_correct", "corrected", "introduced", "final_correct", "delta"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.input_dir / "report.json"
    score_path = args.input_dir / "candidate_context_diffusion.csv.gz"
    for path in (report_path, score_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scores = pd.read_csv(score_path, low_memory=False)
    if report.get("status") != "bioaware_b40_soft_graph_completion_cache_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B40-M0 report status")
    if report.get("outcomes_read") or report.get("models_fitted") or report.get("contract", {}).get("truth_or_rank_used"):
        raise RuntimeError("B40-M0 crossed its outcome-blind boundary")
    leaked = FORBIDDEN & set(scores.columns)
    if leaked:
        raise RuntimeError(f"B40-M0 leaked outcomes: {sorted(leaked)}")
    if scores.duplicated(["query_id", "seed_stratum", "candidate_id"]).any():
        raise RuntimeError("duplicate candidate-context diffusion row")
    if len(scores) != report.get("candidate_context_rows"):
        raise RuntimeError("B40 score row count mismatch")
    if report.get("provenance", {}).get("scores") != sha256(score_path):
        raise RuntimeError("B40 score provenance mismatch")
    if (scores[["real_diffusion_score", "null_diffusion_mean", "null_diffusion_max"]].to_numpy(float) < -1e-12).any():
        raise RuntimeError("negative graph diffusion probability")
    print(
        "[validate_bioaware_b40_soft_graph_completion] PASS",
        {"contexts": report["contexts"], "score_rows": len(scores), "pass": report["pass_to_fixed_action_evaluation"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
