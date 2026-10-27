#!/usr/bin/env python3
"""Verify that the portable Full23 release exactly replays application scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CANDIDATES = 22_046
MAXIMUM_ABSOLUTE_DRIFT = 2e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-tsv", type=Path, required=True)
    parser.add_argument("--release-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = pd.read_csv(
        args.scorer_tsv,
        sep="\t",
        usecols=["candidate_pair_id", "unified_candidate_ranker_score"],
        dtype={"candidate_pair_id": "string"},
    )
    with np.load(args.release_scores, allow_pickle=False) as released:
        pair_ids = released["candidate_pair_ids"].astype("U")
        scores = released["unified_candidate_ranker_score"].astype(np.float64)
    expected_ids = source["candidate_pair_id"].astype(str).to_numpy(dtype="U")
    expected_scores = pd.to_numeric(
        source["unified_candidate_ranker_score"], errors="coerce",
    ).to_numpy(np.float64)
    if len(scores) != EXPECTED_CANDIDATES or not np.array_equal(pair_ids, expected_ids):
        raise RuntimeError("Full23 portable replay candidate axis drifted")
    if not np.array_equal(np.isnan(scores), np.isnan(expected_scores)):
        raise RuntimeError("Full23 portable replay availability drifted")
    finite = np.isfinite(expected_scores)
    maximum = float(np.max(np.abs(scores[finite] - expected_scores[finite])))
    if maximum > MAXIMUM_ABSOLUTE_DRIFT:
        raise RuntimeError(f"Full23 portable replay score drifted: {maximum}")

    report = {
        "status": "FULL23_APPLICATION_PORTABLE_REPLAY_PASS",
        "candidates": int(len(scores)),
        "finite_candidates": int(np.sum(finite)),
        "maximum_absolute_score_drift": maximum,
        "tolerance": MAXIMUM_ABSOLUTE_DRIFT,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
