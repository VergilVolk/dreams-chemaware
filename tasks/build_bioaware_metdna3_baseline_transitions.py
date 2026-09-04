#!/usr/bin/env python
"""Build identity-heldout baseline transitions from frozen DreaMS scores.

This contains no network evidence and performs no fitting.  It only expands
each query into the preregistered identity rotations in which its truth is
held out from the seed set.
"""
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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_top(group: pd.DataFrame) -> tuple[str, bool]:
    maximum = float(group.spectral_score.max())
    tied = group[np.isclose(group.spectral_score.astype(float), maximum, rtol=0, atol=1e-12)]
    return str(tied.sort_values("candidate_id").iloc[0].candidate_id), len(tied) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.scores, args.splits):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    scores = pd.read_csv(args.scores)
    splits = pd.read_csv(args.splits)
    required = {"query_id", "candidate_id", "spectral_score", "truth_candidate_id", "truth_formula"}
    if not required.issubset(scores):
        raise RuntimeError(f"score schema missing {sorted(required-set(scores))}")
    if not {"fold", "ik14", "role"}.issubset(splits):
        raise RuntimeError("identity split schema mismatch")
    decisions = []
    for query_id, group in scores.groupby("query_id", sort=False):
        truth = str(group.truth_candidate_id.iloc[0])
        top, unique = unique_top(group)
        formula = str(group.truth_formula.iloc[0])
        heldout = splits[(splits.ik14.astype(str) == truth) & splits.role.eq("heldout")]
        if heldout.empty:
            raise RuntimeError(f"truth identity has no heldout rotations: {truth}")
        for fold in sorted(heldout.fold.astype(int).unique()):
            correct = bool(unique and top == truth)
            decisions.append({
                "query_id": str(query_id), "truth_candidate_id": truth,
                "baseline_top_candidate": top, "final_top_candidate": top,
                "baseline_correct": correct, "final_correct": correct,
                "corrected": False, "introduced": False, "fold": int(fold),
                "truth_formula": formula, "delta": 0,
            })
    frame = pd.DataFrame(decisions)
    query_count = int(scores.query_id.nunique())
    if frame.query_id.nunique() != query_count or frame.duplicated(["fold", "query_id"]).any():
        raise RuntimeError("baseline rotation expansion is incomplete or duplicated")
    args.output_dir.mkdir(parents=True)
    transitions = args.output_dir / "raw_transitions.csv.gz"
    frame.to_csv(transitions, index=False, compression="gzip")
    report = {
        "status": "bioaware_metdna3_baseline_transitions_complete",
        "formal": True,
        "queries": query_count,
        "rotation_rows": int(len(frame)),
        "identities": int(scores.truth_candidate_id.nunique()),
        "formulas": int(scores.truth_formula.nunique()),
        "baseline_recall1": float(frame.groupby("query_id").baseline_correct.first().mean()),
        "contracts": {"network_evidence_used": False, "fit_performed": False, "P2b": "forbidden"},
        "provenance": {
            "scores_sha256": sha256(args.scores), "splits_sha256": sha256(args.splits),
            "transitions_sha256": sha256(transitions),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
