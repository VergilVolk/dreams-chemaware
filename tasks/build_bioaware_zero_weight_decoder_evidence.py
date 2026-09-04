#!/usr/bin/env python
"""Materialise the frozen zero-weight decoder family without fitting on validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.scores, args.artifact):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    weight = float(artifact["router"]["rank_consensus_expert"]["weights"]["family_decoder"])
    if weight != 0.0:
        raise RuntimeError("frozen decoder family is not zero-weight; neutral materialisation is invalid")
    frame = pd.read_csv(args.scores)
    required = ["query_id", "candidate_id", "truth_candidate_id", "truth_formula"]
    if not set(required).issubset(frame):
        raise RuntimeError("candidate score schema mismatch")
    output = frame[required].copy()
    output["decoder_score"] = 0.0
    output["heldout_rotations"] = 0
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "candidate_scores.csv.gz"
    output.to_csv(path, index=False, compression="gzip")
    report = {
        "status": "bioaware_zero_weight_decoder_evidence_complete",
        "formal": True,
        "candidate_rows": int(len(output)),
        "decoder_weight": weight,
        "fit_performed": False,
        "provenance": {
            "scores_sha256": sha256(args.scores), "artifact_sha256": sha256(args.artifact),
            "output_sha256": sha256(path),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
