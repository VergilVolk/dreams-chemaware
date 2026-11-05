#!/usr/bin/env python3
"""Export frozen CALMS NPZ predictions to the application long-table axis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-name", default="CALMS")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.scores, allow_pickle=False) as body:
        required = {"candidate_pair_ids", "calms_score", "evidence_available"}
        missing = sorted(required - set(body.files))
        if missing:
            raise RuntimeError(f"CALMS score archive lacks arrays: {missing}")
        pair_ids = np.asarray(body["candidate_pair_ids"]).astype("U")
        scores = np.asarray(body["calms_score"], dtype=np.float64)
        available = np.asarray(body["evidence_available"], dtype=bool)
    if not (pair_ids.shape == scores.shape == available.shape):
        raise RuntimeError("CALMS output arrays have different axes")
    if len(np.unique(pair_ids)) != len(pair_ids):
        raise RuntimeError("CALMS output has duplicate candidate_pair_id")
    if not np.array_equal(available, np.isfinite(scores)):
        raise RuntimeError("CALMS availability must equal finite-score availability")
    output = pd.DataFrame({
        "candidate_pair_id": pair_ids,
        "method_name": args.method_name,
        "method_score": scores,
        "evidence_available": available,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        args.output, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(json.dumps({
        "status": "CALMS_APPLICATION_SCORE_EXPORT_COMPLETE",
        "output": str(args.output),
        "candidate_pairs": int(len(output)),
        "available_pairs": int(available.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
