#!/usr/bin/env python3
"""Export frozen CALMS parent and coverage-matched WSE application controls.

This is an analysis adapter, not a new model.  It reads the same candidate
evidence used by frozen CALMS, replays the byte-identical scalar parent, and
restricts WSE to exactly the candidate rows on which CALMS can score.  It does
not read labels, responses, or biological context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CALMS_ROOT = REPO_ROOT / "deploy" / "calms_v1"
sys.path.insert(0, str(CALMS_ROOT))

from calms import CALMS, _load_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--calms-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.evidence.is_file() or not args.calms_scores.is_file():
        raise FileNotFoundError("evidence and CALMS score archives are required")

    axis, evidence, chemical = _load_evidence(args.evidence)
    parent, _ = CALMS(device=args.device).parent.score(axis, evidence, chemical)
    with np.load(args.calms_scores, allow_pickle=False) as body:
        required = {"candidate_pair_ids", "calms_score", "evidence_available"}
        missing = sorted(required - set(body.files))
        if missing:
            raise RuntimeError(f"CALMS score archive lacks arrays: {missing}")
        pair_ids = np.asarray(body["candidate_pair_ids"]).astype("U")
        calms = np.asarray(body["calms_score"], dtype=np.float64)
        calms_available = np.asarray(body["evidence_available"], dtype=bool)

    if not np.array_equal(pair_ids, axis.candidate_pair_ids):
        raise RuntimeError("CALMS score and evidence candidate axes differ")
    if not (parent.shape == calms.shape == calms_available.shape):
        raise RuntimeError("CALMS parent/final arrays have different axes")
    parent_available = np.isfinite(parent)
    if not np.array_equal(parent_available, calms_available):
        raise RuntimeError("final CALMS availability differs from its scalar parent")
    wse = np.asarray(evidence["weighted_spectral_entropy"], dtype=np.float64)
    wse_available = calms_available & np.isfinite(wse)

    output = pd.concat([
        pd.DataFrame({
            "candidate_pair_id": pair_ids,
            "method_name": "CALMS_parent",
            "method_score": parent,
            "evidence_available": parent_available,
        }),
        pd.DataFrame({
            "candidate_pair_id": pair_ids,
            "method_name": "WSE_CALMS_coverage",
            "method_score": wse,
            "evidence_available": wse_available,
        }),
    ], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        args.output,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(json.dumps({
        "status": "CALMS_APPLICATION_CONTROLS_COMPLETE",
        "output": str(args.output),
        "candidate_pairs": int(len(pair_ids)),
        "calms_parent_available_pairs": int(parent_available.sum()),
        "coverage_matched_wse_available_pairs": int(wse_available.sum()),
        "response_or_truth_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
