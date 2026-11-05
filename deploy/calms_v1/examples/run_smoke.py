#!/usr/bin/env python3
"""Create retained synthetic CALMS inputs and run the real frozen CPU CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from calms import ENCODER_NAMES  # noqa: E402
from test_calms_frozen_release import write_fixture  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("calms_smoke"),
        help="New directory retaining inputs/ and predictions/ (must not exist).",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"smoke output root already exists; choose a new path: {output_root}"
        )

    input_dir = output_root / "inputs"
    prediction_dir = output_root / "predictions"
    input_dir.mkdir(parents=True, exist_ok=False)
    fixture = write_fixture(input_dir)

    command = [
        sys.executable,
        str(PACKAGE_ROOT / "calms.py"),
        "--evidence",
        str(fixture["evidence_path"]),
        "--evidence-manifest",
        str(fixture["evidence_manifest_path"]),
        "--representation",
        str(fixture["representation_path"]),
        "--representation-manifest",
        str(fixture["representation_manifest_path"]),
    ]
    for name in ENCODER_NAMES:
        command.extend(
            ["--embedding-bank", f"{name}={fixture['bank_paths'][name]}"]
        )
    command.extend(["--output", str(prediction_dir), "--device", "cpu"])

    completed = subprocess.run(
        command,
        cwd=output_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if "CALMS_INFERENCE_COMPLETE" not in completed.stdout:
        raise RuntimeError("CALMS CLI did not report successful completion")

    with np.load(prediction_dir / "candidate_scores.npz", allow_pickle=False) as body:
        candidates = int(len(body["calms_score"]))
    with np.load(prediction_dir / "query_registry.npz", allow_pickle=False) as body:
        queries = int(len(body["query_status"]))
        statuses = body["query_status"].astype("U").tolist()

    print(
        json.dumps(
            {
                "status": "CALMS_RETAINED_SYNTHETIC_SMOKE_PASS",
                "claim_boundary": "synthetic release contract only",
                "output_root": str(output_root),
                "inputs": str(input_dir),
                "predictions": str(prediction_dir),
                "queries": queries,
                "candidates": candidates,
                "query_status": statuses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
