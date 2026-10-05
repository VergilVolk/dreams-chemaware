#!/usr/bin/env python
"""Open B47 ranking truth exactly once after a successful repaired gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--unary-scores", type=Path, required=True)
    parser.add_argument("--exact-event-scores", type=Path, required=True)
    parser.add_argument("--control", action="append", default=[], metavar="METHOD=CSV")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=51976)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    report = json.loads((args.repair_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("authorization_after_repair") is not True:
        raise RuntimeError("B47 truth opening remains forbidden: repaired gate did not authorize")
    frozen = json.loads(
        args.exact_event_scores.with_suffix(args.exact_event_scores.suffix + ".json").read_text(
            encoding="utf-8"
        )
    )
    if frozen.get("status") != "BIOAWARE_B47_REPAIRED_ACTION_FROZEN":
        raise RuntimeError("exact-event scores are not a frozen truth-blind action artifact")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.seal.parent.mkdir(parents=True, exist_ok=True)
    seal_payload = {
        "status": "BIOAWARE_B47_TRUTH_OPENING_CONSUMED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repair_dir": str(args.repair_dir),
        "candidate_manifest": str(args.candidate_manifest),
        "exact_event_scores": str(args.exact_event_scores),
        "output": str(args.output),
    }
    try:
        descriptor = os.open(args.seal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise RuntimeError(f"B47 ranking truth was already opened: {args.seal}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(seal_payload, handle, indent=2)

    command = [
        sys.executable, "-u", str(ROOT / "tasks/evaluate_bioaware_unified_benchmark.py"),
        "--candidate-manifest", str(args.candidate_manifest),
        "--method", f"frozen_unary={args.unary_scores}",
        "--method", f"bioaware_exact_event={args.exact_event_scores}",
        "--baseline-method", "frozen_unary",
        "--benchmark-track", "sample_context",
        "--protocol-scope", "sealed_external",
        "--output-dir", str(args.output),
        "--expected-queries", str(args.expected_queries),
        "--bootstrap-resamples", "10000",
        "--seed", "20261004",
    ]
    for control in args.control:
        command.extend(("--method", control))
    subprocess.run(command, check=True)
    print(json.dumps(seal_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
