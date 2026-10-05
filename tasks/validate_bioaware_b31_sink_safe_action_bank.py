#!/usr/bin/env python
"""Fail-closed validation for BioAware B31."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = (
        args.output / "report.json", args.output / "direct_actions.csv.gz",
        args.output / "direct_action_manifest.npz",
    )
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    frame = pd.read_csv(paths[1])
    with np.load(paths[2], allow_pickle=False) as handle:
        corrective = handle["direct_corrective"].astype(bool)
        safety = handle["direct_safety"].astype(bool)
        uncertain = handle["uncertain_sink_conflict"].astype(bool)
    if report.get("status") != "bioaware_b31_sink_safe_action_bank_complete":
        raise RuntimeError("B31 status changed")
    if report.get("pass_to_direct_gradient_canary") is not True or not all(report.get("gates", {}).values()):
        raise RuntimeError("B31 gate failed")
    if len(frame) != 860 or (int(corrective.sum()), int(safety.sum()), int(uncertain.sum())) != (54, 7, 3):
        raise RuntimeError("B31 action coverage changed")
    if np.any(corrective & safety) or np.any(uncertain & (corrective | safety)):
        raise RuntimeError("B31 action roles overlap")
    print("[validate_bioaware_b31_sink_safe_action_bank] PASS", flush=True)


if __name__ == "__main__":
    main()
