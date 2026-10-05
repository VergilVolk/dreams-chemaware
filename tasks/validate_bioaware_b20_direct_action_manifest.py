#!/usr/bin/env python
"""Validate a completed B20 direct-action manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    actions = pd.read_csv(args.output_dir / "direct_actions.csv.gz")
    manifest = np.load(args.output_dir / "direct_action_manifest.npz", allow_pickle=False)
    if report.get("pass_to_direct_gradient_canary") is not True:
        raise RuntimeError("B20 direct-action gate false")
    if len(actions) != 860 or len(manifest["query_id"]) != 860:
        raise RuntimeError("B20 action coverage changed")
    if int(actions["corrected"].sum()) != 57 or int(actions["introduced"].sum()) != 7:
        raise RuntimeError("B20 B17 outcomes changed")
    if manifest["query_tensor"].shape != (860, 101, 2):
        raise RuntimeError("B20 query tensor shape changed")
    if manifest["reference_tensor"].shape[1:] != (101, 2):
        raise RuntimeError("B20 reference tensor shape changed")
    if not np.isfinite(manifest["query_tensor"]).all() or not np.isfinite(
        manifest["reference_tensor"]
    ).all():
        raise RuntimeError("B20 non-finite tensors")
    print("[validate_bioaware_b20_direct_action_manifest] PASS", report["B17_replay"])


if __name__ == "__main__":
    main()
