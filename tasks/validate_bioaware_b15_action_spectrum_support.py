#!/usr/bin/env python
"""Validate a completed B15 spectrum-support artifact."""
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
    actions = pd.read_csv(args.output_dir / "spectrum_supported_actions.csv.gz")
    manifest = np.load(args.output_dir / "spectrum_support_manifest.npz", allow_pickle=False)
    if report.get("pass_to_direct_shared_embedding_design") is not True:
        raise RuntimeError("B15 scientific gate is false")
    if len(actions) != 860 or len(manifest["query_id"]) != 860:
        raise RuntimeError("B15 row coverage changed")
    if int(actions["corrected"].sum()) != 58 or int(actions["introduced"].sum()) != 8:
        raise RuntimeError("B12 outcomes changed inside B15")
    if manifest["query_tensor"].shape != (860, 101, 2):
        raise RuntimeError("query tensor shape changed")
    if manifest["reference_tensor"].shape[1:] != (101, 2):
        raise RuntimeError("reference tensor shape changed")
    if not np.isfinite(manifest["query_tensor"]).all():
        raise RuntimeError("query tensors contain non-finite values")
    print(
        "[validate_bioaware_b15_action_spectrum_support] PASS",
        report["B12_replay"],
    )


if __name__ == "__main__":
    main()
