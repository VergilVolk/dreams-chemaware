#!/usr/bin/env python
"""Fail-closed validator for the frozen BioAware B32 action bank."""
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
    report_path = args.output / "report.json"
    table_path = args.output / "direct_actions.csv.gz"
    manifest_path = args.output / "direct_action_manifest.npz"
    for path in (report_path, table_path, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(table_path)
    with np.load(manifest_path, allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    if report.get("status") != "bioaware_b32_comprehensive_action_bank_complete":
        raise RuntimeError("B32 status changed")
    if report.get("pass_to_direct_gradient_canary") is not True:
        raise RuntimeError("B32 scientific gate failed")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("B32 individual gate failed")
    if len(frame) != 860 or len(body["query_id"]) != 860:
        raise RuntimeError("B32 coverage changed")
    corrective = body["direct_corrective"].astype(bool)
    safety = body["direct_safety"].astype(bool)
    uncertain = body["uncertain_sink_conflict"].astype(bool)
    if (int(corrective.sum()), int(safety.sum()), int(uncertain.sum())) != (63, 14, 3):
        raise RuntimeError("B32 role counts changed")
    if np.any(corrective & safety) or np.any(uncertain & (corrective | safety)):
        raise RuntimeError("B32 roles overlap")
    if body["reference_tensor"].shape[1:] != (101, 2):
        raise RuntimeError("B32 reference tensor shape changed")
    positions = body["final_reference_position"].astype(np.int64)
    if np.any(positions < 0) or np.any(positions >= len(body["reference_tensor"])):
        raise RuntimeError("B32 reference positions invalid")
    print(
        "[validate_bioaware_b32_comprehensive_action_bank] PASS "
        f"corrective={int(corrective.sum())} safety={int(safety.sum())} "
        f"uncertain={int(uncertain.sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
