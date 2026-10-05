#!/usr/bin/env python
"""Independent validator for B42 independent catalogue-topology audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b11_catalog_interaction_action import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.input_dir / "report.json"
    transitions_path = args.input_dir / "cross_catalog_transitions.csv.gz"
    selection_path = args.input_dir / "inner_selection_ledger.csv.gz"
    features_path = args.input_dir / "candidate_catalog_features.csv.gz"
    for path in (report_path, transitions_path, selection_path, features_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transitions = pd.read_csv(transitions_path, low_memory=False)
    if report.get("status") != "bioaware_b42_independent_catalog_topology_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B42 status")
    if "strict-KEGG" not in report.get("semantic_correction", ""):
        raise RuntimeError("B42 semantic correction is absent")
    if not report.get("archived_replay", {}).get("pass"):
        raise RuntimeError("B42 archived strict-KEGG replay failed")
    if report.get("pass_to_shared_embedding"):
        raise RuntimeError("B42 cannot authorize shared embedding")
    counts = transitions.groupby("arm")["query_id"].nunique().astype(int)
    if set(counts.index) != set(report.get("arm_reports", {})) or counts.ne(860).any():
        raise RuntimeError(f"B42 transition coverage drift: {counts.to_dict()}")
    contrasts = report.get("paired_contrasts", {})
    if len(contrasts) != 8:
        raise RuntimeError(f"B42 paired contrast count drift: {len(contrasts)}")
    for name, contrast in contrasts.items():
        left = transitions.loc[transitions["arm"].eq(contrast["left_arm"]), ["query_id", "final_correct"]]
        right = transitions.loc[transitions["arm"].eq(contrast["right_arm"]), ["query_id", "final_correct"]]
        paired = left.merge(right, on="query_id", validate="one_to_one", suffixes=("_left", "_right"))
        effect = paired["final_correct_left"].astype(int).to_numpy() - paired["final_correct_right"].astype(int).to_numpy()
        if (
            len(paired) != 860
            or not np.isclose(effect.mean(), contrast["mean_top1_difference"], atol=1e-15)
            or int((effect > 0).sum()) != contrast["left_better"]
            or int((effect < 0).sum()) != contrast["right_better"]
        ):
            raise RuntimeError(f"B42 paired contrast mismatch: {name}")
    incremental = report.get("multi_catalog_incremental", {})
    if bool(incremental.get("pass")) != bool(all(incremental.get("gates", {}).values())):
        raise RuntimeError("B42 incremental gate mismatch")
    for key, path in {
        "cross_catalog_transitions": transitions_path,
        "inner_selection_ledger": selection_path,
        "candidate_catalog_features": features_path,
    }.items():
        if report.get("provenance", {}).get(key) != sha256(path):
            raise RuntimeError(f"B42 provenance mismatch: {key}")
    print("[validate_bioaware_b42_independent_catalog_topology] PASS", {
        "rhea_portable": report["independent_rhea_portability"],
        "incremental": report["multi_catalog_incremental"]["pass"],
    }, flush=True)


if __name__ == "__main__":
    main()
