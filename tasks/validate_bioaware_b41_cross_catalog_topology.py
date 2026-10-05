#!/usr/bin/env python
"""Independent validator for B41 cross-catalog topology audit."""
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
    if report.get("status") != "bioaware_b41_cross_catalog_topology_complete" or not report.get("formal"):
        raise RuntimeError("unexpected B41 status")
    if not report.get("rhea_replay", {}).get("pass"):
        raise RuntimeError("B41 exact Rhea replay failed")
    if report.get("pass_to_shared_embedding"):
        raise RuntimeError("catalogue audit cannot authorize shared embedding")
    arms = set(report.get("arm_reports", {}))
    counts = transitions.groupby("arm")["query_id"].nunique().astype(int)
    if set(counts.index) != arms or counts.ne(860).any():
        raise RuntimeError(f"B41 transition coverage drift: {counts.to_dict()}")
    contrasts = report.get("paired_contrasts", {})
    required_contrasts = {
        "all_catalogues_vs_rhea",
        "rhea_kegg_consensus_vs_rhea",
        "kegg_vs_rhea",
        "emrn_vs_rhea",
        "consensus_vs_null_r00",
        "consensus_vs_null_r01",
        "consensus_vs_null_r02",
    }
    if set(contrasts) != required_contrasts:
        raise RuntimeError(f"B41 paired contrast set drift: {sorted(contrasts)}")
    for name, contrast in contrasts.items():
        left = transitions.loc[
            transitions["arm"].eq(contrast["left_arm"]),
            ["query_id", "final_correct"],
        ]
        right = transitions.loc[
            transitions["arm"].eq(contrast["right_arm"]),
            ["query_id", "final_correct"],
        ]
        paired = left.merge(
            right, on="query_id", validate="one_to_one", suffixes=("_left", "_right")
        )
        effect = (
            paired["final_correct_left"].astype(int)
            - paired["final_correct_right"].astype(int)
        ).to_numpy(float)
        checks = {
            "queries": len(paired) == 860,
            "mean": np.isclose(effect.mean(), contrast["mean_top1_difference"], atol=1e-15),
            "left_better": int((effect > 0).sum()) == contrast["left_better"],
            "right_better": int((effect < 0).sum()) == contrast["right_better"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"B41 paired contrast mismatch {name}: {checks}")
    incremental = report.get("multi_catalog_incremental", {})
    gates = incremental.get("gates", {})
    if not gates or bool(incremental.get("pass")) != bool(all(gates.values())):
        raise RuntimeError("B41 multi-catalog incremental gate mismatch")
    provenance = report.get("provenance", {})
    expected = {
        "cross_catalog_transitions": transitions_path,
        "inner_selection_ledger": selection_path,
        "candidate_catalog_features": features_path,
    }
    for key, path in expected.items():
        if provenance.get(key) != sha256(path):
            raise RuntimeError(f"B41 provenance mismatch: {key}")
    print(
        "[validate_bioaware_b41_cross_catalog_topology] PASS",
        {"portable": report["portable_catalog_prior"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
