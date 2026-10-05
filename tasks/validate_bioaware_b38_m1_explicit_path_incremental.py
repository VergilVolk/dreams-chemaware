#!/usr/bin/env python
"""Independent structural validator for BioAware B38-M1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "report": args.input_dir / "report.json",
        "transitions": args.input_dir / "explicit_path_arm_transitions.csv.gz",
        "selection": args.input_dir / "inner_gate_selection_ledger.csv.gz",
        "features": args.input_dir / "candidate_explicit_path_features.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b38_m1_explicit_path_incremental_complete":
        raise RuntimeError("unexpected B38-M1 status")
    if not report.get("b37_topology_replay", {}).get("pass"):
        raise RuntimeError("B37 topology replay did not pass")
    transitions = pd.read_csv(paths["transitions"])
    features = pd.read_csv(paths["features"])
    expected_arms = 23
    counts = transitions.groupby("arm")["query_id"].nunique()
    if len(counts) != expected_arms or not counts.eq(860).all():
        raise RuntimeError(f"B38-M1 arm coverage changed: {counts.to_dict()}")
    if len(features) != 6695 or features.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B38-M1 candidate feature denominator changed")
    null = report.get("primary_degree_rewire_null", {})
    if null.get("repeats") != 20 or len(null.get("effects", [])) != 20:
        raise RuntimeError("B38-M1 does not report all 20 degree-preserving nulls")
    expected_pass = bool(all(report.get("gates", {}).values()))
    if bool(report.get("pass_to_b38_m2_path_set_model")) != expected_pass:
        raise RuntimeError("B38-M2 decision does not equal frozen gate conjunction")
    provenance = report.get("provenance", {})
    for key in ("transitions", "selection", "features"):
        report_key = {
            "transitions": "explicit_path_arm_transitions",
            "selection": "inner_gate_selection_ledger",
            "features": "candidate_explicit_path_features",
        }[key]
        if provenance.get(report_key) != sha256(paths[key]):
            raise RuntimeError(f"provenance mismatch: {report_key}")
    print(
        "[validate_bioaware_b38_m1] PASS",
        {
            "arms": len(counts),
            "queries_per_arm": int(counts.iloc[0]),
            "incremental_delta": report["primary_real_vs_topology"]["mean_top1_difference"],
            "rewire_empirical_p": null["empirical_upper_p"],
            "pass_to_m2": bool(report["pass_to_b38_m2_path_set_model"]),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
