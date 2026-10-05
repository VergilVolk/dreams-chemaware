#!/usr/bin/env python
"""Independent structural validator for BioAware B38-M0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = {
    "truth_candidate_id", "truth_formula", "is_positive", "baseline_correct",
    "corrected", "introduced", "final_correct", "delta",
}


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
        "events": args.input_dir / "explicit_direct_path_events.csv.gz",
        "candidates": args.input_dir / "candidate_path_coverage.csv.gz",
        "queries": args.input_dir / "query_path_coverage.csv.gz",
        "query_contexts": args.input_dir / "query_context_path_coverage.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    events = pd.read_csv(paths["events"])
    candidates = pd.read_csv(paths["candidates"])
    queries = pd.read_csv(paths["queries"])
    query_contexts = pd.read_csv(paths["query_contexts"])
    if report.get("status") != "bioaware_b38_m0_explicit_path_coverage_complete":
        raise RuntimeError("unexpected B38-M0 status")
    if report.get("model_fitted") or report.get("embedding_values_read"):
        raise RuntimeError("B38-M0 read a forbidden outcome/model channel")
    if FORBIDDEN & set(events.columns):
        raise RuntimeError(f"event outcome leakage: {sorted(FORBIDDEN & set(events.columns))}")
    if len(queries) != 860 or queries["query_id"].nunique() != 860:
        raise RuntimeError("query denominator is not the frozen 860")
    if candidates.duplicated(["query_id", "candidate_id", "seed_stratum"]).any():
        raise RuntimeError("duplicate candidate-context coverage rows")
    if candidates[["query_id", "candidate_id"]].drop_duplicates().shape[0] != 6695:
        raise RuntimeError("candidate contexts do not replay the complete frozen 6,695 rows")
    evaluated_candidates = candidates.loc[candidates["polarity"].eq("negative")]
    if evaluated_candidates[["query_id", "candidate_id"]].drop_duplicates().shape[0] != 3314:
        raise RuntimeError("negative candidate contexts do not replay the frozen 3,314 rows")
    if set(evaluated_candidates["query_id"].astype(str)) != set(queries["query_id"].astype(str)):
        raise RuntimeError("candidate/query coverage mismatch")
    if query_contexts.duplicated(["query_id", "seed_stratum"]).any():
        raise RuntimeError("duplicate query-context rows")
    if len(query_contexts) != 4148 or len(evaluated_candidates) != 15332 or len(candidates) != 38999:
        raise RuntimeError(
            f"rotation denominator changed: contexts={len(query_contexts)} "
            f"negative_candidate_contexts={len(evaluated_candidates)} "
            f"training_candidate_contexts={len(candidates)}"
        )
    if set(query_contexts["query_id"].astype(str)) != set(queries["query_id"].astype(str)):
        raise RuntimeError("query-context/query coverage mismatch")
    if len(events) == 0 or events.duplicated(
        ["query_id", "candidate_id", "seed_stratum", "seed_identity", "edge_source", "edge_key", "reaction_id"]
    ).any():
        raise RuntimeError("empty or duplicate explicit event ledger")
    provenance = report.get("provenance", {})
    expected = {
        "explicit_direct_path_events": sha256(paths["events"]),
        "candidate_path_coverage": sha256(paths["candidates"]),
        "query_path_coverage": sha256(paths["queries"]),
        "query_context_path_coverage": sha256(paths["query_contexts"]),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"provenance mismatch: {key}")
    gates = report.get("gates", {})
    primary = gates.get("primary_explicit_path", {})
    joint = gates.get("joint_data_layer_specificity", {})
    expected_primary = bool(primary and all(primary.values()))
    expected_joint = bool(expected_primary and joint and all(joint.values()))
    if bool(report.get("pass_to_b38_m1")) != expected_primary:
        raise RuntimeError("B38-M1 primary decision does not equal its gate conjunction")
    if bool(report.get("pass_to_b38_m1_joint_data_layer")) != expected_joint:
        raise RuntimeError("B38-M1 joint-data decision does not equal its gate conjunction")
    print(
        "[validate_bioaware_b38_m0] PASS",
        {
            "queries": len(queries),
            "candidate_rows": len(candidates),
            "query_contexts": len(query_contexts),
            "events": len(events),
            "pass_to_b38_m1": bool(report["pass_to_b38_m1"]),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
