#!/usr/bin/env python
"""Independent structural validator for the BioAware B37 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_ARMS = {
    "spectral_only", "catalog_network_member", "catalog_degree",
    "catalog_mass_coverage", "catalog_reference_density", "catalog_topology",
    "catalog_observability", "catalog_full", "catalog_without_network_member",
    "catalog_without_degree", "catalog_without_mass_coverage",
    "catalog_without_reference_density",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "report": args.input_dir / "report.json",
        "transitions": args.input_dir / "catalog_ablation_transitions.csv.gz",
        "selection": args.input_dir / "catalog_inner_selection_ledger.csv.gz",
        "features": args.input_dir / "reaction_feature_per_query.csv.gz",
        "queries": args.input_dir / "reaction_observability_per_query.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    transitions = pd.read_csv(paths["transitions"])
    features = pd.read_csv(paths["features"])
    queries = pd.read_csv(paths["queries"])
    if set(transitions["arm"].astype(str)) != EXPECTED_ARMS:
        raise RuntimeError("B37 arm set changed")
    counts = transitions.groupby("arm")["query_id"].agg(["size", "nunique"])
    if not counts.eq(860).all().all():
        raise RuntimeError(f"B37 arm coverage invalid: {counts.to_dict()}")
    if queries["query_id"].nunique() != 860 or len(queries) != 860:
        raise RuntimeError("B37 reaction query table does not contain 860 unique queries")
    if len(features) != 8600 or features.groupby("query_id")["feature"].nunique().ne(10).any():
        raise RuntimeError("B37 reaction feature table is not 860 queries x 10 features")
    full = transitions.loc[transitions["arm"].eq("catalog_full")]
    corrected = int(full["corrected"].astype(bool).sum())
    introduced = int(full["introduced"].astype(bool).sum())
    baseline = float(full["baseline_correct"].astype(bool).mean())
    final = float(full["final_correct"].astype(bool).mean())
    if abs((final - baseline) - (corrected - introduced) / 860) > 1e-12:
        raise RuntimeError("B37 catalogue arithmetic mismatch")
    if not report["b36_catalog_querywise_replay"]["pass"]:
        raise RuntimeError("B37 did not exactly replay B36 catalogue decisions")
    provenance = report["provenance"]
    for name, key in (
        ("transitions", "catalog_ablation_transitions_sha256"),
        ("selection", "catalog_inner_selection_ledger_sha256"),
        ("features", "reaction_feature_per_query_sha256"),
        ("queries", "reaction_observability_per_query_sha256"),
    ):
        if sha256(paths[name]) != provenance[key]:
            raise RuntimeError(f"B37 {name} SHA256 mismatch")
    print(json.dumps({
        "status": "bioaware_b37_validation_passed",
        "queries_per_arm": 860,
        "arms": len(EXPECTED_ARMS),
        "catalog_corrected": corrected,
        "catalog_introduced": introduced,
        "mechanism": report["catalog_ablation"]["mechanism_classification"],
        "decision": report["decision"],
        "catalog_mechanism_audit_pass": bool(report["catalog_mechanism_audit_pass"]),
        "pass_to_raw_path_diagnostic": bool(report["pass_to_raw_path_diagnostic"]),
        "pass_to_shared_embedding": bool(report["pass_to_shared_embedding"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
