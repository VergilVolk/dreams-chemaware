#!/usr/bin/env python3
"""Freeze metadata-matched Pan-ReDU pairs as candidates, never as truth.

This ledger is deliberately created before transport scores exist.  Equal ReDU
comparison strings nominate pairs for later paper/design review; they do not
make those pairs biological positives.  All emitted rows therefore remain
``unknown`` and are ineligible for claim-bearing evaluation.
"""
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


def build_registry(topology: pd.DataFrame, panel_pairs: pd.DataFrame) -> pd.DataFrame:
    required_topology = {
        "panel_id", "contrast_id", "preliminary_source_family",
        "comparison_stratum", "topology_status",
    }
    required_pairs = {"left_panel_id", "right_panel_id", "left_source_family", "right_source_family"}
    if missing := required_topology - set(topology):
        raise RuntimeError(f"topology lacks columns: {sorted(missing)}")
    if missing := required_pairs - set(panel_pairs):
        raise RuntimeError(f"panel pairs lack columns: {sorted(missing)}")
    topology = topology.loc[topology.topology_status.astype(str).str.startswith("PASS")].copy()
    if topology.duplicated(["panel_id", "contrast_id"]).any():
        raise RuntimeError("duplicate PASS panel/contrast keys")
    if panel_pairs.left_source_family.eq(panel_pairs.right_source_family).any():
        raise RuntimeError("pair manifest contains a source-family-overlap pair")
    by_panel = {panel: block.copy() for panel, block in topology.groupby("panel_id", sort=False)}
    rows = []
    for pair in panel_pairs.itertuples(index=False):
        if pair.left_panel_id not in by_panel or pair.right_panel_id not in by_panel:
            raise RuntimeError("pair manifest refers to a panel absent from PASS topology")
        for left, right in (
            (by_panel[pair.left_panel_id], by_panel[pair.right_panel_id]),
            (by_panel[pair.right_panel_id], by_panel[pair.left_panel_id]),
        ):
            merged = left.merge(right, on="comparison_stratum", suffixes=("__query", "__target"))
            for relation in merged.itertuples(index=False):
                rows.append({
                    "query_panel_id": relation.panel_id__query,
                    "query_contrast_id": relation.contrast_id__query,
                    "target_panel_id": relation.panel_id__target,
                    "target_contrast_id": relation.contrast_id__target,
                    "truth_class": "unknown",
                    "expected_direction": "not_applicable",
                    "truth_family_id": f"metadata_candidate::{relation.comparison_stratum}",
                    "eligible_primary": False,
                    "evidence_basis": "same_metadata_stratum_only_not_paper_adjudicated_truth",
                    "source_independence_status": "preliminary_source_disjoint_only",
                    "truth_frozen_before_scores": True,
                    "selection_blind_to_transport_scores": True,
                    "comparison_stratum": relation.comparison_stratum,
                })
    registry = pd.DataFrame(rows)
    if not registry.empty and registry.duplicated([
        "query_panel_id", "query_contrast_id", "target_panel_id", "target_contrast_id"
    ]).any():
        raise RuntimeError("candidate registry contains duplicate ordered pairs")
    return registry.sort_values([
        "query_panel_id", "query_contrast_id", "target_panel_id", "target_contrast_id"
    ], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--panel-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    topology = pd.read_csv(args.topology, sep="\t", dtype=str)
    panel_pairs = pd.read_csv(args.panel_pairs, sep="\t", dtype=str)
    registry = build_registry(topology, panel_pairs)
    pass_topology = topology.loc[topology.topology_status.astype(str).str.startswith("PASS")]
    contrast_counts = pass_topology.groupby("panel_id").contrast_id.nunique()
    ordered_universe = int(sum(
        2 * int(contrast_counts[left]) * int(contrast_counts[right])
        for left, right in zip(panel_pairs.left_panel_id, panel_pairs.right_panel_id)
    ))
    args.output_dir.mkdir(parents=True)
    registry_path = args.output_dir / "truth_registry.tsv"
    registry.to_csv(registry_path, sep="\t", index=False)
    report = {
        "schema": "pan_redu_pretruth_candidate_registry_v1",
        "status": "PRETRUTH_METADATA_CANDIDATES_FROZEN_ZERO_GOLD",
        "denominators": {
            "ordered_candidate_universe": ordered_universe,
            "same_metadata_stratum_candidate_pairs": len(registry),
            "unlabeled_or_unreviewed_pairs": ordered_universe,
            "claim_bearing_replication_pairs": 0,
            "claim_bearing_truth_families": 0,
        },
        "selection_rule": "same comparison_stratum across preliminary source-disjoint panels",
        "leakage_status": "topology and source provenance only; no abundance, effect, embedding, transport or score read",
        "scientific_boundary": (
            "Equal sex, age or generic health metadata is a review candidate, not biological truth; "
            "all rows remain unknown until paper/design adjudication confirms an exact response axis."
        ),
        "provenance": {
            "topology_sha256": sha256(args.topology),
            "panel_pairs_sha256": sha256(args.panel_pairs),
            "registry_sha256": sha256(registry_path),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
