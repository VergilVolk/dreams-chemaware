#!/usr/bin/env python
"""Evaluate frozen B40 actions separately in synthetic, sample-local and hidden-standard strata.

This is an opened-data mechanism audit.  It deliberately reuses the exact
B40-M1 manifest, B40-M0 scores, and B37 topology decisions.  No threshold,
weight, graph parameter or action is selected from the results produced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from evaluate_bioaware_b39_m2_fixed_action import (  # noqa: E402
    atomic_csv_gzip,
    atomic_json,
    sha256,
)
from evaluate_bioaware_b40_m1_graph_completion import (  # noqa: E402
    aggregate_scores,
    proposals,
    summarize,
)


EXPECTED_SOURCE_QUERIES = {
    "BV2cell": 95,
    "Mouse_brain": 131,
    "Mouse_liver": 176,
    "NIST_plasma": 146,
    "ST001154_same_formula_10ppm": 150,
    "KGMN200STD_hidden_seed": 162,
}
STRATUM_BY_SOURCE = {
    "BV2cell": "synthetic_rotation",
    "Mouse_brain": "synthetic_rotation",
    "Mouse_liver": "synthetic_rotation",
    "NIST_plasma": "synthetic_rotation",
    "ST001154_same_formula_10ppm": "sample_local_leave_one_seed_out",
    "KGMN200STD_hidden_seed": "hidden_standard",
}


def prepare_topology(transitions: pd.DataFrame, b37_report: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    topology = transitions.loc[transitions["arm"].eq("catalog_topology")].copy()
    if len(topology) != 860 or topology["query_id"].nunique() != 860:
        raise RuntimeError("B37 topology panel must contain exactly 860 unique queries")
    counts = topology.groupby("source")["query_id"].nunique().astype(int).to_dict()
    if counts != EXPECTED_SOURCE_QUERIES:
        raise RuntimeError(f"B37 topology source counts drifted: {counts}")
    if topology.duplicated("query_id").any():
        raise RuntimeError("duplicate B37 topology query")
    topology["baseline_correct"] = topology["baseline_correct"].astype(bool)
    topology["topology_correct"] = topology["final_correct"].astype(bool)
    topology["topology_candidate_id"] = topology["final_candidate_id"].astype(str)
    topology["common_risk_eligible"] = (
        topology["baseline_gap"].to_numpy(float)
        <= topology["gate_margin"].to_numpy(float) + 1e-12
    )
    archived = b37_report["catalog_ablation"]["arms"]["catalog_topology"]
    checks: dict[str, Any] = {"source_counts": counts, "by_source": {}}
    for source, expected in archived["by_domain"].items():
        local = topology.loc[topology["source"].eq(source)]
        observed = {
            "baseline_recall1": float(local["baseline_correct"].mean()),
            "recall1": float(local["topology_correct"].mean()),
            "corrected": int((~local["baseline_correct"] & local["topology_correct"]).sum()),
            "introduced": int((local["baseline_correct"] & ~local["topology_correct"]).sum()),
        }
        maximum_error = max(
            abs(observed["baseline_recall1"] - float(expected["baseline_recall1"])),
            abs(observed["recall1"] - float(expected["recall1"])),
        )
        count_match = (
            observed["corrected"] == int(expected["corrected"])
            and observed["introduced"] == int(expected["introduced"])
        )
        if maximum_error > 1e-12 or not count_match:
            raise RuntimeError(f"B37 topology replay failed for {source}: {observed} vs {expected}")
        checks["by_source"][source] = {**observed, "maximum_recall_error": maximum_error}
    checks["pass"] = True
    return topology, checks


def apply_action(base: pd.DataFrame, proposal: pd.DataFrame) -> pd.DataFrame:
    local = base.merge(proposal, on="query_id", validate="one_to_one")
    local["identifiable_mapped_competition"] = local["mapped_candidate_count"].ge(2)
    local["graph_intervene"] = (
        local["identifiable_mapped_competition"]
        & local["common_risk_eligible"]
        & local["graph_proposal_unique"]
        & local["graph_proposed_candidate_id"].astype(str).ne(local["topology_candidate_id"])
    )
    local["graph_final_candidate_id"] = np.where(
        local["graph_intervene"],
        local["graph_proposed_candidate_id"],
        local["topology_candidate_id"],
    )
    local["graph_final_correct"] = local["graph_final_candidate_id"].astype(str).eq(
        local["truth_candidate_id"].astype(str)
    )
    local["context_stratum"] = local["source"].map(STRATUM_BY_SOURCE)
    if local["context_stratum"].isna().any():
        raise RuntimeError("unknown B40 context stratum")
    return local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--b40-m0-dir", type=Path, required=True)
    parser.add_argument(
        "--b37-dir", type=Path,
        default=Path("data/validation/bioaware_b37_local_fullcheck_v2_20260913"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_report_path = args.manifest_dir / "report.json"
    cells_path = args.manifest_dir / "cells.json"
    b40_report_path = args.b40_m0_dir / "report.json"
    scores_path = args.b40_m0_dir / "candidate_context_diffusion.csv.gz"
    b37_report_path = args.b37_dir / "report.json"
    transitions_path = args.b37_dir / "catalog_ablation_transitions.csv.gz"
    for path in (
        manifest_report_path, cells_path, b40_report_path, scores_path,
        b37_report_path, transitions_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    cells = json.loads(cells_path.read_text(encoding="utf-8"))
    b40_report = json.loads(b40_report_path.read_text(encoding="utf-8"))
    b37_report = json.loads(b37_report_path.read_text(encoding="utf-8"))
    if manifest_report.get("status") != "bioaware_b40_m1_graph_completion_manifest_frozen":
        raise RuntimeError("invalid frozen B40-M1 manifest")
    if manifest_report.get("provenance", {}).get("cells") != sha256(cells_path):
        raise RuntimeError("B40-M1 cell provenance mismatch")
    if not b40_report.get("pass_to_fixed_action_evaluation"):
        raise RuntimeError("B40-M0 did not pass")
    if b40_report.get("provenance", {}).get("scores") != sha256(scores_path):
        raise RuntimeError("B40 score provenance mismatch")
    if b37_report.get("provenance", {}).get("catalog_ablation_transitions_sha256") != sha256(transitions_path):
        raise RuntimeError("B37 transition provenance mismatch")

    scores = pd.read_csv(scores_path, low_memory=False)
    transitions = pd.read_csv(transitions_path, low_memory=False)
    topology, topology_replay = prepare_topology(transitions, b37_report)
    query_ids = set(topology["query_id"].astype(str))
    scores = scores.loc[scores["query_id"].astype(str).isin(query_ids)].copy()
    if scores["query_id"].nunique() != 860:
        raise RuntimeError("B40 scores do not cover all 860 B37 queries")

    cell_reports: dict[str, Any] = {}
    output_frames: list[pd.DataFrame] = []
    for index, cell in enumerate(cells):
        aggregate = aggregate_scores(scores, str(cell["score"]))
        local = apply_action(topology, proposals(aggregate))
        strata: dict[str, Any] = {}
        for offset, (stratum, frame) in enumerate(local.groupby("context_stratum", sort=True)):
            primary = frame.loc[frame["identifiable_mapped_competition"]].copy()
            strata[str(stratum)] = {
                "all_queries": summarize(frame, args.bootstrap_resamples, args.seed + 100 * index + 10 * offset),
                "mapped_competition": summarize(primary, args.bootstrap_resamples, args.seed + 100 * index + 10 * offset + 2),
            }
        by_source = {
            str(source): {
                "all_queries": summarize(frame, args.bootstrap_resamples, args.seed + 100 * index + 60 + offset),
                "mapped_competition": summarize(
                    frame.loc[frame["identifiable_mapped_competition"]],
                    args.bootstrap_resamples,
                    args.seed + 100 * index + 70 + offset,
                ),
            }
            for offset, (source, frame) in enumerate(local.groupby("source", sort=True))
        }
        cell_reports[str(cell["cell_id"])] = {
            "cell": cell,
            "strata": strata,
            "by_source": by_source,
        }
        local["cell_id"] = str(cell["cell_id"])
        local["cell_name"] = str(cell["name"])
        output_frames.append(local[[
            "cell_id", "cell_name", "query_id", "source", "context_stratum",
            "truth_candidate_id", "truth_formula", "baseline_candidate_id",
            "baseline_correct", "topology_candidate_id", "topology_correct",
            "baseline_gap", "gate_margin", "common_risk_eligible",
            "mapped_candidate_count", "identifiable_mapped_competition",
            "graph_max_score", "graph_proposal_unique", "graph_proposed_candidate_id",
            "graph_intervene", "graph_final_candidate_id", "graph_final_correct",
        ]])

    core = cell_reports["B40-00"]
    sample = core["strata"]["sample_local_leave_one_seed_out"]["mapped_competition"]
    hidden = core["strata"]["hidden_standard"]["mapped_competition"]
    synthetic = core["strata"]["synthetic_rotation"]["mapped_competition"]
    null_sample_deltas = [
        cell_reports[cell_id]["strata"]["sample_local_leave_one_seed_out"]["mapped_competition"]["delta_vs_topology"]
        for cell_id in ("B40-02", "B40-03", "B40-04")
    ]
    gates = {
        "sample_local_delta_ge_0_03": bool(sample["delta_vs_topology"] >= 0.03),
        "sample_local_corrected_gt_2x_introduced": bool(
            sample["corrected_vs_topology"] > 2 * sample["introduced_vs_topology"]
        ),
        "sample_local_identity_ci_low_positive": bool(sample["identity_cluster_bootstrap"]["ci_low"] > 0),
        "sample_local_formula_ci_low_positive": bool(sample["formula_cluster_bootstrap"]["ci_low"] > 0),
        "sample_local_beats_each_seed_null": bool(sample["delta_vs_topology"] > max(null_sample_deltas) + 1e-15),
        "hidden_standard_nonnegative": bool(hidden["delta_vs_topology"] >= -1e-15),
        "synthetic_nonnegative": bool(synthetic["delta_vs_topology"] >= -1e-15),
    }
    passed = bool(all(gates.values()))

    transitions_out = args.output_dir / "cell_context_transitions.csv.gz"
    atomic_csv_gzip(transitions_out, pd.concat(output_frames, ignore_index=True))
    report = {
        "status": "bioaware_b40_m2_context_strata_evaluation_complete",
        "formal": True,
        "protocol": "frozen B40-M1 actions evaluated separately in synthetic-rotation, sample-local leave-one-seed-out and hidden-standard opened strata",
        "evaluation_queries": 860,
        "context_stratum_queries": {
            "synthetic_rotation": 548,
            "sample_local_leave_one_seed_out": 150,
            "hidden_standard": 162,
        },
        "cell_reports": cell_reports,
        "core_decision_gates": gates,
        "pass_to_prospective_context_reconstruction": passed,
        "pass_to_context_representation": False,
        "topology_replay": topology_replay,
        "provenance": {
            "manifest_report": sha256(manifest_report_path),
            "cells": sha256(cells_path),
            "b40_report": sha256(b40_report_path),
            "b40_scores": sha256(scores_path),
            "b37_report": sha256(b37_report_path),
            "b37_transitions": sha256(transitions_path),
            "context_transitions": sha256(transitions_out),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Opened source-stratified mechanism audit. ST is leave-one-seed-out from an already known sample seed table and KGMN is a standards mixture; neither is prospective unknown annotation, blind SOTA evidence, reaction causality or shared-embedding gain.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
