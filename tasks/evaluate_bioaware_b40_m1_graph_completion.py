#!/usr/bin/env python
"""Evaluate preregistered B40 soft graph-completion actions and nulls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from evaluate_bioaware_b39_m2_fixed_action import (  # noqa: E402
    INTERNAL_SOURCES,
    atomic_csv_gzip,
    atomic_json,
    cluster_ci,
    exact_mcnemar,
    parse_bool,
    replay_inputs,
    sha256,
)


def aggregate_scores(scores: pd.DataFrame, score_column: str) -> pd.DataFrame:
    aggregate = scores.groupby(["query_id", "candidate_id"], sort=False).agg(
        graph_score=(score_column, "mean"),
        candidate_rhea_mapped=("candidate_rhea_mapped", "max"),
        visible_contexts=("seed_stratum", "nunique"),
    ).reset_index()
    mapping = aggregate.groupby("query_id", sort=False)["candidate_rhea_mapped"].sum().rename("mapped_candidate_count").reset_index()
    aggregate = aggregate.merge(mapping, on="query_id", validate="many_to_one")
    return aggregate


def proposals(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, local in aggregate.groupby("query_id", sort=False):
        maximum = float(local["graph_score"].max())
        winners = local.loc[np.isclose(local["graph_score"], maximum, atol=1e-12, rtol=0.0), "candidate_id"].astype(str).tolist()
        rows.append({
            "query_id": str(query_id),
            "mapped_candidate_count": int(local["mapped_candidate_count"].iloc[0]),
            "graph_max_score": maximum,
            "graph_proposal_unique": bool(maximum > 0 and len(winners) == 1),
            "graph_proposed_candidate_id": winners[0] if maximum > 0 and len(winners) == 1 else "",
        })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, Any]:
    effect = frame["graph_final_correct"].astype(int) - frame["topology_correct"].astype(int)
    corrected = int((effect > 0).sum())
    introduced = int((effect < 0).sum())
    by_source = {}
    for source, local in frame.groupby("source", sort=False):
        local_effect = local["graph_final_correct"].astype(int) - local["topology_correct"].astype(int)
        by_source[str(source)] = {
            "queries": int(len(local)),
            "delta_vs_topology": float(local_effect.mean()),
            "corrected": int((local_effect > 0).sum()),
            "introduced": int((local_effect < 0).sum()),
        }
    return {
        "queries": int(len(frame)),
        "dreams_recall1": float(frame["baseline_correct"].mean()),
        "topology_recall1": float(frame["topology_correct"].mean()),
        "graph_recall1": float(frame["graph_final_correct"].mean()),
        "delta_vs_topology": float(effect.mean()),
        "delta_vs_dreams": float((frame["graph_final_correct"].astype(int) - frame["baseline_correct"].astype(int)).mean()),
        "corrected_vs_topology": corrected,
        "introduced_vs_topology": introduced,
        "risk_net_lambda2": int(corrected - 2 * introduced),
        "interventions": int(frame["graph_intervene"].sum()),
        "unique_proposals": int(frame["graph_proposal_unique"].sum()),
        "identity_cluster_bootstrap": cluster_ci(effect.to_numpy(float), frame["truth_candidate_id"].astype(str), repeats, seed),
        "formula_cluster_bootstrap": cluster_ci(effect.to_numpy(float), frame["truth_formula"].astype(str), repeats, seed + 1),
        "mcnemar_exact_p": exact_mcnemar(corrected, introduced),
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--b40-m0-dir", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, default=Path("data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz"))
    parser.add_argument("--b37-dir", type=Path, default=Path("data/validation/bioaware_b37_local_fullcheck_v2_20260913"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
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
    for path in (manifest_report_path, cells_path, b40_report_path, scores_path, args.candidate_features, b37_report_path, transitions_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    matrix = json.loads(cells_path.read_text(encoding="utf-8"))
    b40_report = json.loads(b40_report_path.read_text(encoding="utf-8"))
    b37_report = json.loads(b37_report_path.read_text(encoding="utf-8"))
    if manifest_report.get("status") != "bioaware_b40_m1_graph_completion_manifest_frozen" or manifest_report.get("provenance", {}).get("cells") != sha256(cells_path):
        raise RuntimeError("invalid B40-M1 manifest")
    if not b40_report.get("pass_to_fixed_action_evaluation"):
        raise RuntimeError(f"B40-M0 did not pass: {b40_report.get('gates')}")
    if b40_report.get("provenance", {}).get("scores") != sha256(scores_path):
        raise RuntimeError("B40-M0 score provenance mismatch")
    if b37_report.get("provenance", {}).get("catalog_ablation_transitions_sha256") != sha256(transitions_path):
        raise RuntimeError("B37 transition provenance mismatch")

    scores = pd.read_csv(scores_path, low_memory=False)
    candidates = pd.read_csv(args.candidate_features, low_memory=False)
    transitions = pd.read_csv(transitions_path, low_memory=False)
    topology = transitions.loc[transitions["arm"].eq("catalog_topology") & transitions["source"].isin(INTERNAL_SOURCES)].copy()
    if len(topology) != 548 or topology["query_id"].nunique() != 548:
        raise RuntimeError("internal B37 topology panel drift")
    topology, replay = replay_inputs(candidates, topology)
    queries = set(topology["query_id"].astype(str))
    scores = scores.loc[scores["query_id"].astype(str).isin(queries)].copy()
    if scores["query_id"].nunique() != 548:
        raise RuntimeError("B40 score query coverage drift")
    base = topology.rename(columns={"final_candidate_id": "topology_candidate_id", "final_correct": "topology_correct"}).copy()
    base["common_risk_eligible"] = base["baseline_gap"].to_numpy(float) <= base["gate_margin"].to_numpy(float) + 1e-12

    reports: dict[str, Any] = {}
    transition_frames = []
    for index, cell in enumerate(matrix):
        aggregate = aggregate_scores(scores, str(cell["score"]))
        proposal = proposals(aggregate)
        local = base.merge(proposal, on="query_id", validate="one_to_one")
        local["identifiable_mapped_competition"] = local["mapped_candidate_count"].ge(2)
        local["graph_intervene"] = (
            local["identifiable_mapped_competition"]
            & local["common_risk_eligible"]
            & local["graph_proposal_unique"]
            & local["graph_proposed_candidate_id"].astype(str).ne(local["topology_candidate_id"].astype(str))
        )
        local["graph_final_candidate_id"] = np.where(local["graph_intervene"], local["graph_proposed_candidate_id"], local["topology_candidate_id"])
        local["graph_final_correct"] = local["graph_final_candidate_id"].astype(str).eq(local["truth_candidate_id"].astype(str))
        primary = local.loc[local["identifiable_mapped_competition"]].copy()
        if len(primary) != 298:
            raise RuntimeError(f"mapped-competition subgroup drift: {len(primary)}")
        reports[str(cell["cell_id"])] = {
            "cell": cell,
            "overall": summarize(local, args.bootstrap_resamples, args.seed + 20 * index),
            "mapped_competition_primary": summarize(primary, args.bootstrap_resamples, args.seed + 20 * index + 2),
        }
        local["cell_id"] = str(cell["cell_id"])
        local["cell_name"] = str(cell["name"])
        transition_frames.append(local[[
            "cell_id", "cell_name", "query_id", "source", "truth_candidate_id", "truth_formula",
            "baseline_candidate_id", "baseline_correct", "topology_candidate_id", "topology_correct",
            "baseline_gap", "gate_margin", "common_risk_eligible", "mapped_candidate_count",
            "identifiable_mapped_competition", "graph_max_score", "graph_proposal_unique",
            "graph_proposed_candidate_id", "graph_intervene", "graph_final_candidate_id", "graph_final_correct",
        ]])

    null_deltas = [reports[cell_id]["mapped_competition_primary"]["delta_vs_topology"] for cell_id in ("B40-02", "B40-03", "B40-04")]
    selected = []
    for cell_id in ("B40-00", "B40-01"):
        item = reports[cell_id]
        primary = item["mapped_competition_primary"]
        overall = item["overall"]
        source_nonnegative = all(value["delta_vs_topology"] >= -1e-15 for value in primary["by_source"].values())
        gates = {
            "primary_delta_ge_0_03": bool(primary["delta_vs_topology"] >= 0.03),
            "overall_nonnegative": bool(overall["delta_vs_topology"] >= -1e-15),
            "corrected_gt_2x_introduced": bool(primary["corrected_vs_topology"] > 2 * primary["introduced_vs_topology"]),
            "identity_ci_low_positive": bool(primary["identity_cluster_bootstrap"]["ci_low"] > 0),
            "formula_ci_low_positive": bool(primary["formula_cluster_bootstrap"]["ci_low"] > 0),
            "each_internal_source_nonnegative": source_nonnegative,
            "interventions_ge_30": bool(primary["interventions"] >= 30),
            "beats_each_seed_null_delta": bool(primary["delta_vs_topology"] > max(null_deltas) + 1e-15),
        }
        item["gates"] = gates
        item["candidate_for_external_reconstruction"] = bool(all(gates.values()))
        if item["candidate_for_external_reconstruction"]:
            selected.append(cell_id)
    for cell_id in ("B40-02", "B40-03", "B40-04", "B40-05"):
        reports[cell_id]["candidate_for_external_reconstruction"] = False

    transition_frame = pd.concat(transition_frames, ignore_index=True)
    transitions_out = args.output_dir / "cell_transitions.csv.gz"
    atomic_csv_gzip(transitions_out, transition_frame)
    report = {
        "status": "bioaware_b40_m1_graph_completion_evaluation_complete",
        "formal": True,
        "protocol": "fixed four-hop Rhea diffusion versus matched seed permutations on outcome-blind mapped-competition subgroup; common topology start and DreaMS-only risk layer",
        "evaluation_queries": 548,
        "mapped_competition_primary_queries": 298,
        "cell_reports": reports,
        "seed_null_primary_deltas": null_deltas,
        "cells_candidate_for_external_reconstruction": selected,
        "pass_to_external_reconstruction": bool(selected),
        "pass_to_context_representation": False,
        "input_replay": replay,
        "provenance": {
            "manifest_report": sha256(manifest_report_path),
            "cells": sha256(cells_path),
            "b40_m0_report": sha256(b40_report_path),
            "b40_scores": sha256(scores_path),
            "candidate_features": sha256(args.candidate_features),
            "b37_report": sha256(b37_report_path),
            "b37_transitions": sha256(transitions_path),
            "cell_transitions": sha256(transitions_out),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Opened internal graph-completion screen. Passing only permits source-stratified external reconstruction; it does not establish prospective annotation, reaction causality, embedding gain or SOTA.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
