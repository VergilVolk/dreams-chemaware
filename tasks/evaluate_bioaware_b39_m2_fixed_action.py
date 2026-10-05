#!/usr/bin/env python
"""Evaluate the preregistered B39-M2 atomic BioAware actions.

The evaluator consumes the outcome-blind M0/M1 ledgers only after the action
matrix has been frozen.  All cells start from the same frozen B37 catalogue
topology output and share one DreaMS-only risk layer.  No cell-specific model,
threshold or gate is fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


INTERNAL_SOURCES = ("BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary, index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} misses columns: {sorted(missing)}")


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def cluster_ci(
    values: np.ndarray,
    clusters: Iterable[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "cluster": list(clusters)})
    grouped = frame.groupby("cluster", sort=False)["value"].agg(["sum", "count"])
    if grouped.empty:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "clusters": 0, "resamples": repeats}
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def exact_mcnemar(corrected: int, introduced: int) -> float:
    discordant = int(corrected + introduced)
    if discordant == 0:
        return 1.0
    lower = min(int(corrected), int(introduced))
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * tail))


def replay_inputs(
    candidates: pd.DataFrame,
    topology: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = candidates.loc[candidates["query_id"].isin(topology["query_id"])].copy()
    query = candidates.groupby("query_id", sort=False).agg(
        truth_candidate_id=("truth_candidate_id", "first"),
        truth_formula=("truth_formula", "first"),
        source=("source", "first"),
        baseline_candidate_id=("baseline_candidate_id", "first"),
        baseline_correct=("baseline_correct", "first"),
        baseline_gap=("baseline_gap", "first"),
        candidate_count=("candidate_id", "nunique"),
        positive_count=("is_positive", "sum"),
    ).reset_index()
    merged = topology.merge(query, on="query_id", suffixes=("_b37", "_candidate"), validate="one_to_one")
    if len(merged) != len(topology):
        raise RuntimeError(f"candidate/topology query mismatch: {len(merged)} != {len(topology)}")
    categorical = {}
    for column in ("truth_candidate_id", "truth_formula", "source", "baseline_candidate_id", "baseline_correct"):
        left = merged[f"{column}_b37"].astype(str)
        right = merged[f"{column}_candidate"].astype(str)
        categorical[column] = int((left != right).sum())
    numeric = {
        "baseline_gap": float(np.max(np.abs(
            merged["baseline_gap_b37"].to_numpy(float) - merged["baseline_gap_candidate"].to_numpy(float)
        ))),
        "candidate_count": int(np.max(np.abs(
            merged["candidate_count_b37"].to_numpy(int) - merged["candidate_count_candidate"].to_numpy(int)
        ))),
    }
    if any(categorical.values()) or numeric["baseline_gap"] > 1e-10 or numeric["candidate_count"] != 0:
        raise RuntimeError(f"B37/candidate replay mismatch: categorical={categorical} numeric={numeric}")
    if not (merged["positive_count"].to_numpy(int) == 1).all():
        raise RuntimeError("evaluation requires exactly one positive candidate per query")
    keep = topology.copy()
    keep["baseline_correct"] = parse_bool(keep["baseline_correct"])
    keep["final_correct"] = parse_bool(keep["final_correct"])
    keep["intervene"] = parse_bool(keep["intervene"])
    return keep, {
        "queries": int(len(keep)),
        "categorical_mismatches": categorical,
        "maximum_baseline_gap_error": numeric["baseline_gap"],
        "maximum_candidate_count_error": numeric["candidate_count"],
        "pass": True,
    }


def direction_mask(events: pd.DataFrame, direction: str) -> pd.Series:
    status = events["direction_status"].astype(str)
    if direction == "supported":
        return status.eq("supported")
    if direction == "supported_bidirectional":
        return status.eq("supported_bidirectional")
    if direction == "supported_or_bidirectional":
        return status.isin({"supported", "supported_bidirectional"})
    if direction == "nonconflicted":
        return ~status.eq("conflicted")
    if direction == "conflicted":
        return status.eq("conflicted")
    raise RuntimeError(f"unsupported constructible direction: {direction}")


def score_events(events: pd.DataFrame, cell: dict[str, Any]) -> pd.DataFrame:
    mask = events["edge_source"].astype(str).eq("Rhea")
    mask &= ~parse_bool(events["reaction_identity_noop"])
    mask &= direction_mask(events, str(cell["direction"]))
    if cell["require_hyperedge_complete"]:
        mask &= parse_bool(events["hyperedge_context_complete"])
    if cell["require_candidate_specific"]:
        mask &= parse_bool(events["event_candidate_specific"])
    local = events.loc[mask].copy()
    evidence = str(cell["evidence"])
    if evidence == "spectral":
        available = parse_bool(local["spectral_available"])
        strength = np.clip(local["spectral_excess_dual_view"].fillna(0.0).to_numpy(float), 0.0, 1.0)
    elif evidence == "coabundance":
        available = parse_bool(local["coabundance_available"])
        strength = np.clip(local["coabundance_excess_positive"].fillna(0.0).to_numpy(float), 0.0, 1.0)
    elif evidence == "both":
        available = parse_bool(local["spectral_available"]) & parse_bool(local["coabundance_available"])
        spectral = np.clip(local["spectral_excess_dual_view"].fillna(0.0).to_numpy(float), 0.0, 1.0)
        coabundance = np.clip(local["coabundance_excess_positive"].fillna(0.0).to_numpy(float), 0.0, 1.0)
        strength = np.minimum(spectral, coabundance)
    elif evidence == "knowledge_only":
        available = pd.Series(True, index=local.index)
        strength = np.ones(len(local), dtype=float)
    else:
        raise RuntimeError(f"unsupported evidence transform: {evidence}")
    local = local.loc[available].copy()
    local["event_support"] = np.asarray(strength)[available.to_numpy(bool)]
    local = local.loc[local["event_support"] > 0].copy()
    return local


def candidate_supports(
    events: pd.DataFrame,
    contexts: pd.DataFrame,
    cell: dict[str, Any],
) -> pd.DataFrame:
    selected = score_events(events, cell)
    denominators = contexts.groupby(["query_id", "candidate_id"], sort=False).agg(
        visible_contexts=("seed_stratum", "nunique")
    ).reset_index()
    if selected.empty:
        denominators["supported_contexts"] = 0
        denominators["support_sum"] = 0.0
        denominators["candidate_support"] = 0.0
        return denominators
    dependency = selected.groupby(
        ["query_id", "candidate_id", "seed_stratum", "dependency_group_id"], sort=False
    )["event_support"].max().reset_index()
    per_context = dependency.groupby(
        ["query_id", "candidate_id", "seed_stratum"], sort=False
    )["event_support"].max().reset_index()
    aggregate = per_context.groupby(["query_id", "candidate_id"], sort=False).agg(
        supported_contexts=("seed_stratum", "nunique"),
        support_sum=("event_support", "sum"),
    ).reset_index()
    output = denominators.merge(aggregate, on=["query_id", "candidate_id"], how="left", validate="one_to_one")
    output[["supported_contexts", "support_sum"]] = output[["supported_contexts", "support_sum"]].fillna(0.0)
    output["candidate_support"] = output["support_sum"] / output["visible_contexts"].clip(lower=1)
    return output


def proposals(support: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, local in support.groupby("query_id", sort=False):
        maximum = float(local["candidate_support"].max())
        winners = local.loc[np.isclose(local["candidate_support"], maximum, atol=1e-12, rtol=0.0), "candidate_id"].astype(str).tolist()
        rows.append({
            "query_id": str(query_id),
            "atomic_max_support": maximum,
            "atomic_supported_candidates": int((local["candidate_support"] > 0).sum()),
            "atomic_proposal_unique": bool(maximum > 0 and len(winners) == 1),
            "atomic_proposed_candidate_id": winners[0] if maximum > 0 and len(winners) == 1 else "",
        })
    return pd.DataFrame(rows)


def summarize_transition(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, Any]:
    delta_topology = frame["atomic_final_correct"].astype(int) - frame["topology_correct"].astype(int)
    delta_dreams = frame["atomic_final_correct"].astype(int) - frame["baseline_correct"].astype(int)
    corrected_topology = int((delta_topology > 0).sum())
    introduced_topology = int((delta_topology < 0).sum())
    by_source = {}
    for source, local in frame.groupby("source", sort=False):
        delta = local["atomic_final_correct"].astype(int) - local["topology_correct"].astype(int)
        by_source[str(source)] = {
            "queries": int(len(local)),
            "delta_vs_topology": float(delta.mean()),
            "corrected_vs_topology": int((delta > 0).sum()),
            "introduced_vs_topology": int((delta < 0).sum()),
        }
    return {
        "queries": int(len(frame)),
        "dreams_recall1": float(frame["baseline_correct"].mean()),
        "topology_recall1": float(frame["topology_correct"].mean()),
        "atomic_recall1": float(frame["atomic_final_correct"].mean()),
        "delta_vs_topology": float(delta_topology.mean()),
        "delta_vs_dreams": float(delta_dreams.mean()),
        "corrected_vs_topology": corrected_topology,
        "introduced_vs_topology": introduced_topology,
        "risk_net_lambda2_vs_topology": int(corrected_topology - 2 * introduced_topology),
        "atomic_interventions": int(frame["atomic_intervene"].sum()),
        "common_risk_eligible": int(frame["common_risk_eligible"].sum()),
        "unique_positive_proposals": int(frame["atomic_proposal_unique"].sum()),
        "identity_cluster_bootstrap_vs_topology": cluster_ci(
            delta_topology.to_numpy(float), frame["truth_candidate_id"].astype(str), repeats, seed,
        ),
        "formula_cluster_bootstrap_vs_topology": cluster_ci(
            delta_topology.to_numpy(float), frame["truth_formula"].astype(str), repeats, seed + 1,
        ),
        "mcnemar_exact_p_vs_topology": exact_mcnemar(corrected_topology, introduced_topology),
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--m0-dir", type=Path, required=True)
    parser.add_argument("--m1-dir", type=Path, required=True)
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
    m0_report_path = args.m0_dir / "report.json"
    contexts_path = args.m0_dir / "candidate_context_semantics.csv.gz"
    m1_report_path = args.m1_dir / "report.json"
    events_path = args.m1_dir / "internal_atomic_events_with_edge_evidence.csv.gz"
    b37_report_path = args.b37_dir / "report.json"
    transitions_path = args.b37_dir / "catalog_ablation_transitions.csv.gz"
    required_paths = [manifest_report_path, cells_path, m0_report_path, contexts_path, m1_report_path, events_path, args.candidate_features, b37_report_path, transitions_path]
    for path in required_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    cells = json.loads(cells_path.read_text(encoding="utf-8"))
    m0_report = json.loads(m0_report_path.read_text(encoding="utf-8"))
    m1_report = json.loads(m1_report_path.read_text(encoding="utf-8"))
    b37_report = json.loads(b37_report_path.read_text(encoding="utf-8"))
    if manifest_report.get("status") != "bioaware_b39_m2_fixed_action_manifest_frozen" or manifest_report.get("provenance", {}).get("cells_sha256") != sha256(cells_path):
        raise RuntimeError("invalid or drifted B39-M2 manifest")
    if m0_report.get("provenance", {}).get("candidate_contexts") != sha256(contexts_path):
        raise RuntimeError("B39-M0 candidate-context provenance mismatch")
    if m1_report.get("provenance", {}).get("joined_events") != sha256(events_path):
        raise RuntimeError("B39-M1 joined-event provenance mismatch")
    if b37_report.get("provenance", {}).get("catalog_ablation_transitions_sha256") != sha256(transitions_path):
        raise RuntimeError("B37 transition provenance mismatch")

    candidates = pd.read_csv(args.candidate_features, low_memory=False)
    contexts = pd.read_csv(contexts_path, low_memory=False)
    events = pd.read_csv(events_path, dtype={"reaction_id": str}, low_memory=False)
    transitions = pd.read_csv(transitions_path, low_memory=False)
    require_columns(candidates, ["query_id", "candidate_id", "truth_candidate_id", "truth_formula", "source", "is_positive", "baseline_candidate_id", "baseline_correct", "baseline_gap"], "candidates")
    require_columns(contexts, ["query_id", "candidate_id", "source", "context_class", "seed_stratum"], "contexts")
    require_columns(events, ["query_id", "candidate_id", "seed_stratum", "dependency_group_id", "edge_source", "direction_status", "reaction_identity_noop", "hyperedge_context_complete", "event_candidate_specific", "spectral_available", "coabundance_available", "spectral_excess_dual_view", "coabundance_excess_positive"], "events")
    topology = transitions.loc[transitions["arm"].eq("catalog_topology") & transitions["source"].isin(INTERNAL_SOURCES)].copy()
    if len(topology) != 548 or topology["query_id"].nunique() != 548:
        raise RuntimeError(f"expected 548 internal B37 topology queries, got {len(topology)}")
    topology, replay = replay_inputs(candidates, topology)
    evaluation_queries = set(topology["query_id"].astype(str))
    candidates = candidates.loc[candidates["query_id"].astype(str).isin(evaluation_queries)].copy()
    contexts = contexts.loc[contexts["query_id"].astype(str).isin(evaluation_queries) & contexts["context_class"].eq("S")].copy()
    events = events.loc[events["query_id"].astype(str).isin(evaluation_queries) & events["context_class"].eq("S")].copy()
    if candidates["query_id"].nunique() != 548 or contexts["query_id"].nunique() != 548:
        raise RuntimeError("evaluation query coverage drift")
    if events["query_id"].nunique() >= 548:
        raise RuntimeError("unexpectedly every query has an atomic event; missing-path abstention audit broken")

    candidate_keys = candidates[["query_id", "candidate_id"]].drop_duplicates()
    context_keys = contexts[["query_id", "candidate_id"]].drop_duplicates()
    if len(candidate_keys.merge(context_keys, on=["query_id", "candidate_id"], how="left", indicator=True).query("_merge != 'both'")):
        raise RuntimeError("candidate-context ledger misses an evaluation candidate")

    base = topology.rename(columns={"final_candidate_id": "topology_candidate_id", "final_correct": "topology_correct"}).copy()
    base["common_risk_eligible"] = base["baseline_gap"].to_numpy(float) <= base["gate_margin"].to_numpy(float) + 1e-12
    all_transitions: list[pd.DataFrame] = []
    cell_reports: dict[str, Any] = {}
    for index, cell in enumerate(cells):
        cell_id = str(cell["cell_id"])
        if not cell["constructible_from_m1"]:
            cell_reports[cell_id] = {
                "cell": cell,
                "status": "not_constructible_from_m1",
                "candidate_for_external_reconstruction": False,
                "reason": cell["interpretation"],
            }
            continue
        support = candidate_supports(events, contexts, cell)
        proposal = proposals(support)
        local = base.merge(proposal, on="query_id", how="left", validate="one_to_one")
        local["atomic_max_support"] = local["atomic_max_support"].fillna(0.0)
        local["atomic_supported_candidates"] = local["atomic_supported_candidates"].fillna(0).astype(int)
        local["atomic_proposal_unique"] = parse_bool(local["atomic_proposal_unique"])
        local["atomic_proposed_candidate_id"] = local["atomic_proposed_candidate_id"].fillna("").astype(str)
        local["atomic_intervene"] = (
            local["common_risk_eligible"].astype(bool)
            & local["atomic_proposal_unique"].astype(bool)
            & local["atomic_proposed_candidate_id"].ne(local["topology_candidate_id"].astype(str))
        )
        local["atomic_final_candidate_id"] = np.where(
            local["atomic_intervene"], local["atomic_proposed_candidate_id"], local["topology_candidate_id"].astype(str)
        )
        local["atomic_final_correct"] = local["atomic_final_candidate_id"].astype(str).eq(local["truth_candidate_id"].astype(str))
        local["cell_id"] = cell_id
        local["cell_name"] = str(cell["name"])
        summary = summarize_transition(local, args.bootstrap_resamples, args.seed + 10 * index)
        internal_sources_nonnegative = all(item["delta_vs_topology"] >= -1e-15 for item in summary["by_source"].values())
        candidate = bool(
            cell["role"] == "core"
            and summary["delta_vs_topology"] >= 0.03
            and summary["corrected_vs_topology"] > 2 * summary["introduced_vs_topology"]
            and summary["identity_cluster_bootstrap_vs_topology"]["ci_low"] > 0
            and summary["formula_cluster_bootstrap_vs_topology"]["ci_low"] > 0
            and internal_sources_nonnegative
            and summary["atomic_interventions"] >= 100
        )
        cell_reports[cell_id] = {
            "cell": cell,
            "status": "evaluated",
            "summary": summary,
            "internal_sources_nonnegative": internal_sources_nonnegative,
            "candidate_for_external_reconstruction": candidate,
            "identifiability": "nonidentifying_all_events_candidate_specific" if cell["role"] == "ablation_nonidentifying" and bool(events["event_candidate_specific"].all()) else "identified_at_declared_resolution",
        }
        all_transitions.append(local[[
            "cell_id", "cell_name", "query_id", "source", "truth_candidate_id", "truth_formula",
            "baseline_candidate_id", "baseline_correct", "baseline_gap", "gate_margin",
            "topology_candidate_id", "topology_correct", "common_risk_eligible",
            "atomic_max_support", "atomic_supported_candidates", "atomic_proposal_unique",
            "atomic_proposed_candidate_id", "atomic_intervene", "atomic_final_candidate_id",
            "atomic_final_correct",
        ]])

    transition_frame = pd.concat(all_transitions, ignore_index=True)
    transitions_out = args.output_dir / "cell_transitions.csv.gz"
    atomic_csv_gzip(transitions_out, transition_frame)
    selected = [cell_id for cell_id, item in cell_reports.items() if item.get("candidate_for_external_reconstruction")]
    report = {
        "status": "bioaware_b39_m2_internal_fixed_action_complete",
        "formal": True,
        "protocol": "preregistered atomic-event action cells; fixed B37 topology start; shared DreaMS-only risk layer; four internal sources",
        "evaluation_queries": int(len(base)),
        "evaluation_sources": list(INTERNAL_SOURCES),
        "queries_with_any_atomic_event": int(events["query_id"].nunique()),
        "official_dreams_recall1": float(base["baseline_correct"].mean()),
        "frozen_topology_recall1": float(base["topology_correct"].mean()),
        "frozen_topology_delta_vs_dreams": float(base["topology_correct"].mean() - base["baseline_correct"].mean()),
        "common_risk_eligible_queries": int(base["common_risk_eligible"].sum()),
        "input_replay": replay,
        "cell_reports": cell_reports,
        "cells_candidate_for_external_reconstruction": selected,
        "pass_to_external_event_reconstruction": bool(selected),
        "pass_to_context_representation": False,
        "context_representation_blockers": [
            "M1 contains no event-specific L/H external-source evidence",
            "no prospective-unknown source exists in the current six-source panel",
            "matched non-neighbour and structure-rewire atomic null rows are not yet constructible",
            "candidate-specificity ablation is non-identifying because every current event is candidate-specific",
        ],
        "coverage": {
            "event_candidate_specific_fraction": float(parse_bool(events["event_candidate_specific"]).mean()),
            "rhea_hyperedge_complete_fraction": float(parse_bool(events.loc[events["edge_source"].eq("Rhea"), "hyperedge_context_complete"]).mean()),
            "event_direction_counts": dict(Counter(events["direction_status"].astype(str))),
        },
        "provenance": {
            "manifest_report": sha256(manifest_report_path),
            "cells": sha256(cells_path),
            "m0_report": sha256(m0_report_path),
            "candidate_contexts": sha256(contexts_path),
            "m1_report": sha256(m1_report_path),
            "atomic_events_with_edge_evidence": sha256(events_path),
            "candidate_features": sha256(args.candidate_features),
            "b37_report": sha256(b37_report_path),
            "b37_transitions": sha256(transitions_path),
            "cell_transitions": sha256(transitions_out),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Internal fixed-action screen only. It cannot establish external BioAware gain, prospective unknown annotation, reaction-ID causality or shared-embedding improvement.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
