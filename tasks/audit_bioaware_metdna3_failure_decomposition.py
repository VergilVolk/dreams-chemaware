#!/usr/bin/env python
"""Decompose BioAware failures on the consumed MetDNA3 HILIC development set.

The audit separates graph topology, available Level-1 seed coverage, raw-MS2
edge availability, raw-MS2 edge direction, and downstream rescue headroom.
Unique queries and identity-rotation instances are reported independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def official_top1(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_id, group in scores.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        maximum = float(group["spectral_score"].max())
        tied = group[np.isclose(group["spectral_score"], maximum, rtol=0, atol=1e-12)]
        ordered = tied.sort_values("candidate_id")
        predicted = str(ordered.iloc[0].candidate_id)
        correct = len(tied) == 1 and predicted == truth
        rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": truth,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_top_candidate": predicted,
            "baseline_correct": bool(correct),
            "baseline_tie_size": int(len(tied)),
            "candidate_count": int(len(group)),
        })
    return pd.DataFrame(rows)


def eligible_opposite_side_neighbors(
    participants: pd.DataFrame, available_identities: set[str], maximum_seed_degree: int
) -> tuple[set[str], dict[str, set[str]], set[str]]:
    frame = participants.drop_duplicates(["reaction_id", "side", "compound_id"]).copy()
    degree = frame.groupby("compound_id")["reaction_id"].nunique()
    currency = set(frame.loc[frame["is_currency"].astype(bool), "compound_id"].astype(str))
    eligible_seeds = {
        identity for identity in available_identities
        if 0 < int(degree.get(identity, 0)) <= maximum_seed_degree and identity not in currency
    }
    nodes = set(frame["compound_id"].astype(str))
    neighbor_map: dict[str, set[str]] = {}
    for _, reaction in frame.groupby("reaction_id", sort=False):
        left = set(reaction.loc[reaction["side"] == "left", "compound_id"].astype(str))
        right = set(reaction.loc[reaction["side"] == "right", "compound_id"].astype(str))
        for candidate in left:
            neighbor_map.setdefault(candidate, set()).update(right & eligible_seeds)
        for candidate in right:
            neighbor_map.setdefault(candidate, set()).update(left & eligible_seeds)
    return nodes, neighbor_map, eligible_seeds


def primary_bottleneck(row: pd.Series) -> str:
    if not row["truth_in_rhea"]:
        return "A_truth_absent_from_rhea"
    if row["eligible_level1_seed_neighbors"] == 0:
        return "B_no_eligible_level1_seed_neighbor"
    if row["truth_network_path_rotations"] == 0:
        return "C_eligible_neighbor_but_no_rotation_path"
    if row["truth_raw_path_rotations"] == 0:
        return "D_network_path_but_no_raw_seed_spectrum"
    if row["paired_raw_rotations"] == 0:
        return "E_raw_truth_path_without_competing_raw_edge"
    if row["mean_raw_truth_minus_wrong"] <= 0:
        return "F_raw_ms2_edge_favors_wrong_or_ties"
    if not row["network_can_rank_truth_first"]:
        return "G_positive_raw_edge_but_network_cannot_rescue"
    return "H_rescue_headroom_exists"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--transitions", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_eval_v1/dependency_corrected_transitions.csv.gz"))
    parser.add_argument("--paths", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_eval_v1/evidence_paths.csv.gz"))
    parser.add_argument("--raw-paths", type=Path, default=Path(
        "data/validation/bioaware_metdna3_raw_ms2_layer_v2.paths.csv.gz"))
    parser.add_argument("--raw-pairs", type=Path, default=Path(
        "data/validation/bioaware_metdna3_raw_ms2_layer_v2.pairs.csv"))
    parser.add_argument("--headroom", type=Path, default=Path(
        "data/validation/bioaware_metdna3_safe_gate_development_v1/error_headroom.csv"))
    parser.add_argument("--participants", type=Path, default=Path(
        "data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"))
    parser.add_argument("--splits", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"))
    parser.add_argument("--external-spectra", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v2/external_spectra.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_metdna3_failure_decomposition_v1"))
    parser.add_argument("--maximum-seed-degree", type=int, default=250)
    args = parser.parse_args()

    inputs = [args.scores, args.transitions, args.paths, args.raw_paths, args.raw_pairs,
              args.headroom, args.participants, args.splits, args.external_spectra]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    scores = pd.read_csv(args.scores)
    transitions = pd.read_csv(args.transitions)
    paths = pd.read_csv(args.paths)
    raw_paths = pd.read_csv(args.raw_paths)
    raw_pairs = pd.read_csv(args.raw_pairs)
    headroom = pd.read_csv(args.headroom)
    participants = pd.read_csv(args.participants)
    splits = pd.read_csv(args.splits)
    external = pd.read_csv(args.external_spectra)

    top1 = official_top1(scores)
    errors = top1[~top1["baseline_correct"]].copy()
    if len(errors) != 22 or errors["query_id"].nunique() != 22:
        raise RuntimeError(f"expected 22 official errors, observed {len(errors)}")
    error_ids = set(errors["query_id"])
    rotation = transitions[transitions["query_id"].isin(error_ids)].copy()
    if rotation.duplicated(["fold", "query_id"]).any():
        raise RuntimeError("transition table has duplicate fold/query rows")
    if len(rotation) != 154:
        raise RuntimeError(f"expected 154 error rotation instances, observed {len(rotation)}")

    level1_identities = set(splits["ik14"].astype(str))
    spectral_identities = set(external["truth_ik14"].astype(str))
    nodes, neighbor_map, eligible_seeds = eligible_opposite_side_neighbors(
        participants, level1_identities, args.maximum_seed_degree
    )

    network_paths = paths[paths["query_id"].isin(error_ids)].copy()
    raw_error_paths = raw_paths[raw_paths["query_id"].isin(error_ids)].copy()
    raw_error_pairs = raw_pairs[raw_pairs["query_id"].isin(error_ids)].copy()
    headroom_error = headroom[headroom["query_id"].isin(error_ids)].copy()

    rotation_rows: list[dict[str, object]] = []
    for base in rotation.itertuples(index=False):
        query_id = str(base.query_id)
        truth = str(base.truth_candidate_id)
        wrong = str(base.baseline_top_candidate)
        net = network_paths[(network_paths["fold"] == base.fold) &
                            (network_paths["query_id"] == query_id)]
        raw = raw_error_paths[(raw_error_paths["fold"] == base.fold) &
                              (raw_error_paths["query_id"] == query_id)]
        pair = raw_error_pairs[(raw_error_pairs["fold"] == base.fold) &
                               (raw_error_pairs["query_id"] == query_id)]
        h = headroom_error[(headroom_error["fold"] == base.fold) &
                           (headroom_error["query_id"] == query_id)]
        truth_net = net[net["query_candidate_id"].astype(str) == truth]
        wrong_net = net[net["query_candidate_id"].astype(str) != truth]
        truth_raw = raw[raw["query_candidate_id"].astype(str) == truth]
        wrong_raw = raw[raw["query_candidate_id"].astype(str) != truth]
        rotation_rows.append({
            "fold": int(base.fold), "query_id": query_id,
            "truth_formula": str(base.truth_formula), "truth_candidate_id": truth,
            "baseline_top_candidate": wrong,
            "truth_network_path": bool(len(truth_net)),
            "wrong_network_path": bool(len(wrong_net)),
            "truth_raw_path": bool(len(truth_raw)),
            "wrong_raw_path": bool(len(wrong_raw)),
            "paired_raw": bool(len(pair)),
            "raw_truth_minus_wrong": float(pair.iloc[0].delta) if len(pair) else np.nan,
            "network_can_rank_truth_first": bool(h["network_can_rank_truth_first"].any()) if len(h) else False,
            "network_corrected": bool(base.corrected),
        })
    rotation_frame = pd.DataFrame(rotation_rows)

    query_rows: list[dict[str, object]] = []
    for error in errors.itertuples(index=False):
        query_id = str(error.query_id)
        truth = str(error.truth_candidate_id)
        wrong = str(error.baseline_top_candidate)
        subset = rotation_frame[rotation_frame["query_id"] == query_id]
        neighbors = neighbor_map.get(truth, set())
        spectral_neighbors = neighbors & spectral_identities
        paired_values = subset["raw_truth_minus_wrong"].dropna().to_numpy(float)
        query_rows.append({
            "query_id": query_id, "truth_formula": str(error.truth_formula),
            "truth_candidate_id": truth, "baseline_top_candidate": wrong,
            "candidate_count": int(error.candidate_count),
            "truth_in_rhea": truth in nodes,
            "baseline_wrong_in_rhea": wrong in nodes,
            "eligible_level1_seed_neighbors": int(len(neighbors)),
            "eligible_spectral_seed_neighbors": int(len(spectral_neighbors)),
            "rotation_instances": int(len(subset)),
            "truth_network_path_rotations": int(subset["truth_network_path"].sum()),
            "wrong_network_path_rotations": int(subset["wrong_network_path"].sum()),
            "truth_raw_path_rotations": int(subset["truth_raw_path"].sum()),
            "wrong_raw_path_rotations": int(subset["wrong_raw_path"].sum()),
            "paired_raw_rotations": int(subset["paired_raw"].sum()),
            "mean_raw_truth_minus_wrong": float(np.mean(paired_values)) if len(paired_values) else np.nan,
            "raw_truth_advantage_rotations": int(np.sum(paired_values > 0)),
            "raw_wrong_or_tie_rotations": int(np.sum(paired_values <= 0)),
            "network_can_rank_truth_first": bool(subset["network_can_rank_truth_first"].any()),
            "network_corrected_in_any_rotation": bool(subset["network_corrected"].any()),
        })
    query_frame = pd.DataFrame(query_rows)
    query_frame["primary_bottleneck"] = query_frame.apply(primary_bottleneck, axis=1)

    bottlenecks = {
        name: {
            "queries": int(len(group)),
            "formulas": int(group["truth_formula"].nunique()),
        }
        for name, group in query_frame.groupby("primary_bottleneck", sort=True)
    }
    report = {
        "status": "bioaware_metdna3_failure_decomposition_complete",
        "formal": True,
        "official_dreams": {
            "queries": int(len(top1)), "errors": int(len(errors)),
            "recall1": float(top1["baseline_correct"].mean()),
            "error_formulas": int(errors["truth_formula"].nunique()),
        },
        "rotation_protocol": {
            "error_rotation_instances": int(len(rotation_frame)),
            "truth_network_path_instances": int(rotation_frame["truth_network_path"].sum()),
            "wrong_network_path_instances": int(rotation_frame["wrong_network_path"].sum()),
            "truth_raw_path_instances": int(rotation_frame["truth_raw_path"].sum()),
            "wrong_raw_path_instances": int(rotation_frame["wrong_raw_path"].sum()),
            "paired_raw_instances": int(rotation_frame["paired_raw"].sum()),
            "network_rescue_headroom_instances": int(rotation_frame["network_can_rank_truth_first"].sum()),
            "network_corrected_instances": int(rotation_frame["network_corrected"].sum()),
        },
        "unique_error_queries": {
            "any_network_path": int(((query_frame["truth_network_path_rotations"] +
                                       query_frame["wrong_network_path_rotations"]) > 0).sum()),
            "truth_network_path": int((query_frame["truth_network_path_rotations"] > 0).sum()),
            "truth_raw_path": int((query_frame["truth_raw_path_rotations"] > 0).sum()),
            "paired_raw_evidence": int((query_frame["paired_raw_rotations"] > 0).sum()),
            "raw_truth_mean_advantage": int((query_frame["mean_raw_truth_minus_wrong"] > 0).sum()),
            "network_rescue_headroom": int(query_frame["network_can_rank_truth_first"].sum()),
            "network_corrected_any_rotation": int(query_frame["network_corrected_in_any_rotation"].sum()),
        },
        "primary_bottlenecks": bottlenecks,
        "coverage": {
            "level1_identity_pool": int(len(level1_identities)),
            "raw_ms2_identity_pool": int(len(spectral_identities)),
            "rhea_nodes": int(len(nodes)),
            "eligible_level1_seed_nodes": int(len(eligible_seeds)),
        },
        "contracts": {
            "development_only": True,
            "unique_queries_and_rotations_separated": True,
            "graph_topology_test_is_outcome_independent": True,
            "P2b": "forbidden",
            "RP_opened": False,
        },
        "provenance": {path.name: sha256(path) for path in inputs},
        "claim_limit": (
            "Failure decomposition on consumed HILIC development only. Graph coverage and raw-MS2 "
            "direction are diagnostic; they do not establish deployable annotation gain."
        ),
    }
    query_frame.sort_values(["primary_bottleneck", "truth_formula", "query_id"]).to_csv(
        output / "per_error_query.csv", index=False)
    rotation_frame.sort_values(["query_id", "fold"]).to_csv(
        output / "per_error_rotation.csv.gz", index=False, compression="gzip")
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
