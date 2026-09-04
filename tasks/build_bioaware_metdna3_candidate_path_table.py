#!/usr/bin/env python
"""Build candidate-specific known-reaction paths for the MetDNA3 benchmark.

This is a non-parametric evidence audit.  It never converts truth-aware
headroom into a deployable score and never opens the RP benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from audit_bioaware_metdna3_recursive_headroom import nearest_feature
except ModuleNotFoundError:
    from tasks.audit_bioaware_metdna3_recursive_headroom import nearest_feature


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def shortest_seed_evidence(
    candidate: str,
    adjacency: dict[str, set[str]],
    seeds: set[str],
    observed: set[str],
    maximum_depth: int,
) -> dict:
    """Return candidate-specific shortest-path evidence without path explosion."""
    if candidate not in observed:
        return {
            "path_available": False, "minimum_depth": None, "shortest_seed_count": 0,
            "frontier_count": 0, "candidate_degree": len(adjacency.get(candidate, set())),
        }
    visited = {candidate}
    frontier = {candidate}
    for depth in range(1, maximum_depth + 1):
        next_frontier: set[str] = set()
        reached_seeds: set[str] = set()
        for source in frontier:
            for target in adjacency.get(source, set()):
                if target in seeds:
                    reached_seeds.add(target)
                elif target in observed and target not in visited:
                    next_frontier.add(target)
        if reached_seeds:
            return {
                "path_available": True,
                "minimum_depth": depth,
                "shortest_seed_count": len(reached_seeds),
                "frontier_count": len(frontier),
                "candidate_degree": len(adjacency.get(candidate, set())),
            }
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return {
        "path_available": False, "minimum_depth": None, "shortest_seed_count": 0,
        "frontier_count": 0, "candidate_degree": len(adjacency.get(candidate, set())),
    }


def evidence_key(row: pd.Series) -> tuple:
    """Truth-free lexicographic evidence key; higher is stronger."""
    if not bool(row["path_available"]):
        return (0, 0, 0, 0)
    return (
        1,
        -int(row["minimum_depth"]),
        int(row["shortest_seed_count"]),
        -int(row["candidate_degree"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recursive-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_recursive_headroom_v1"),
    )
    parser.add_argument(
        "--network-dir", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828"),
    )
    parser.add_argument(
        "--development-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1"),
    )
    parser.add_argument(
        "--query-cache", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"),
    )
    parser.add_argument(
        "--candidate-scores", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"),
    )
    parser.add_argument(
        "--baseline-transitions", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_candidate_paths_v1"),
    )
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--maximum-depth", type=int, default=3)
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()
    paths = {
        "recursive_report": args.recursive_dir / "report.json",
        "nodes": args.recursive_dir / "stable_ms1_feature_nodes.csv.gz",
        "assignments": args.recursive_dir / "feature_candidate_assignments.csv.gz",
        "edges": args.network_dir / "metdna2_emrn_edges.csv.gz",
        "truth": args.development_dir / args.truth_name,
        "splits": args.development_dir / "identity_splits.csv.gz",
        "queries": args.query_cache,
        "candidate_scores": args.candidate_scores,
        "baseline_transitions": args.baseline_transitions,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")
    recursive_report = json.loads(paths["recursive_report"].read_text(encoding="utf-8"))
    if not recursive_report.get("formal"):
        raise RuntimeError("candidate paths require a formal recursive feature graph")
    if recursive_report.get("contracts", {}).get("feature_selection_outcome_blind") is not True:
        raise RuntimeError("recursive graph does not prove outcome-blind threshold selection")
    selected_noise_threshold = float(recursive_report["selected_noise_threshold"])

    nodes = pd.read_csv(paths["nodes"])
    assignments = pd.read_csv(paths["assignments"])
    assignments = assignments[assignments["maximum_step"].eq(0)]
    feature_candidates = assignments.groupby("feature_node")["candidate_ik14"].agg(set).to_dict()
    observed = set(assignments["candidate_ik14"].astype(str))
    edges = pd.read_csv(paths["edges"])
    edges = edges[edges["minimum_step"].eq(0)]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        left, right = str(row.ik14_a), str(row.ik14_b)
        if left in observed or right in observed:
            adjacency[left].add(right)
            adjacency[right].add(left)

    truth = pd.read_csv(paths["truth"])
    truth["feature_node"] = [
        nearest_feature(nodes, str(row.polarity), float(row.mz), float(row.rt), args.ppm, args.rt_sec)
        for row in truth.itertuples(index=False)
    ]
    truth["feature_recovered"] = truth["feature_node"].notna()
    splits = pd.read_csv(paths["splits"])
    queries = pd.read_csv(paths["queries"])
    queries["feature_node"] = [
        nearest_feature(
            nodes, str(row.polarity), float(row.feature_mz), float(row.feature_rt_sec),
            args.ppm, args.rt_sec,
        )
        for row in queries.itertuples(index=False)
    ]
    candidates = pd.read_csv(paths["candidate_scores"])
    expected_queries = int(queries.query_id.nunique())
    if candidates["query_id"].nunique() != expected_queries:
        raise RuntimeError("frozen DreaMS candidate query coverage changed")
    query_meta = queries.set_index("query_id")
    baseline_frame = pd.read_csv(paths["baseline_transitions"])
    baseline_by_query = baseline_frame.groupby("query_id")["baseline_top_candidate"].first()
    if len(baseline_by_query) != expected_queries:
        raise RuntimeError("frozen baseline transition table changed")

    records: list[dict] = []
    for fold in range(10):
        seed_ids = set(splits[(splits["fold"].eq(fold)) & splits["role"].eq("seed")]["ik14"])
        recovered_seeds = set(truth.loc[truth["feature_recovered"], "ik14"]) & seed_ids
        heldout = set(splits[(splits["fold"].eq(fold)) & splits["role"].eq("heldout")]["ik14"])
        fold_candidates = candidates[candidates["truth_candidate_id"].isin(heldout)]
        for row in fold_candidates.itertuples(index=False):
            meta = query_meta.loc[str(row.query_id)]
            node = None if pd.isna(meta.feature_node) else int(meta.feature_node)
            mass_candidates = set() if node is None else feature_candidates.get(node, set())
            candidate = str(row.candidate_id)
            evidence = shortest_seed_evidence(
                candidate, adjacency, recovered_seeds, observed, args.maximum_depth
            ) if candidate in mass_candidates else {
                "path_available": False, "minimum_depth": None, "shortest_seed_count": 0,
                "frontier_count": 0, "candidate_degree": len(adjacency.get(candidate, set())),
            }
            records.append({
                "fold": fold,
                "query_id": str(row.query_id),
                "candidate_id": candidate,
                "truth_candidate_id": str(row.truth_candidate_id),
                "truth_formula": str(row.truth_formula),
                "spectral_score": float(row.spectral_score),
                "feature_recovered": node is not None,
                "mass_candidate": candidate in mass_candidates,
                **evidence,
            })
    path_table = pd.DataFrame(records)
    expected_rotation_rows = 0
    for query_id, group in candidates.groupby("query_id", sort=False):
        truth_id = str(group.truth_candidate_id.iloc[0])
        rotations = splits[(splits.ik14.astype(str) == truth_id) & splits.role.eq("heldout")].fold.nunique()
        expected_rotation_rows += int(rotations) * len(group)
    if len(path_table) != expected_rotation_rows:
        raise RuntimeError(f"expected {expected_rotation_rows} candidate rotations, got {len(path_table)}")
    path_table_path = output / "candidate_paths.csv.gz"
    path_table.to_csv(path_table_path, index=False, compression="gzip")

    transition_rows: list[dict] = []
    for (fold, query_id), group in path_table.groupby(["fold", "query_id"], sort=True):
        baseline_id = str(baseline_by_query.loc[query_id])
        baseline_rows = group[group["candidate_id"].eq(baseline_id)]
        if len(baseline_rows) != 1:
            raise RuntimeError(f"query {query_id} lacks its frozen baseline candidate")
        baseline = baseline_rows.iloc[0]
        truth = group[group["candidate_id"].eq(group["truth_candidate_id"].iloc[0])]
        if len(truth) != 1:
            raise RuntimeError(f"query {query_id} has {len(truth)} truth candidates")
        truth = truth.iloc[0]
        baseline_key = evidence_key(baseline)
        truth_key = evidence_key(truth)
        best_network_key = max(evidence_key(row) for _, row in group.iterrows())
        best_network = group[[evidence_key(row) == best_network_key for _, row in group.iterrows()]]
        network_unique_top = len(best_network) == 1
        final = best_network.iloc[0] if network_unique_top else baseline
        baseline_correct = str(baseline.candidate_id) == str(truth.candidate_id)
        final_correct = str(final.candidate_id) == str(truth.candidate_id)
        transition_rows.append({
            "fold": int(fold), "query_id": query_id,
            "truth_candidate_id": str(truth.candidate_id),
            "baseline_top_candidate": str(baseline.candidate_id),
            "network_top_candidate": str(final.candidate_id),
            "baseline_correct": baseline_correct,
            "network_correct": final_correct,
            "corrected": (not baseline_correct) and final_correct,
            "introduced": baseline_correct and (not final_correct),
            "truth_strict_advantage_over_baseline": truth_key > baseline_key,
            "any_wrong_strict_advantage_over_truth": any(
                evidence_key(row) > truth_key
                for _, row in group[group["candidate_id"].ne(truth.candidate_id)].iterrows()
            ),
            "network_unique_top": network_unique_top,
        })
    transitions = pd.DataFrame(transition_rows)
    transition_path = output / "transitions.csv.gz"
    transitions.to_csv(transition_path, index=False, compression="gzip")
    errors = transitions[~transitions["baseline_correct"]]
    correct = transitions[transitions["baseline_correct"]]
    report = {
        "status": "bioaware_metdna3_candidate_path_table_complete",
        "formal": True,
        "scope": args.scope,
        "protocol": "known MetDNA2 step0 reaction pairs; observed MS1 nodes; candidate-specific shortest paths",
        "selected_noise_threshold": selected_noise_threshold,
        "candidate_rows": int(len(path_table)),
        "query_rotations": int(len(transitions)),
        "queries": int(transitions["query_id"].nunique()),
        "candidate_path_coverage": int(path_table["path_available"].sum()),
        "headroom": {
            "official_error_rotations": int(len(errors)),
            "truth_strict_advantage": int(errors["truth_strict_advantage_over_baseline"].sum()),
            "unique_error_queries_with_truth_strict_advantage": int(
                errors.loc[errors["truth_strict_advantage_over_baseline"], "query_id"].nunique()
            ),
            "correct_query_rotations": int(len(correct)),
            "correct_queries_with_stronger_wrong_path": int(
                correct["any_wrong_strict_advantage_over_truth"].sum()
            ),
        },
        "naive_lexicographic_transition": {
            "corrected": int(transitions["corrected"].sum()),
            "introduced": int(transitions["introduced"].sum()),
            "net": int(transitions["corrected"].sum() - transitions["introduced"].sum()),
            "claim": "diagnostic only; not a frozen scorer",
        },
        "provenance": {key: sha256(path) for key, path in paths.items()} | {
            "candidate_paths_sha256": sha256(path_table_path),
            "transitions_sha256": sha256(transition_path),
        },
        "contracts": {
            "predicted_step1_used": False,
            "truth_identity_used_for_path_construction": False,
            "candidate_identity_is_deployment_input": True,
            "truth_used_for_headroom_evaluation_only": True,
            "P2b_used": False,
            "external_test_opened": False,
        },
        "claim_limit": "Candidate-path audit only. Raw-MS2 edge validation and formula-OOF gating remain required.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
