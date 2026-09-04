#!/usr/bin/env python
"""Validate known-reaction candidate paths with raw MS2 at every feature edge.

Primary analysis is the known MetDNA2 step-0 network at depth <=2.  Depth 3 is
reported as an ablation.  Candidate paths are built without truth labels;
truth is consulted only after all path scores have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from metdna3_similarity import metdna3_reverse_dot
except ModuleNotFoundError:
    from tasks.metdna3_similarity import metdna3_reverse_dot

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


def enumerate_shortest_paths(
    candidate: str,
    adjacency: dict[str, set[str]],
    seeds: set[str],
    observed: set[str],
    maximum_depth: int,
    maximum_paths: int = 500,
) -> tuple[list[tuple[str, ...]], bool]:
    """Enumerate outcome-blind shortest paths; non-seed intermediates must be observed."""
    if candidate not in observed:
        return [], False
    frontier = [(candidate,)]
    for _depth in range(1, maximum_depth + 1):
        reached: list[tuple[str, ...]] = []
        next_frontier: list[tuple[str, ...]] = []
        for path in frontier:
            for target in sorted(adjacency.get(path[-1], set())):
                if target in path:
                    continue
                extended = path + (target,)
                if target in seeds:
                    reached.append(extended)
                elif target in observed:
                    next_frontier.append(extended)
        if reached:
            reached = sorted(set(reached))
            return reached[:maximum_paths], len(reached) > maximum_paths
        frontier = next_frontier
        if not frontier:
            break
    return [], False


def best_bottleneck_for_identity_path(
    identity_path: tuple[str, ...],
    query_tensor: np.ndarray,
    node_options: dict[str, list[int]],
    node_tensors: dict[int, np.ndarray],
    edge_cache: dict[tuple[int, int], float],
) -> tuple[float | None, int]:
    """Maximise the weakest raw-MS2 edge while retaining the combination count."""
    states: dict[int, float] = {-1: 1.0}  # -1 denotes the exact query MS2 spectrum
    combinations = 0
    for identity in identity_path[1:]:
        targets = node_options.get(identity, [])
        if not targets:
            return None, combinations
        updated: dict[int, float] = {}
        for source, previous in states.items():
            left = query_tensor if source == -1 else node_tensors[source]
            for target in targets:
                if source == target:
                    continue
                if source == -1:
                    # The exact query spectrum differs across queries and must
                    # never share a cache key merely because its sentinel is -1.
                    edge_score = metdna3_reverse_dot(left, node_tensors[target])
                else:
                    key = tuple(sorted((source, target)))
                    if key not in edge_cache:
                        edge_cache[key] = metdna3_reverse_dot(left, node_tensors[target])
                    edge_score = edge_cache[key]
                score = min(previous, edge_score)
                updated[target] = max(updated.get(target, -np.inf), score)
                combinations += 1
        states = updated
        if not states:
            return None, combinations
    return (max(states.values()) if states else None), combinations


def bootstrap_delta(values: np.ndarray, formulas: np.ndarray, seed: int) -> dict:
    unique = np.unique(formulas.astype(str))
    grouped = {item: values[formulas.astype(str) == item] for item in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(5000, float)
    for index in range(len(boot)):
        sampled = rng.choice(unique, len(unique), replace=True)
        boot[index] = np.mean(np.concatenate([grouped[item] for item in sampled]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "formulas": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recursive-dir", type=Path, default=Path("data/validation/bioaware_metdna3_recursive_headroom_v1"))
    parser.add_argument("--network-dir", type=Path, default=Path("data/reference/metdna2_emrn_network_20260828"))
    parser.add_argument("--development-dir", type=Path, default=Path("data/validation/bioaware_metdna3_development_v1"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v2"))
    parser.add_argument("--candidate-scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--baseline-transitions", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"))
    parser.add_argument("--feature-ms2-dir", type=Path, default=Path("data/validation/bioaware_metdna3_feature_ms2_cache_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_ms2_v2"))
    parser.add_argument("--maximum-paths", type=int, default=500)
    parser.add_argument(
        "--maximum-network-step", type=int, choices=(0, 1), default=0,
        help="0=reported reaction pairs only; 1=include the frozen predicted eMRN layer",
    )
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()
    files = {
        "nodes": args.recursive_dir / "stable_ms1_feature_nodes.csv.gz",
        "assignments": args.recursive_dir / "feature_candidate_assignments.csv.gz",
        "edges": args.network_dir / "metdna2_emrn_edges.csv.gz",
        "truth": args.development_dir / args.truth_name,
        "splits": args.development_dir / "identity_splits.csv.gz",
        "queries": args.query_cache / "queries.csv.gz",
        "query_tensors": args.query_cache / "query_tensors.npz",
        "candidate_scores": args.candidate_scores,
        "baseline": args.baseline_transitions,
        "feature_ms2": args.feature_ms2_dir / "feature_ms2.csv.gz",
        "feature_tensors": args.feature_ms2_dir / "feature_ms2_tensors.npz",
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    nodes = pd.read_csv(files["nodes"])
    assignments = pd.read_csv(files["assignments"])
    assignments = assignments[assignments["maximum_step"].le(args.maximum_network_step)]
    observed = set(assignments["candidate_ik14"].astype(str))
    identity_nodes_all = assignments.groupby("candidate_ik14")["feature_node"].agg(
        lambda values: sorted(set(map(int, values)))
    ).to_dict()
    node_polarity = nodes.set_index("feature_node")["polarity"].astype(str).to_dict()
    edges = pd.read_csv(files["edges"])
    edges = edges[edges["minimum_step"].le(args.maximum_network_step)]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        left, right = str(row.ik14_a), str(row.ik14_b)
        adjacency[left].add(right)
        adjacency[right].add(left)

    feature_meta = pd.read_csv(files["feature_ms2"])
    feature_array = np.load(files["feature_tensors"], allow_pickle=False)["feature_ms2_tensor"]
    if len(feature_meta) != len(feature_array):
        raise RuntimeError("feature MS2 metadata/tensor mismatch")
    node_tensors = {
        int(row.feature_node): feature_array[position]
        for position, row in enumerate(feature_meta.itertuples(index=False))
    }
    truth = pd.read_csv(files["truth"])
    truth["feature_node"] = [
        nearest_feature(nodes, str(row.polarity), float(row.mz), float(row.rt), args.ppm, args.rt_sec)
        for row in truth.itertuples(index=False)
    ]
    seed_nodes: dict[str, list[int]] = defaultdict(list)
    for row in truth.dropna(subset=["feature_node"]).itertuples(index=False):
        node = int(row.feature_node)
        if node in node_tensors:
            seed_nodes[str(row.ik14)].append(node)
    seed_nodes = {key: sorted(set(value)) for key, value in seed_nodes.items()}
    node_options_by_polarity: dict[str, dict[str, list[int]]] = {}
    for polarity in ("positive", "negative"):
        node_options_by_polarity[polarity] = {
            identity: [
                node for node in node_ids
                if node in node_tensors and node_polarity[node] == polarity
            ]
            for identity, node_ids in identity_nodes_all.items()
        }
        node_options_by_polarity[polarity] = {
            identity: node_ids
            for identity, node_ids in node_options_by_polarity[polarity].items()
            if node_ids
        }

    queries = pd.read_csv(files["queries"])
    query_tensors = np.load(files["query_tensors"], allow_pickle=False)["query_tensor"]
    if len(queries) != len(query_tensors):
        raise RuntimeError("query metadata/tensor mismatch")
    query_tensor = dict(zip(queries["query_id"].astype(str), query_tensors, strict=True))
    query_polarity = queries.set_index("query_id")["polarity"].astype(str).to_dict()
    candidates = pd.read_csv(files["candidate_scores"])
    splits = pd.read_csv(files["splits"])
    baseline = pd.read_csv(files["baseline"]).groupby("query_id")["baseline_top_candidate"].first().astype(str)

    records: list[dict] = []
    truncated_paths = 0
    edge_cache: dict[tuple[int, int], float] = {}
    for fold in range(10):
        seeds = set(splits[(splits.fold.eq(fold)) & splits.role.eq("seed")].ik14.astype(str))
        seeds &= set(seed_nodes)
        heldout = set(splits[(splits.fold.eq(fold)) & splits.role.eq("heldout")].ik14.astype(str))
        subset = candidates[candidates["truth_candidate_id"].astype(str).isin(heldout)]
        fold_node_options: dict[str, dict[str, list[int]]] = {}
        for polarity in ("positive", "negative"):
            local = dict(node_options_by_polarity[polarity])
            for identity in seeds:
                eligible = [
                    node for node in seed_nodes.get(identity, [])
                    if node_polarity[node] == polarity
                ]
                if eligible:
                    local[identity] = eligible
            fold_node_options[polarity] = local
        for row in subset.itertuples(index=False):
            query_id = str(row.query_id)
            polarity = query_polarity[query_id]
            candidate = str(row.candidate_id)
            local_nodes = fold_node_options[polarity]
            for maximum_depth in (2, 3):
                paths, truncated = enumerate_shortest_paths(
                    candidate, adjacency, seeds, observed, maximum_depth, args.maximum_paths
                )
                truncated_paths += int(truncated)
                scores: list[float] = []
                combinations = 0
                complete_paths = 0
                for path in paths:
                    score, count = best_bottleneck_for_identity_path(
                        path, query_tensor[query_id], local_nodes, node_tensors, edge_cache
                    )
                    combinations += count
                    if score is not None:
                        scores.append(float(score))
                        complete_paths += 1
                records.append({
                    "fold": fold,
                    "query_id": query_id,
                    "candidate_id": candidate,
                    "truth_candidate_id": str(row.truth_candidate_id),
                    "truth_formula": str(row.truth_formula),
                    "maximum_depth": maximum_depth,
                    "identity_paths": len(paths),
                    "complete_ms2_paths": complete_paths,
                    "node_combinations_evaluated": combinations,
                    "path_truncated": truncated,
                    "best_bottleneck": max(scores) if scores else np.nan,
                    "median_bottleneck": float(np.median(scores)) if scores else np.nan,
                })
    frame = pd.DataFrame(records)
    evidence_path = output / "candidate_edge_evidence.csv.gz"
    frame.to_csv(evidence_path, index=False, compression="gzip")

    paired_rows: list[dict] = []
    for (depth, fold, query_id), group in frame.groupby(["maximum_depth", "fold", "query_id"]):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        baseline_id = str(baseline.loc[query_id])
        truth_row = group[group.candidate_id.eq(truth_id)]
        baseline_row = group[group.candidate_id.eq(baseline_id)]
        wrong_rows = group[group.candidate_id.ne(truth_id)]
        if len(truth_row) != 1 or len(baseline_row) != 1:
            raise RuntimeError(f"candidate pairing mismatch for {query_id}")
        truth_score = float(truth_row.iloc[0].best_bottleneck)
        baseline_score = float(baseline_row.iloc[0].best_bottleneck)
        finite_wrong = wrong_rows.loc[np.isfinite(wrong_rows.best_bottleneck), "best_bottleneck"]
        strongest_wrong_score = float(finite_wrong.max()) if len(finite_wrong) else np.nan
        baseline_comparable = np.isfinite(truth_score) and np.isfinite(baseline_score)
        strongest_wrong_comparable = np.isfinite(truth_score) and np.isfinite(strongest_wrong_score)
        paired_rows.append({
            "maximum_depth": int(depth), "fold": int(fold), "query_id": query_id,
            "truth_formula": str(group.truth_formula.iloc[0]),
            "baseline_correct": baseline_id == truth_id,
            "truth_score": truth_score, "baseline_score": baseline_score,
            "strongest_wrong_score": strongest_wrong_score,
            "baseline_comparable": baseline_comparable,
            "strongest_wrong_comparable": strongest_wrong_comparable,
            "delta_vs_baseline": truth_score - baseline_score if baseline_comparable else np.nan,
            "delta_vs_strongest_wrong": (
                truth_score - strongest_wrong_score if strongest_wrong_comparable else np.nan
            ),
        })
    paired = pd.DataFrame(paired_rows)
    paired_path = output / "paired_truth_vs_baseline.csv.gz"
    paired.to_csv(paired_path, index=False, compression="gzip")
    summaries: dict[str, dict] = {}
    for depth, group in paired.groupby("maximum_depth"):
        errors = group[(~group.baseline_correct) & group.baseline_comparable]
        all_hard = group[group.strongest_wrong_comparable]
        correct = all_hard[all_hard.baseline_correct]
        summaries[f"depth{int(depth)}"] = {
            "query_rotations": int(len(group)),
            "strongest_wrong_comparable_rotations": int(len(all_hard)),
            "error_baseline_comparable": int(len(errors)),
            "error_truth_gt_official_wrong": int((errors.delta_vs_baseline > 0).sum()),
            "error_truth_le_official_wrong": int((errors.delta_vs_baseline <= 0).sum()),
            "correct_strongest_wrong_comparable": int(len(correct)),
            "correct_risk_strongest_wrong_gt_truth": int((correct.delta_vs_strongest_wrong < 0).sum()),
            "correct_safe_truth_ge_strongest_wrong": int((correct.delta_vs_strongest_wrong >= 0).sum()),
            "all_truth_gt_strongest_wrong": int((all_hard.delta_vs_strongest_wrong > 0).sum()),
            "error_formula_cluster_delta": (
                bootstrap_delta(
                    errors.delta_vs_baseline.to_numpy(float),
                    errors.truth_formula.to_numpy(str), 20260828 + int(depth),
                )
                if len(errors) else None
            ),
        }
    report = {
        "status": "bioaware_metdna3_candidate_edge_ms2_audit_complete",
        "formal": True,
        "scope": args.scope,
        "protocol": (
            f"eMRN step<={args.maximum_network_step} shortest identity paths; stable observed MS1 "
            "nodes; exact query MS2; deterministic representative intermediate MS2; "
            "MetDNA3 reverse score at every edge"
        ),
        "maximum_network_step": int(args.maximum_network_step),
        "candidate_rows": int(len(frame)),
        "truncated_candidate_depth_rows": int(truncated_paths),
        "unique_feature_edges_scored": int(len(edge_cache)),
        "results": summaries,
        "primary": "depth2",
        "depth3": "ablation only",
        "contracts": {
            "truth_used_for_path_construction": False,
            "outcomes_used_for_path_construction": False,
            "predicted_step1_used": args.maximum_network_step == 1,
            "P2b_used": False,
            "external_test_opened": False,
            "threshold_selected": False,
        },
        "provenance": {key: sha256(path) for key, path in files.items()} | {
            "candidate_edge_evidence_sha256": sha256(evidence_path),
            "paired_sha256": sha256(paired_path),
        },
        "claim_limit": "Development-set edge-specificity audit only; no annotation-gain claim and no threshold frozen.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
