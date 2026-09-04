#!/usr/bin/env python
"""Audit public MetDNA2 KEGG edges as an incremental BioAware extension.

No score threshold or fusion weight is selected. The script measures topology
coverage and the direction of the current MetDNA3 raw-MS2 edge on consumed
HILIC development rotations.
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

from metdna3_similarity import metdna3_reverse_dot


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


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict:
    unique = np.unique(clusters.astype(str))
    if not len(unique):
        return {"mean": None, "ci_low": None, "ci_high": None, "clusters": 0}
    grouped = {value: values[clusters.astype(str) == value] for value in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(5000, dtype=float)
    for repeat in range(len(boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boot[repeat] = np.mean(np.concatenate([grouped[value] for value in sampled]))
    return {
        "mean": float(np.mean(values)), "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)), "clusters": int(len(unique)),
    }


def official_top(scores: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for query_id, group in scores.groupby("query_id", sort=False):
        maximum = float(group["spectral_score"].max())
        tied = group[np.isclose(group["spectral_score"], maximum, rtol=0, atol=1e-12)]
        result[str(query_id)] = str(tied.sort_values("candidate_id").iloc[0].candidate_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--splits", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"))
    parser.add_argument("--edges", type=Path, default=Path(
        "data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz"))
    parser.add_argument("--cache-dir", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v2"))
    parser.add_argument("--rhea-decomposition", type=Path, default=Path(
        "data/validation/bioaware_metdna3_failure_decomposition_v1/per_error_query.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_metdna3_kegg_extension_v1"))
    args = parser.parse_args()
    manifest_path = args.cache_dir / "external_spectra.csv.gz"
    tensor_path = args.cache_dir / "external_tensors.npz"
    query_path = args.cache_dir / "queries.csv.gz"
    inputs = [args.scores, args.splits, args.edges, manifest_path, tensor_path,
              query_path, args.rhea_decomposition]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    scores = pd.read_csv(args.scores)
    splits = pd.read_csv(args.splits)
    edges = pd.read_csv(args.edges)
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    tensors = np.load(tensor_path, allow_pickle=False)["external_tensor"]
    queries = pd.read_csv(query_path).set_index("query_id")
    rhea_errors = pd.read_csv(args.rhea_decomposition)
    if len(manifest) != len(tensors):
        raise RuntimeError("external spectrum manifest/tensor mismatch")

    adjacency: dict[str, set[str]] = {}
    for edge in edges.itertuples(index=False):
        adjacency.setdefault(str(edge.ik14_a), set()).add(str(edge.ik14_b))
        adjacency.setdefault(str(edge.ik14_b), set()).add(str(edge.ik14_a))
    positions_by_identity = {
        str(identity): group.index.to_numpy(int)
        for identity, group in manifest.groupby("truth_ik14", sort=False)
    }
    spectrum_position = {str(key): position for position, key in enumerate(manifest.spectrum_key)}
    truth_by_query = scores.groupby("query_id").truth_candidate_id.first().astype(str).to_dict()
    formula_by_query = scores.groupby("query_id").truth_formula.first().astype(str).to_dict()
    baseline_by_query = official_top(scores)
    similarity_cache: dict[tuple[str, str], float] = {}

    path_rows: list[dict[str, object]] = []
    for fold in range(10):
        seeds = set(splits[(splits["fold"] == fold) & (splits["role"] == "seed")]["ik14"].astype(str))
        heldout = set(splits[(splits["fold"] == fold) & (splits["role"] == "heldout")]["ik14"].astype(str))
        fold_scores = scores[scores["truth_candidate_id"].astype(str).isin(heldout)]
        for query_id, group in fold_scores.groupby("query_id", sort=False):
            query_id = str(query_id)
            query_position = spectrum_position[str(queries.loc[query_id, "spectrum_key"])]
            for candidate in group["candidate_id"].astype(str):
                for seed_identity in sorted(adjacency.get(candidate, set()) & seeds):
                    seed_positions = positions_by_identity.get(seed_identity, np.empty(0, dtype=int))
                    if not len(seed_positions):
                        continue
                    key = (query_id, seed_identity)
                    if key not in similarity_cache:
                        similarity_cache[key] = float(max(
                            metdna3_reverse_dot(tensors[query_position], tensors[position])
                            for position in seed_positions
                        ))
                    path_rows.append({
                        "fold": fold, "query_id": query_id, "query_candidate_id": candidate,
                        "seed_compound_id": seed_identity,
                        "raw_ms2_similarity": similarity_cache[key],
                        "seed_spectra": int(len(seed_positions)),
                        "truth_path": candidate == truth_by_query[query_id],
                    })
    paths = pd.DataFrame(path_rows)
    if paths.empty:
        raise RuntimeError("no evaluable KEGG raw-MS2 paths")
    candidate = (
        paths.groupby(["fold", "query_id", "query_candidate_id"], sort=False)
        ["raw_ms2_similarity"].max().reset_index()
    )
    pair_rows: list[dict[str, object]] = []
    for (fold, query_id), group in candidate.groupby(["fold", "query_id"], sort=False):
        truth = truth_by_query[query_id]
        truth_score = group.loc[group["query_candidate_id"] == truth, "raw_ms2_similarity"]
        wrong_score = group.loc[group["query_candidate_id"] != truth, "raw_ms2_similarity"]
        if not len(truth_score) or not len(wrong_score):
            continue
        pair_rows.append({
            "fold": int(fold), "query_id": query_id,
            "truth_formula": formula_by_query[query_id],
            "baseline_wrong": baseline_by_query[query_id] != truth,
            "truth_max_similarity": float(truth_score.max()),
            "wrong_max_similarity": float(wrong_score.max()),
            "delta": float(truth_score.max() - wrong_score.max()),
        })
    pairs = pd.DataFrame(pair_rows)
    error_pairs = pairs[pairs["baseline_wrong"]]

    error_ids = set(rhea_errors["query_id"].astype(str))
    kegg_truth_error_ids = set(paths[
        paths["query_id"].isin(error_ids) & paths["truth_path"]
    ]["query_id"].astype(str))
    rhea_truth_error_ids = set(rhea_errors.loc[
        rhea_errors["truth_network_path_rotations"] > 0, "query_id"
    ].astype(str))
    newly_covered = sorted(kegg_truth_error_ids - rhea_truth_error_ids)
    report = {
        "status": "bioaware_metdna3_kegg_extension_audit_complete",
        "formal": True,
        "protocol": "public MetDNA2 KEGG version1 edges; current MetDNA3 scoreReverse raw-MS2 edge; 10 frozen identity rotations",
        "topology_and_raw_coverage": {
            "path_rows": int(len(paths)), "truth_path_rows": int(paths["truth_path"].sum()),
            "wrong_path_rows": int((~paths["truth_path"]).sum()),
            "unique_queries_with_any_path": int(paths["query_id"].nunique()),
            "official_error_queries_with_truth_path": int(len(kegg_truth_error_ids)),
            "official_error_queries_new_beyond_rhea": int(len(newly_covered)),
            "newly_covered_error_query_ids": newly_covered,
            "rhea_or_kegg_error_truth_path_coverage": int(len(rhea_truth_error_ids | kegg_truth_error_ids)),
        },
        "paired_raw_ms2": {
            "instances": int(len(pairs)),
            "truth_gt_wrong_fraction": float((pairs["delta"] > 0).mean()),
            "formula_cluster_mean_delta": cluster_bootstrap(
                pairs["delta"].to_numpy(float), pairs["truth_formula"].to_numpy(str), 20260828),
            "baseline_error_instances": int(len(error_pairs)),
            "error_truth_gt_wrong_fraction": float((error_pairs["delta"] > 0).mean()) if len(error_pairs) else None,
            "error_formula_cluster_mean_delta": cluster_bootstrap(
                error_pairs["delta"].to_numpy(float),
                error_pairs["truth_formula"].to_numpy(str), 20260829),
        },
        "contracts": {
            "development_only": True, "threshold_or_weight_selected": False,
            "version2_predicted_edges_used": False, "P2b": "forbidden", "RP_opened": False,
        },
        "provenance": {path.name: sha256(path) for path in inputs},
        "claim_limit": "Incremental topology and edge-direction audit only; no deployable BioAware gain.",
    }
    paths.to_csv(output / "paths.csv.gz", index=False, compression="gzip")
    pairs.to_csv(output / "paired_candidates.csv", index=False)
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
