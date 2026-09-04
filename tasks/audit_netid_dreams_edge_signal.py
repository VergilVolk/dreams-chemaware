#!/usr/bin/env python
"""Test DreaMS edge signal inside the frozen public NetID mouse-liver graph.

This is a component-isolated *feasibility* audit.  Author-selected graph edges
are compared with mass/RT/degree-matched nonedges.  The NetID assignments are
not treated as independent molecular truth and no annotation accuracy claim is
made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate_feature_embeddings(
    embeddings: np.ndarray,
    feature_ids: np.ndarray,
    precursor_mz: np.ndarray,
    rt: np.ndarray,
    peak_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(feature_ids.astype(np.int64))
    vectors: list[np.ndarray] = []
    mz_values: list[float] = []
    rt_values: list[float] = []
    count_values: list[int] = []
    for feature in unique:
        mask = feature_ids == feature
        vector = embeddings[mask].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid feature centroid for {feature}")
        vectors.append((vector / norm).astype(np.float32))
        mz_values.append(float(np.median(precursor_mz[mask])))
        rt_values.append(float(np.median(rt[mask])))
        count_values.append(int(np.max(peak_counts[mask])))
    return (
        unique,
        np.stack(vectors),
        np.asarray(mz_values),
        np.asarray(rt_values),
        np.asarray(count_values),
    )


class UnionFind:
    def __init__(self, values: np.ndarray):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_matched_decoys(
    feature_ids: np.ndarray,
    precursor_mz: np.ndarray,
    rt: np.ndarray,
    peak_counts: np.ndarray,
    degree: np.ndarray,
    positive_pairs: np.ndarray,
    controls_per_edge: int,
) -> np.ndarray:
    """Return feature-index pairs matched without using DreaMS similarity."""

    n = len(feature_ids)
    left, right = np.triu_indices(n, k=1)
    id_to_index = {int(value): index for index, value in enumerate(feature_ids)}
    positive_indices = np.asarray(
        [[id_to_index[int(a)], id_to_index[int(b)]] for a, b in positive_pairs],
        dtype=np.int64,
    )
    graph_set = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in positive_indices
    }
    allowed = np.asarray(
        [(int(a), int(b)) not in graph_set for a, b in zip(left, right, strict=True)],
        dtype=bool,
    )
    candidates = np.stack([left[allowed], right[allowed]], axis=1)
    if len(candidates) < controls_per_edge:
        raise RuntimeError("too few candidate nonedges")

    def descriptors(pairs: np.ndarray) -> np.ndarray:
        a, b = pairs[:, 0], pairs[:, 1]
        return np.stack(
            [
                np.log1p(np.abs(precursor_mz[a] - precursor_mz[b])),
                np.log1p(np.abs(rt[a] - rt[b])),
                np.log1p(degree[a] + degree[b]),
                np.log1p(np.minimum(peak_counts[a], peak_counts[b])),
            ],
            axis=1,
        )

    candidate_x = descriptors(candidates)
    positive_x = descriptors(positive_indices)
    median = np.median(candidate_x, axis=0)
    scale = np.subtract(*np.percentile(candidate_x, [75, 25], axis=0))
    scale[scale <= 1e-12] = 1.0
    candidate_x = (candidate_x - median) / scale
    positive_x = (positive_x - median) / scale
    chosen = np.empty((len(positive_pairs), controls_per_edge, 2), dtype=np.int64)
    # Deterministic nearest-neighbour matching.  Reuse is permitted and is
    # accounted for by clustering inference at the source positive edge.
    for index, target in enumerate(positive_x):
        distance = np.sum((candidate_x - target) ** 2, axis=1)
        nearest = exact_topk_indices(distance, candidates, controls_per_edge)
        chosen[index] = candidates[nearest]
    return feature_ids[chosen]


def exact_topk_indices(
    distance: np.ndarray, candidates: np.ndarray, k: int
) -> np.ndarray:
    """Return the exact lexicographic top-k without sorting the full array.

    Ordering is identical to ``lexsort((right, left, distance))``.  The kth
    distance threshold is partitioned in linear time, and all exact ties at
    that threshold are retained before the final small lexicographic sort.
    """

    if k <= 0 or k > len(distance):
        raise ValueError("k must be between one and the number of candidates")
    threshold = float(np.partition(distance, k - 1)[k - 1])
    pool = np.flatnonzero(distance <= threshold)
    order = np.lexsort(
        (candidates[pool, 1], candidates[pool, 0], distance[pool])
    )
    return pool[order[:k]]


def cluster_bootstrap(
    delta: np.ndarray, clusters: np.ndarray, repeats: int, seed: int
) -> dict[str, Any]:
    unique = np.unique(clusters)
    grouped = {value: delta[clusters == value] for value in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        selected = rng.choice(unique, size=len(unique), replace=True)
        draws[repeat] = float(np.mean(np.concatenate([grouped[value] for value in selected])))
    return {
        "mean": float(np.mean(delta)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(unique)),
        "resamples": int(repeats),
    }


def summarize_family(
    pair_table: pd.DataFrame, family: str, repeats: int, seed: int
) -> dict[str, Any]:
    frame = pair_table if family == "overall" else pair_table[pair_table["family"] == family]
    if frame.empty:
        return {"edges": 0}
    decoy_columns = [column for column in frame if column.startswith("decoy_similarity_")]
    decoy = frame[decoy_columns].to_numpy(float)
    positive = frame["dreams_similarity"].to_numpy(float)
    delta = positive - decoy.mean(axis=1)
    labels = np.concatenate([np.ones(len(positive)), np.zeros(decoy.size)])
    scores = np.concatenate([positive, decoy.ravel()])
    return {
        "edges": len(frame),
        "positive_similarity_mean": float(np.mean(positive)),
        "matched_nonedge_similarity_mean": float(np.mean(decoy)),
        "paired_mean_delta": float(np.mean(delta)),
        "matched_auc": float(roc_auc_score(labels, scores)),
        "component_cluster_bootstrap": cluster_bootstrap(
            delta, frame["component"].to_numpy(int), repeats, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/external/netid_v1/source/LiChenPU-NetID-9f63202"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/validation/netid_mouse_liver_dreams_20260831"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_dreams_edge_signal_20260831"),
    )
    parser.add_argument("--controls-per-edge", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--panel", default="Mouse_liver_neg")
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    pair_path = args.output_dir / "edge_matched_nonedges.csv.gz"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") not in {
            "netid_dreams_edge_signal_passed",
            "netid_dreams_edge_signal_failed",
        }:
            raise RuntimeError("invalid existing edge-signal result")
        if not pair_path.is_file() or sha256(pair_path) != report["provenance"]["pair_table_sha256"]:
            raise RuntimeError("existing edge-signal pair table changed")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {args.output_dir}")
    embedding_report_path = args.embedding_dir / "report.json"
    embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
    if embedding_report.get("status") != "netid_mouse_liver_dreams_embeddings_frozen":
        raise RuntimeError("official NetID DreaMS embeddings are not frozen")
    embedding_path = args.embedding_dir / "official_dreams_embeddings.npz"
    if sha256(embedding_path) != embedding_report["provenance"]["embeddings_sha256"]:
        raise RuntimeError("embedding hash mismatch")
    with np.load(embedding_path, allow_pickle=False) as cache:
        feature_ids, vectors, mz, rt, peak_counts = aggregate_feature_embeddings(
            np.asarray(cache["embeddings"], dtype=np.float32),
            np.asarray(cache["netid_peak_id"], dtype=np.int64),
            np.asarray(cache["precursor_mz"], dtype=float),
            np.asarray(cache["raw_rt_min"], dtype=float),
            np.asarray(cache["n_fragment_peaks"], dtype=np.int64),
        )
    feature_index = {int(value): index for index, value in enumerate(feature_ids)}
    panel = args.source_root / args.panel
    nodes = pd.read_csv(panel / "cyto_nodes.csv")
    edges = pd.read_csv(panel / "cyto_edges.csv")
    if nodes["ilp_node_id"].duplicated().any() or nodes["node_id"].duplicated().any():
        raise RuntimeError("post-solution NetID node identifiers are not unique")
    node_lookup = nodes.set_index("ilp_node_id")["node_id"]
    edge_frame = edges.copy()
    edge_frame["feature1"] = edge_frame["ilp_nodes1"].map(node_lookup)
    edge_frame["feature2"] = edge_frame["ilp_nodes2"].map(node_lookup)
    edge_frame = edge_frame[
        edge_frame["feature1"].isin(feature_index)
        & edge_frame["feature2"].isin(feature_index)
        & edge_frame["feature1"].ne(edge_frame["feature2"])
    ].copy()
    edge_frame[["feature1", "feature2"]] = np.sort(
        edge_frame[["feature1", "feature2"]].to_numpy(dtype=np.int64), axis=1
    )
    if edge_frame.duplicated(["feature1", "feature2"]).any():
        raise RuntimeError("multiple author edge records map to the same feature pair")
    edge_frame["family"] = np.where(
        edge_frame["category"].eq("Biotransform"), "biotransform", "ion_phenomenon"
    )
    positive_pairs = edge_frame[["feature1", "feature2"]].to_numpy(dtype=np.int64)
    if len(positive_pairs) < 150:
        raise RuntimeError(f"too few public author edges with DreaMS spectra: {len(positive_pairs)}")

    degree_map = pd.concat([edge_frame["feature1"], edge_frame["feature2"]]).value_counts()
    degree = np.asarray([int(degree_map.get(feature, 0)) for feature in feature_ids])
    decoys = build_matched_decoys(
        feature_ids,
        mz,
        rt,
        peak_counts,
        degree,
        positive_pairs,
        args.controls_per_edge,
    )
    union = UnionFind(feature_ids)
    for left, right in positive_pairs:
        union.union(int(left), int(right))
    edge_frame["component"] = [union.find(int(left)) for left in edge_frame["feature1"]]
    edge_frame["dreams_similarity"] = [
        float(vectors[feature_index[int(left)]] @ vectors[feature_index[int(right)]])
        for left, right in positive_pairs
    ]
    for control in range(args.controls_per_edge):
        edge_frame[f"decoy_feature1_{control}"] = decoys[:, control, 0]
        edge_frame[f"decoy_feature2_{control}"] = decoys[:, control, 1]
        edge_frame[f"decoy_similarity_{control}"] = [
            float(vectors[feature_index[int(left)]] @ vectors[feature_index[int(right)]])
            for left, right in decoys[:, control]
        ]

    summaries = {
        family: summarize_family(edge_frame, family, args.bootstrap_resamples, args.seed + index)
        for index, family in enumerate(["overall", "biotransform", "ion_phenomenon"])
    }
    overall = summaries["overall"]
    gates = {
        "author_edges_ge_150": overall["edges"] >= 150,
        "overall_component_ci_low_gt_zero": overall["component_cluster_bootstrap"]["ci_low"] > 0,
        "biotransform_mean_delta_nonnegative": summaries["biotransform"].get("paired_mean_delta", -1) >= 0,
        "ion_phenomenon_mean_delta_nonnegative": summaries["ion_phenomenon"].get("paired_mean_delta", -1) >= 0,
    }
    gates["pass_to_calibrated_netid_overlay"] = all(gates.values())
    status = (
        "netid_dreams_edge_signal_passed"
        if gates["pass_to_calibrated_netid_overlay"]
        else "netid_dreams_edge_signal_failed"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(pair_path, edge_frame)
    report = {
        "status": status,
        "formal": True,
        "protocol": (
            "official DreaMS centroid cosine on author post-solution feature edges "
            "versus precursor-mass, RT, graph-degree and peak-count matched nonedges"
        ),
        "panel": args.panel,
        "features_with_embeddings": len(feature_ids),
        "author_edges_with_both_spectra": len(edge_frame),
        "controls_per_edge": args.controls_per_edge,
        "families": summaries,
        "gates": gates,
        "contracts": {
            "matching_uses_dreams_similarity": False,
            "author_assignments_are_independent_truth": False,
            "model_or_threshold_fitted": False,
            "P2b_used": False,
            "phenotype_used": False,
            "component_cluster_inference": True,
        },
        "provenance": {
            "embedding_report_sha256": sha256(embedding_report_path),
            "embeddings_sha256": sha256(embedding_path),
            "cyto_nodes_sha256": sha256(panel / "cyto_nodes.csv"),
            "cyto_edges_sha256": sha256(panel / "cyto_edges.csv"),
            "pair_table_sha256": sha256(pair_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "next_step": (
            "If the gate passes, freeze a component-cross-fitted DreaMS edge calibrator "
            "and compare fixed-FDR network coverage. If it fails, do not inject DreaMS "
            "edges into NetID from this source."
        ),
        "claim_limit": (
            "Development feasibility on author post-solution graph labels; not an exact "
            "NetID rerun, independent annotation accuracy, shared-embedding improvement, "
            "or SOTA evidence."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
