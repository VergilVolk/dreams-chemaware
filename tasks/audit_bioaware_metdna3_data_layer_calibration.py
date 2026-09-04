#!/usr/bin/env python
"""Audit whether DreaMS cosine can serve as the MetDNA3 data-layer constraint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    repeats: int = 5000,
    seed: int = 20260828,
) -> dict[str, float | int]:
    """Bootstrap a mean while preserving all observations from each formula."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    unique_clusters = np.unique(clusters)
    if not len(values) or not len(unique_clusters):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "clusters": 0}
    grouped = {cluster: values[clusters == cluster] for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        boot[repeat] = np.mean(np.concatenate([grouped[cluster] for cluster in sampled]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "clusters": int(len(unique_clusters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v1"))
    parser.add_argument("--embedding", type=Path, default=Path("data/validation/bioaware_metdna3_data_layer_embeddings.npz"))
    parser.add_argument("--scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--paths", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/evidence_paths.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/bioaware_metdna3_data_layer_calibration.json"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.cache_dir / "external_spectra.csv.gz")
    query = pd.read_csv(args.cache_dir / "queries.csv.gz").set_index("query_id")
    embedding = np.load(args.embedding, allow_pickle=False)["embedding"]
    scores = pd.read_csv(args.scores)
    paths = pd.read_csv(args.paths)
    if len(manifest) != len(embedding):
        raise RuntimeError("manifest/embedding mismatch")
    spectrum_position = {key: pos for pos, key in enumerate(manifest.spectrum_key)}
    positions_by_identity = {
        identity: group.index.to_numpy(int)
        for identity, group in manifest.groupby("truth_ik14", sort=False)
    }
    truth_by_query = scores.groupby("query_id").truth_candidate_id.first().to_dict()
    baseline_by_query = (
        scores.sort_values(["query_id", "spectral_score", "candidate_id"], ascending=[True, False, True])
        .groupby("query_id").candidate_id.first().to_dict()
    )
    formula_by_query = scores.groupby("query_id").truth_formula.first().to_dict()
    unique = paths[["fold", "query_id", "query_candidate_id", "seed_compound_id"]].drop_duplicates().copy()
    similarities = []
    for row in unique.itertuples(index=False):
        q = embedding[spectrum_position[str(query.loc[row.query_id, "spectrum_key"])]]
        positions = positions_by_identity.get(str(row.seed_compound_id))
        similarities.append(float(np.max(embedding[positions] @ q)) if positions is not None else np.nan)
    unique["data_similarity"] = similarities
    unique = unique.dropna(subset=["data_similarity"])
    unique["truth_path"] = unique.apply(
        lambda row: str(row.query_candidate_id) == str(truth_by_query[row.query_id]), axis=1
    )
    candidate_similarity = (
        unique.groupby(["fold", "query_id", "query_candidate_id"], sort=False)
        .data_similarity.max().reset_index()
    )
    pair_rows = []
    for (fold, query_id), group in candidate_similarity.groupby(["fold", "query_id"], sort=False):
        truth = str(truth_by_query[query_id])
        truth_values = group[group.query_candidate_id.astype(str) == truth].data_similarity
        wrong_values = group[group.query_candidate_id.astype(str) != truth].data_similarity
        if not len(truth_values) or not len(wrong_values):
            continue
        pair_rows.append({
            "fold": int(fold), "query_id": query_id, "truth_formula": formula_by_query[query_id],
            "baseline_wrong": str(baseline_by_query[query_id]) != truth,
            "truth_max_similarity": float(truth_values.max()),
            "wrong_max_similarity": float(wrong_values.max()),
            "delta": float(truth_values.max() - wrong_values.max()),
        })
    pair = pd.DataFrame(pair_rows)
    thresholds = {}
    for threshold in np.arange(0.1, 0.91, 0.1):
        truth_pass = pair.truth_max_similarity >= threshold
        wrong_pass = pair.wrong_max_similarity >= threshold
        thresholds[f"{threshold:.1f}"] = {
            "truth_path_retention": float((unique[unique.truth_path].data_similarity >= threshold).mean()),
            "wrong_path_retention": float((unique[~unique.truth_path].data_similarity >= threshold).mean()),
            "paired_truth_retained_fraction": float(truth_pass.mean()) if len(pair) else None,
            "paired_wrong_retained_fraction": float(wrong_pass.mean()) if len(pair) else None,
            "paired_truth_only_fraction": float((truth_pass & ~wrong_pass).mean()) if len(pair) else None,
            "paired_wrong_only_fraction": float((wrong_pass & ~truth_pass).mean()) if len(pair) else None,
        }
    error_pair = pair[pair.baseline_wrong]
    delta_bootstrap = cluster_bootstrap_mean(
        pair.delta.to_numpy(float), pair.truth_formula.to_numpy(str)
    )
    error_delta_bootstrap = cluster_bootstrap_mean(
        error_pair.delta.to_numpy(float), error_pair.truth_formula.to_numpy(str), seed=20260829
    )
    report = {
        "status": "bioaware_metdna3_data_layer_calibration_complete", "formal": True,
        "path_rows": int(len(unique)), "truth_path_rows": int(unique.truth_path.sum()),
        "wrong_path_rows": int((~unique.truth_path).sum()),
        "similarity": {
            "truth_path_median": float(unique[unique.truth_path].data_similarity.median()),
            "wrong_path_median": float(unique[~unique.truth_path].data_similarity.median()),
        },
        "paired_candidate_audit": {
            "instances": int(len(pair)), "truth_similarity_gt_wrong_fraction": float((pair.delta > 0).mean()),
            "median_truth_minus_wrong": float(pair.delta.median()),
            "formula_cluster_mean_delta_bootstrap": delta_bootstrap,
            "baseline_error_instances": int(len(error_pair)),
            "error_truth_similarity_gt_wrong_fraction": float((error_pair.delta > 0).mean()) if len(error_pair) else None,
            "error_median_truth_minus_wrong": float(error_pair.delta.median()) if len(error_pair) else None,
            "error_formula_cluster_mean_delta_bootstrap": error_delta_bootstrap,
        },
        "threshold_screen": thresholds,
        "decision": (
            "Calibrate DreaMS data-layer edges only if truth-supporting reaction paths separate from "
            "within-query wrong-candidate paths; otherwise use the published raw-MS2 score or a learned peak-token edge."
        ),
        "claim_limit": "Consumed HILIC development diagnostic; thresholds are not frozen and RP remains unopened.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"fail-closed: {args.output}")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pair.to_csv(args.output.with_suffix(".pairs.csv"), index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
