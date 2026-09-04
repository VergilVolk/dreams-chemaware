#!/usr/bin/env python
"""Audit the paper-aligned MetDNA3 raw-MS2 data-layer edge on HILIC dev."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict[str, float | int]:
    unique = np.unique(clusters.astype(str))
    grouped = {value: values[clusters.astype(str) == value] for value in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(5000, dtype=float)
    for repeat in range(len(boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boot[repeat] = np.mean(np.concatenate([grouped[value] for value in sampled]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "clusters": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v1"))
    parser.add_argument("--scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--paths", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/evidence_paths.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/bioaware_metdna3_raw_ms2_layer.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: {args.output}")

    manifest_path = args.cache_dir / "external_spectra.csv.gz"
    tensor_path = args.cache_dir / "external_tensors.npz"
    query_path = args.cache_dir / "queries.csv.gz"
    manifest = pd.read_csv(manifest_path).reset_index(drop=True)
    tensors = np.load(tensor_path, allow_pickle=False)["external_tensor"]
    query = pd.read_csv(query_path).set_index("query_id")
    scores = pd.read_csv(args.scores)
    paths = pd.read_csv(args.paths)
    if len(manifest) != len(tensors):
        raise RuntimeError("external spectrum manifest/tensor mismatch")
    spectrum_position = {str(key): position for position, key in enumerate(manifest.spectrum_key)}
    positions_by_identity = {
        str(identity): group.index.to_numpy(int)
        for identity, group in manifest.groupby("truth_ik14", sort=False)
    }
    truth_by_query = scores.groupby("query_id").truth_candidate_id.first().astype(str).to_dict()
    formula_by_query = scores.groupby("query_id").truth_formula.first().astype(str).to_dict()
    baseline_by_query = (
        scores.sort_values(["query_id", "spectral_score", "candidate_id"], ascending=[True, False, True])
        .groupby("query_id").candidate_id.first().astype(str).to_dict()
    )

    unique_paths = paths[["fold", "query_id", "query_candidate_id", "seed_compound_id"]].drop_duplicates().copy()
    path_rows: list[dict[str, object]] = []
    for row in unique_paths.itertuples(index=False):
        query_position = spectrum_position[str(query.loc[row.query_id, "spectrum_key"])]
        seed_positions = positions_by_identity.get(str(row.seed_compound_id), np.empty(0, dtype=int))
        if not len(seed_positions):
            continue
        similarities = [
            metdna3_reverse_dot(tensors[query_position], tensors[position])
            for position in seed_positions
        ]
        path_rows.append({
            "fold": int(row.fold),
            "query_id": row.query_id,
            "query_candidate_id": str(row.query_candidate_id),
            "seed_compound_id": str(row.seed_compound_id),
            "raw_ms2_similarity": float(max(similarities)),
            "seed_spectra": int(len(seed_positions)),
            "truth_path": str(row.query_candidate_id) == truth_by_query[row.query_id],
        })
    path_frame = pd.DataFrame(path_rows)
    if path_frame.empty:
        raise RuntimeError("no evaluable MetDNA3 raw-MS2 paths")

    candidate = (
        path_frame.groupby(["fold", "query_id", "query_candidate_id"], sort=False)
        .raw_ms2_similarity.max().reset_index()
    )
    paired_rows: list[dict[str, object]] = []
    for (fold, query_id), group in candidate.groupby(["fold", "query_id"], sort=False):
        truth = truth_by_query[query_id]
        truth_score = group.loc[group.query_candidate_id == truth, "raw_ms2_similarity"]
        wrong_score = group.loc[group.query_candidate_id != truth, "raw_ms2_similarity"]
        if not len(truth_score) or not len(wrong_score):
            continue
        paired_rows.append({
            "fold": int(fold),
            "query_id": query_id,
            "truth_formula": formula_by_query[query_id],
            "baseline_wrong": baseline_by_query[query_id] != truth,
            "truth_max_similarity": float(truth_score.max()),
            "wrong_max_similarity": float(wrong_score.max()),
            "delta": float(truth_score.max() - wrong_score.max()),
        })
    paired = pd.DataFrame(paired_rows)
    error = paired[paired.baseline_wrong]

    thresholds: dict[str, dict[str, float]] = {}
    for threshold in np.arange(0.1, 0.91, 0.1):
        truth_pass = paired.truth_max_similarity >= threshold
        wrong_pass = paired.wrong_max_similarity >= threshold
        thresholds[f"{threshold:.1f}"] = {
            "truth_path_retention": float((path_frame.loc[path_frame.truth_path, "raw_ms2_similarity"] >= threshold).mean()),
            "wrong_path_retention": float((path_frame.loc[~path_frame.truth_path, "raw_ms2_similarity"] >= threshold).mean()),
            "paired_truth_only_fraction": float((truth_pass & ~wrong_pass).mean()),
            "paired_wrong_only_fraction": float((wrong_pass & ~truth_pass).mean()),
        }

    report = {
        "status": "bioaware_metdna3_raw_ms2_layer_audit_complete",
        "formal": True,
        "protocol": "MetDNA3 v3.1.1 scoreReverse; smaller-precursor reference; 25 ppm; intensity^1; mz^0",
        "paths": {
            "rows": int(len(path_frame)),
            "truth": int(path_frame.truth_path.sum()),
            "wrong": int((~path_frame.truth_path).sum()),
            "truth_median": float(path_frame.loc[path_frame.truth_path, "raw_ms2_similarity"].median()),
            "wrong_median": float(path_frame.loc[~path_frame.truth_path, "raw_ms2_similarity"].median()),
        },
        "paired_candidate_audit": {
            "instances": int(len(paired)),
            "truth_gt_wrong_fraction": float((paired.delta > 0).mean()),
            "median_delta": float(paired.delta.median()),
            "formula_cluster_mean_delta": cluster_bootstrap(
                paired.delta.to_numpy(float), paired.truth_formula.to_numpy(str), 20260828
            ),
            "baseline_error_instances": int(len(error)),
            "error_truth_gt_wrong_fraction": float((error.delta > 0).mean()),
            "error_median_delta": float(error.delta.median()),
            "error_formula_cluster_mean_delta": cluster_bootstrap(
                error.delta.to_numpy(float), error.truth_formula.to_numpy(str), 20260829
            ),
        },
        "threshold_screen": thresholds,
        "gates": {
            "overall_formula_ci_low_positive": False,
            "error_formula_ci_low_positive": False,
            "error_truth_gt_wrong_fraction_gt_half": False,
        },
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "tensor_sha256": sha256(tensor_path),
            "queries_sha256": sha256(query_path),
            "scores_sha256": sha256(args.scores),
            "paths_sha256": sha256(args.paths),
        },
        "claim_limit": "Consumed HILIC development audit; no threshold is frozen and RP remains unopened.",
    }
    report["gates"]["overall_formula_ci_low_positive"] = (
        report["paired_candidate_audit"]["formula_cluster_mean_delta"]["ci_low"] > 0
    )
    report["gates"]["error_formula_ci_low_positive"] = (
        report["paired_candidate_audit"]["error_formula_cluster_mean_delta"]["ci_low"] > 0
    )
    report["gates"]["error_truth_gt_wrong_fraction_gt_half"] = (
        report["paired_candidate_audit"]["error_truth_gt_wrong_fraction"] > 0.5
    )
    report["pass_to_two_layer_development"] = bool(all(report["gates"].values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    path_frame.to_csv(args.output.with_suffix(".paths.csv.gz"), index=False)
    paired.to_csv(args.output.with_suffix(".pairs.csv"), index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
