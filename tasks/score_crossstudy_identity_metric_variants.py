#!/usr/bin/env python3
"""Truth-blind metric variants over frozen cross-study embeddings.

The script is intentionally narrow: it compares geometry/readout choices for
Noise V1 and its full-MassSpecGym ChemAware continuation.  It never reads
identity labels or biological responses.  Learned supervised projections are
out of scope here because the local machine does not contain the MassSpecGym
training artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "CROSSSTUDY_IDENTITY_METRIC_VARIANTS_COMPLETE"
SEED = 20261025
HASH_BITS = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm > 0)


def study_ids(entity_ids: np.ndarray) -> np.ndarray:
    return np.asarray([str(value).split("::", 1)[0] for value in entity_ids])


def study_center(matrix: np.ndarray, studies: np.ndarray) -> np.ndarray:
    output = np.empty_like(matrix, dtype=np.float32)
    for study in np.unique(studies):
        mask = studies == study
        output[mask] = matrix[mask] - matrix[mask].mean(axis=0, keepdims=True)
    return normalize(output)


def diagonal_whiten(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0, keepdims=True)
    floor = max(float(np.median(scale)) * 1e-3, 1e-6)
    return normalize(centered / np.maximum(scale, floor))


def simhash(matrix: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    # Sparse random hyperplanes reduce local CPU cost while preserving the
    # angular-LSH interpretation. Nonzeros are symmetric and expectation-zero.
    draw = rng.integers(0, 6, size=(matrix.shape[1], HASH_BITS), dtype=np.int8)
    projection = np.zeros(draw.shape, dtype=np.float32)
    projection[draw == 0] = -1.0
    projection[draw == 1] = 1.0
    return np.matmul(matrix, projection) >= 0


def score_pairs(matrix: np.ndarray, query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", matrix[query], matrix[candidate], dtype=np.float32)


def top1_hubness(
    pairs: pd.DataFrame, scores: np.ndarray
) -> dict[str, float | int]:
    work = pairs[["query_id", "candidate_study_id", "candidate_id"]].copy()
    work["score"] = scores
    work = work.sort_values(
        ["query_id", "candidate_study_id", "score", "candidate_id"],
        ascending=[True, True, False, True], kind="mergesort",
    )
    top = work.drop_duplicates(["query_id", "candidate_study_id"], keep="first")
    counts = top["candidate_id"].value_counts().to_numpy(dtype=float)
    if len(counts) == 0:
        return {"top1_queries": 0, "unique_top1_candidates": 0, "maximum_top1_indegree": 0,
                "top1_indegree_gini": 0.0}
    ordered = np.sort(counts)
    n = len(ordered)
    gini = float(
        (2 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum()))
        - (n + 1) / n
    )
    return {
        "top1_queries": int(len(top)),
        "unique_top1_candidates": int(n),
        "maximum_top1_indegree": int(counts.max()),
        "top1_indegree_gini": gini,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoring-dir", type=Path,
        default=Path(
            "data/validation/msv000095331_integrated_multiencoder_identity_run_2404230"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    pair_path = args.scoring_dir / "identity_scoring" / "identity_candidate_scores.tsv.gz"
    pairs = pd.read_csv(pair_path, sep="\t", dtype={
        "candidate_pair_id": str, "query_id": str, "candidate_id": str,
    })
    embedding_paths = {
        "noise": args.scoring_dir / "noise_v1_embeddings.npz",
        "fullmsg": args.scoring_dir / "noise_chemaware_fullmsg_embeddings.npz",
    }
    loaded: dict[str, np.ndarray] = {}
    entity_ids: np.ndarray | None = None
    for name, path in embedding_paths.items():
        with np.load(path, allow_pickle=False) as payload:
            current_ids = np.asarray(payload["entity_id"]).astype(str)
            current = normalize(np.asarray(payload["embeddings"], dtype=np.float32))
        if entity_ids is None:
            entity_ids = current_ids
        elif not np.array_equal(entity_ids, current_ids):
            raise RuntimeError("embedding entity axes differ")
        loaded[name] = current
    assert entity_ids is not None
    index = {value: row for row, value in enumerate(entity_ids)}
    query = pairs["query_id"].map(index)
    candidate = pairs["candidate_id"].map(index)
    if query.isna().any() or candidate.isna().any():
        raise RuntimeError("candidate table contains entities absent from embeddings")
    query_row = query.to_numpy(dtype=np.int64)
    candidate_row = candidate.to_numpy(dtype=np.int64)
    studies = study_ids(entity_ids)

    noise_raw = loaded["noise"]
    full_raw = loaded["fullmsg"]
    noise_centered = study_center(noise_raw, studies)
    full_centered = study_center(full_raw, studies)
    noise_white = diagonal_whiten(noise_raw)
    full_white = diagonal_whiten(full_raw)

    scores: dict[str, np.ndarray] = {
        "metric_noise_raw_cosine": score_pairs(noise_raw, query_row, candidate_row),
        "metric_fullmsg_raw_cosine": score_pairs(full_raw, query_row, candidate_row),
        "metric_noise_study_centered_cosine": score_pairs(
            noise_centered, query_row, candidate_row
        ),
        "metric_fullmsg_study_centered_cosine": score_pairs(
            full_centered, query_row, candidate_row
        ),
        "metric_noise_diagonal_whitened_cosine": score_pairs(
            noise_white, query_row, candidate_row
        ),
        "metric_fullmsg_diagonal_whitened_cosine": score_pairs(
            full_white, query_row, candidate_row
        ),
    }
    cross_forward = np.einsum(
        "ij,ij->i", noise_raw[query_row], full_raw[candidate_row], dtype=np.float32
    )
    cross_reverse = np.einsum(
        "ij,ij->i", full_raw[query_row], noise_raw[candidate_row], dtype=np.float32
    )
    scores["metric_noise_fullmsg_symmetric_cross_cosine"] = (
        0.5 * (cross_forward + cross_reverse)
    ).astype(np.float32)

    full_hash = simhash(full_raw)
    hamming = np.not_equal(full_hash[query_row], full_hash[candidate_row]).mean(axis=1)
    # Expected random-hyperplane disagreement is angle/pi. Mapping it through
    # cos gives an interpretable approximation on the raw cosine scale.
    scores["metric_fullmsg_simhash256"] = np.cos(np.pi * hamming).astype(np.float32)

    original_noise = pairs["noise_v1_score"].to_numpy(dtype=np.float32)
    original_full = pairs["noise_chemaware_fullmsg_score"].to_numpy(dtype=np.float32)
    noise_error = float(np.max(np.abs(scores["metric_noise_raw_cosine"] - original_noise)))
    full_error = float(np.max(np.abs(scores["metric_fullmsg_raw_cosine"] - original_full)))
    if noise_error > 5e-6 or full_error > 5e-6:
        raise RuntimeError(
            f"raw cosine recomputation drifted: noise={noise_error}, fullmsg={full_error}"
        )

    output_blocks = []
    geometry: dict[str, dict] = {}
    for method, values in scores.items():
        block = pairs[["candidate_pair_id"]].copy()
        block["method_name"] = method
        block["method_score"] = values
        block["evidence_available"] = True
        output_blocks.append(block)
        geometry[method] = {
            "score_mean": float(np.mean(values)),
            "score_std": float(np.std(values)),
            "score_q05": float(np.quantile(values, 0.05)),
            "score_median": float(np.median(values)),
            "score_q95": float(np.quantile(values, 0.95)),
            **top1_hubness(pairs, values),
        }
    long = pd.concat(output_blocks, ignore_index=True)

    # Hashing is judged primarily by fidelity to the exact FullMSG angular
    # score. It is not allowed to claim biological accuracy from approximation.
    exact = scores["metric_fullmsg_raw_cosine"]
    approximate = scores["metric_fullmsg_simhash256"]
    hash_fidelity = {
        "pearson_pair_score": float(np.corrcoef(exact, approximate)[0, 1]),
        "mean_absolute_pair_score_error": float(np.mean(np.abs(exact - approximate))),
    }

    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "method_scores_long.tsv.gz"
    long.to_csv(
        score_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report = {
        "status": STATUS,
        "labels_or_responses_accessed": False,
        "candidate_pairs": int(len(pairs)),
        "embedding_entities": int(len(entity_ids)),
        "embedding_dimension": int(noise_raw.shape[1]),
        "raw_cosine_reproduction_max_abs_error": {
            "noise": noise_error, "fullmsg": full_error,
        },
        "metric_geometry": geometry,
        "simhash": {
            "bits": HASH_BITS,
            "seed": SEED,
            "projection": "sparse_random_hyperplane_p_minus1=p_plus1=1/6",
            **hash_fidelity,
        },
        "inputs": {
            "candidate_pairs": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "embeddings": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in embedding_paths.items()
            },
        },
        "outputs": {
            "method_scores": {"path": str(score_path), "sha256": sha256(score_path)},
        },
        "claim_boundary": [
            "All transformations are label-free exploratory readouts of frozen embeddings.",
            "Application response outcomes must not be used to fit or select a metric.",
            "Study centering is transductive and requires an explicit deployment batch definition.",
            "Diagonal whitening is an unsupervised diagnostic, not supervised metric learning.",
            "SimHash evaluates angular compression fidelity and retrieval engineering, not identity accuracy.",
            "A supervised projection or pair head must be trained on MassSpecGym train only and evaluated separately.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
