"""Build the consumed MTBLS1905 candidate-context adapter dataset.

The universal candidate vector is the official reference spectrum attaining the
archived max-per-molecule DreaMS score.  This exactly reproduces the spectral
baseline while leaving biological context candidate-specific and phenotype-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/validation/bioaware_context_evidence_tensor_20260830"))
    parser.add_argument("--query-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/dreams_official_full"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_reference_dreams"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/mtbls1905_context_adapter_dataset_20260830"))
    parser.add_argument("--max-context-edges", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.max_context_edges < 1:
        raise ValueError("max-context-edges must be positive")

    candidate_path = args.tensor_dir / "mtbls1905_auto__candidates.csv.gz"
    edge_path = args.tensor_dir / "mtbls1905_auto__edges.csv.gz"
    label_path = args.tensor_dir / "mtbls1905_auto__labels.csv.gz"
    candidates = pd.read_csv(candidate_path)
    edges = pd.read_csv(edge_path)
    labels = pd.read_csv(label_path)
    candidates = candidates.merge(
        labels[["query_id", "candidate_id", "truth_candidate_id", "truth_formula", "is_positive"]],
        on=["query_id", "candidate_id"], how="left", validate="one_to_one",
    )
    if candidates.is_positive.isna().any():
        raise RuntimeError("candidate labels are incomplete")

    query_manifest = pd.read_csv(args.query_dir / "manifest.csv")
    query_embeddings = normalize(np.load(args.query_dir / "official_embeddings.npy"))
    query_keys = query_manifest.source_file.astype(str) + "|" + query_manifest.spectrum_id.astype(str)
    if query_keys.duplicated().any() or len(query_manifest) != len(query_embeddings):
        raise RuntimeError("query manifest/embedding mismatch")
    query_index = dict(zip(query_keys, range(len(query_keys))))

    reference_manifest = pd.read_csv(args.reference_dir / "manifest.csv")
    reference_embeddings = normalize(np.load(args.reference_dir / "embeddings.npy"))
    if len(reference_manifest) != len(reference_embeddings):
        raise RuntimeError("reference manifest/embedding mismatch")
    reference_manifest["ik14"] = reference_manifest.inchikey.astype(str).str[:14]
    reference_index = {
        key: group.index.to_numpy(np.int64)
        for key, group in reference_manifest.groupby("ik14", sort=False)
    }

    flat_candidate_ids: list[str] = []
    flat_candidate_embeddings: list[np.ndarray] = []
    flat_seed_embeddings: list[np.ndarray] = []
    flat_relations: list[np.ndarray] = []
    flat_features: list[np.ndarray] = []
    flat_masks: list[np.ndarray] = []
    query_ids: list[str] = []
    truth_formulas: list[str] = []
    query_vectors: list[np.ndarray] = []
    offsets = [0]
    positive_indices: list[int] = []
    baseline_max_error = 0.0
    truncated_candidates = 0
    for query_id, group in candidates.groupby("query_id", sort=False):
        group = group.reset_index(drop=True)
        if int(group.is_positive.sum()) != 1:
            raise RuntimeError(f"{query_id}: expected one positive candidate")
        if query_id not in query_index:
            raise RuntimeError(f"missing query embedding: {query_id}")
        qvec = query_embeddings[query_index[query_id]]
        query_ids.append(str(query_id))
        truth_formulas.append(str(group.truth_formula.iloc[0]))
        query_vectors.append(qvec)
        positive_indices.append(int(np.flatnonzero(group.is_positive.to_numpy(bool))[0]))
        for row in group.itertuples(index=False):
            identity = str(row.candidate_id)
            if identity not in reference_index:
                raise RuntimeError(f"missing candidate reference embedding: {identity}")
            rows = reference_index[identity]
            scores = reference_embeddings[rows] @ qvec
            best = int(rows[int(np.argmax(scores))])
            baseline_max_error = max(baseline_max_error, abs(float(scores.max()) - float(row.spectral_score)))
            flat_candidate_ids.append(identity)
            flat_candidate_embeddings.append(reference_embeddings[best])
            local = edges[(edges.query_id.astype(str) == str(query_id)) & (edges.candidate_id.astype(str) == identity)].copy()
            local = local.sort_values(
                ["path_confidence", "experimental_support", "reaction_completeness"],
                ascending=False, kind="stable",
            )
            if len(local) > args.max_context_edges:
                truncated_candidates += 1
            local = local.head(args.max_context_edges)
            seeds = np.zeros((args.max_context_edges, qvec.shape[0]), dtype=np.float32)
            relations = np.zeros(args.max_context_edges, dtype=np.int64)
            features = np.zeros((args.max_context_edges, 4), dtype=np.float32)
            mask = np.zeros(args.max_context_edges, dtype=bool)
            for edge_position, edge in enumerate(local.itertuples(index=False)):
                seed_query_id = str(edge.seed_query_id)
                if seed_query_id not in query_index:
                    raise RuntimeError(f"missing seed query embedding: {seed_query_id}")
                seeds[edge_position] = query_embeddings[query_index[seed_query_id]]
                relations[edge_position] = int(edge.relation_type)
                features[edge_position] = [
                    float(edge.path_confidence), float(edge.experimental_support),
                    float(edge.reaction_completeness), float(edge.conflict),
                ]
                mask[edge_position] = True
            flat_seed_embeddings.append(seeds)
            flat_relations.append(relations)
            flat_features.append(features)
            flat_masks.append(mask)
        offsets.append(len(flat_candidate_ids))

    if baseline_max_error > 1e-5:
        raise RuntimeError(f"official max-per-molecule baseline mismatch: {baseline_max_error}")
    arrays_path = args.output_dir / "dataset.npz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        query_ids=np.asarray(query_ids, dtype=str),
        truth_formulas=np.asarray(truth_formulas, dtype=str),
        query_embeddings=np.asarray(query_vectors, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        positive_indices=np.asarray(positive_indices, dtype=np.int64),
        candidate_ids=np.asarray(flat_candidate_ids, dtype=str),
        candidate_embeddings=np.asarray(flat_candidate_embeddings, dtype=np.float32),
        seed_embeddings=np.asarray(flat_seed_embeddings, dtype=np.float32),
        relation_types=np.asarray(flat_relations, dtype=np.int64),
        edge_features=np.asarray(flat_features, dtype=np.float32),
        edge_masks=np.asarray(flat_masks, dtype=bool),
    )
    report = {
        "status": "mtbls1905_context_adapter_dataset_complete",
        "formal": False,
        "queries": len(query_ids),
        "formulas": len(set(truth_formulas)),
        "candidate_rows": len(flat_candidate_ids),
        "candidates_with_context": int(np.asarray(flat_masks).any(axis=1).sum()),
        "context_edges": int(np.asarray(flat_masks).sum()),
        "max_context_edges": args.max_context_edges,
        "truncated_candidates": truncated_candidates,
        "official_baseline_max_absolute_error": baseline_max_error,
        "contracts": {
            "consumed_development_only": True,
            "phenotype_blind": True,
            "truth_used_only_for_separate_training_labels": True,
            "candidate_vector": "official reference spectrum attaining archived max-per-molecule score",
            "no_context_fallback": "exact universal candidate embedding",
        },
        "provenance": {
            "candidates_sha256": sha256(candidate_path),
            "edges_sha256": sha256(edge_path),
            "labels_sha256": sha256(label_path),
            "query_embeddings_sha256": sha256(args.query_dir / "official_embeddings.npy"),
            "reference_embeddings_sha256": sha256(args.reference_dir / "embeddings.npy"),
            "dataset_sha256": sha256(arrays_path),
        },
        "claim_limit": "Consumed mechanism dataset; no external performance claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
