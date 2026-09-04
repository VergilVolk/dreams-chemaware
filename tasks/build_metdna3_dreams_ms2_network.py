#!/usr/bin/env python
"""Score the exact author-generated MetDNA3 feature-edge set with DreaMS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_COLUMNS = ("truth", "correct", "label", "outcome", "phenotype", "case", "control")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def canonical_key(left: str, right: str) -> str:
    if left == right:
        raise ValueError("self edge")
    return "\x1f".join(sorted((str(left), str(right))))


def validate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if not {"from", "to"}.issubset(edges.columns):
        raise RuntimeError("edge table misses from/to")
    suspicious = [column for column in edges.columns
                  if any(token in column.lower() for token in FORBIDDEN_COLUMNS)]
    if suspicious:
        raise RuntimeError(f"truth/outcome-like columns are forbidden: {suspicious}")
    result = edges.copy()
    result["from"] = result["from"].astype(str)
    result["to"] = result["to"].astype(str)
    result["edge_key"] = [canonical_key(a, b) for a, b in
                          zip(result["from"], result["to"], strict=True)]
    if result["edge_key"].duplicated().any():
        raise RuntimeError("duplicate undirected edges")
    if "edge_index" in result and not np.array_equal(
            result["edge_index"].to_numpy(np.int64), np.arange(len(result))):
        raise RuntimeError("edge_index must be contiguous zero-based author order")
    result["edge_index"] = np.arange(len(result), dtype=np.int64)
    return result


def score_edges(edges: pd.DataFrame, feature_names: np.ndarray,
                embeddings: np.ndarray) -> np.ndarray:
    names = [str(value) for value in feature_names]
    if len(names) != len(set(names)) or len(names) != len(embeddings):
        raise RuntimeError("embedding feature names are duplicated or misaligned")
    lookup = {name: position for position, name in enumerate(names)}
    missing = (set(edges["from"]) | set(edges["to"])) - set(lookup)
    if missing:
        raise RuntimeError(f"embedding cache misses {len(missing)} edge features")
    left = np.asarray([lookup[value] for value in edges["from"]], dtype=np.int64)
    right = np.asarray([lookup[value] for value in edges["to"]], dtype=np.int64)
    score = np.einsum("ij,ij->i", embeddings[left], embeddings[right]).astype(np.float64)
    if not np.isfinite(score).all() or np.any(score < -1.0001) or np.any(score > 1.0001):
        raise RuntimeError("invalid DreaMS cosine")
    return np.clip(score, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.edges, args.embeddings, args.embedding_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    embedding_report = json.loads(args.embedding_report.read_text(encoding="utf-8"))
    contracts = embedding_report.get("contracts", {})
    if not embedding_report.get("formal") or not contracts.get("one_shared_query_reference_encoder"):
        raise RuntimeError("invalid embedding contract")
    if contracts.get("identity_labels_used") or contracts.get("annotation_outcomes_used"):
        raise RuntimeError("label/outcome leakage in embedding cache")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is non-empty: {output}")
    edges = validate_edges(pd.read_csv(args.edges))
    cache = np.load(args.embeddings, allow_pickle=False)
    embeddings = cache["embedding"].astype(np.float32)
    feature_names = cache["feature_name"]
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise RuntimeError("embedding cache is non-finite or not normalized")
    scores = score_edges(edges, feature_names, embeddings)
    result = edges[["edge_index", "from", "to", "edge_key"]].copy()
    result["dreams_cosine"] = scores
    score_path = output / "dreams_edge_scores.csv.gz"
    result.to_csv(score_path, index=False, compression="gzip")
    payload = {
        "status": "metdna3_dreams_edge_scores_complete",
        "formal": True,
        "edges": int(len(result)),
        "features": int(len(set(result["from"]) | set(result["to"]))),
        "score_distribution": {
            "minimum": float(np.min(scores)),
            "p10": float(np.quantile(scores, 0.1)),
            "median": float(np.median(scores)),
            "p90": float(np.quantile(scores, 0.9)),
            "maximum": float(np.max(scores)),
        },
        "contracts": {
            "exact_author_edge_set": True,
            "edge_order_preserved": True,
            "score_is_raw_cosine_not_probability": True,
            "threshold_selected": False,
            "truth_or_outcomes_used": False,
            "P2b_used": False,
        },
        "provenance": {
            "edges_sha256": sha256(args.edges),
            "embeddings_sha256": sha256(args.embeddings),
            "embedding_report_sha256": sha256(args.embedding_report),
            "scores_sha256": sha256(score_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "next_gate": "Fit a frozen development-only calibration/threshold before RDA injection.",
        "claim_limit": "Raw edge scores only; cosine is not a calibrated MetDNA3 ms2_score.",
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
