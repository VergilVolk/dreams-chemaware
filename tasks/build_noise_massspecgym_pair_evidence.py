#!/usr/bin/env python
"""Build the full corrected-graph pair-evidence table for fusion training.

Every directed query-reference edge of the corrected MassSpecGym graph receives
seven deployment-visible scores: V1 cosine, official DreaMS cosine, weighted
spectral entropy, and the three frozen P2b evidence channels plus the frozen
P2b fused score.  No label is read here; the positive-molecule ledger travels
with the graph and is stored unchanged for the trainer.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from build_noise_massspecgym_full_triplets import load_complete_embeddings
from noise_final_core import CandidateGraph, sha256_file, stable_fold
from noise_gnps_article_spectral_scores import (
    frozen_p2b_pair_features,
    prepare_spectrum,
    weighted_entropy_backend,
    weighted_entropy_similarity,
)

SPECTRA: dict[int, np.ndarray] | None = None
PRECURSOR: np.ndarray | None = None
TOLERANCE: float | None = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v1-embedding-shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--official-embedding-shard", type=Path, action="append", required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def _init(spectra, precursor, tolerance) -> None:
    global SPECTRA, PRECURSOR, TOLERANCE
    SPECTRA, PRECURSOR, TOLERANCE = spectra, precursor, float(tolerance)


def _score_pair(payload: tuple[int, int]) -> tuple[tuple[int, int], tuple[float, float, float, float]]:
    left, right = payload
    if SPECTRA is None or PRECURSOR is None or TOLERANCE is None:
        raise RuntimeError("pair-evidence worker was not initialized")
    spectrum_left, spectrum_right = SPECTRA[left], SPECTRA[right]
    precursor_left, precursor_right = float(PRECURSOR[left]), float(PRECURSOR[right])
    sqrt_cosine, entropy, neutral_loss = frozen_p2b_pair_features(
        spectrum_left, precursor_left, spectrum_right, precursor_right, TOLERANCE,
    )
    wse = weighted_entropy_similarity(spectrum_left, spectrum_right, TOLERANCE)
    return payload, (wse, sqrt_cosine, entropy, neutral_loss)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1 or args.fragment_tolerance != 0.02 or args.n_highest_peaks != 100:
        raise ValueError("registered pair-evidence settings drifted")
    backend = weighted_entropy_backend()
    if not backend.startswith("ms_entropy_"):
        raise RuntimeError("formal pair evidence requires the pinned ms_entropy backend")

    graph = CandidateGraph(args.graph)
    with h5py.File(args.data, "r") as handle:
        n_rows = len(handle["spectrum"])
        precursor = np.asarray(handle["precursor_mz"][:], dtype=np.float64)
    v1 = load_complete_embeddings(args.v1_embedding_shard, n_rows)
    official = load_complete_embeddings(args.official_embedding_shard, n_rows)
    if not np.all(np.isfinite(precursor)):
        raise RuntimeError("MassSpecGym precursor array is non-finite")

    n_edges = int(len(graph.pair_candidate_row))
    query_edge_ptr = np.asarray(graph.molecule_ptr, dtype=np.int64)[
        np.asarray(graph.query_ptr, dtype=np.int64)
    ]
    if query_edge_ptr[0] != 0 or query_edge_ptr[-1] != n_edges:
        raise RuntimeError("edge-level query pointer does not span the graph")

    print(f"[pair-evidence] {graph.n_queries:,} queries, {n_edges:,} edges", flush=True)
    spectra: dict[int, np.ndarray] = {}
    used_rows = np.unique(np.concatenate((
        np.asarray(graph.query_row, dtype=np.int64),
        np.asarray(graph.pair_candidate_row, dtype=np.int64),
    )))
    with h5py.File(args.data, "r") as handle:
        spectra_dataset = handle["spectrum"]
        for row in used_rows:
            spectra[int(row)] = prepare_spectrum(
                np.asarray(spectra_dataset[int(row)]), args.n_highest_peaks,
            )
    print(f"[pair-evidence] prepared {len(spectra):,} spectra", flush=True)

    query_rows = np.asarray(graph.query_row, dtype=np.int64)
    edge_query = np.repeat(
        np.arange(graph.n_queries, dtype=np.int64),
        np.diff(query_edge_ptr),
    )
    if len(edge_query) != n_edges:
        raise RuntimeError("edge query expansion drifted")

    # ``Executor.map`` preserves input order, so classical scores land directly
    # in edge order without a multi-gigabyte lookup dictionary.
    pair_inputs = zip(
        query_rows[edge_query].tolist(),
        np.asarray(graph.pair_candidate_row, dtype=np.int64).tolist(),
        strict=True,
    )
    classical = np.empty((n_edges, 4), dtype=np.float32)
    if args.workers == 1:
        _init(spectra, precursor, args.fragment_tolerance)
        iterator = map(_score_pair, pair_inputs)
        executor = None
    else:
        context = mp.get_context("fork")
        executor = ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context,
            initializer=_init, initargs=(spectra, precursor, args.fragment_tolerance),
        )
        iterator = executor.map(_score_pair, pair_inputs, chunksize=256)
    try:
        for index, (_, values) in enumerate(iterator):
            classical[index] = values
            if (index + 1) % 500_000 == 0 or index + 1 == n_edges:
                print(f"[pair-evidence] {index + 1:,}/{n_edges:,} edges", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    # Embedding cosines are computed query-block by query-block: materialising
    # a 6.2M x 1024 edge matrix would need ~25 GB per model.
    v1_cosine = np.empty(n_edges, dtype=np.float32)
    official_cosine = np.empty(n_edges, dtype=np.float32)
    candidate_rows = np.asarray(graph.pair_candidate_row, dtype=np.int64)
    for query in range(graph.n_queries):
        left, right = int(query_edge_ptr[query]), int(query_edge_ptr[query + 1])
        rows = candidate_rows[left:right]
        query_row = int(query_rows[query])
        v1_cosine[left:right] = v1[rows] @ v1[query_row]
        official_cosine[left:right] = official[rows] @ official[query_row]
        if query and query % 20000 == 0:
            print(f"[pair-evidence-cosine] {query:,}/{graph.n_queries:,} queries", flush=True)
    if not (np.all(np.isfinite(v1_cosine)) and np.all(np.isfinite(official_cosine))):
        raise RuntimeError("embedding cosine evidence is non-finite")
    # Frozen P2b fusion per query (vectorized, label-blind).
    from noise_gnps_article_spectral_scores import apply_frozen_p2b

    fused = np.empty(n_edges, dtype=np.float64)
    for query in range(graph.n_queries):
        left, right = int(query_edge_ptr[query]), int(query_edge_ptr[query + 1])
        if right - left <= 0:
            raise RuntimeError("query edge block is empty")
        molecule_left = int(graph.query_ptr[query])
        molecule_right = int(graph.query_ptr[query + 1])
        block_molecule_ptr = np.asarray(
            graph.molecule_ptr[molecule_left:molecule_right + 1], dtype=np.int64,
        ) - left
        fused[left:right] = apply_frozen_p2b(
            type(
                "Block", (), {
                    "molecule_ptr": block_molecule_ptr,
                    "query_ptr": np.asarray(
                        [0, len(block_molecule_ptr) - 1], dtype=np.int64,
                    ),
                },
            )(),
            v1_cosine[left:right],
            classical[left:right, 1].astype(np.float64),
            classical[left:right, 2].astype(np.float64),
            classical[left:right, 3].astype(np.float64),
        ).pair
    if not np.all(np.isfinite(fused)):
        raise RuntimeError("frozen P2b fusion produced non-finite scores")

    report = {
        "status": "NOISE_MSG_PAIR_EVIDENCE_COMPLETE",
        "edges": int(n_edges),
        "queries": int(graph.n_queries),
        "workers": int(args.workers),
        "fragment_tolerance_da": args.fragment_tolerance,
        "wse_backend": backend,
        "formula_folds": "stable_fold(formula, 5, 20260825) joined by the trainer",
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "data_sha256": sha256_file(args.data),
            "v1_shards": [sha256_file(path) for path in args.v1_embedding_shard],
            "official_shards": [sha256_file(path) for path in args.official_embedding_shard],
        },
        "claim_limit": "Deployment-visible pair evidence only; labels stay in the graph ledger.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez_compressed(
            staging / "evidence.npz",
            v1_cosine=v1_cosine.astype(np.float32),
            official_cosine=official_cosine.astype(np.float32),
            weighted_entropy=classical[:, 0],
            sqrt_cosine=classical[:, 1],
            entropy_similarity=classical[:, 2],
            neutral_loss_sqrt_cosine=classical[:, 3],
            p2b_fused=fused.astype(np.float32),
            query_edge_ptr=query_edge_ptr,
            molecule_ptr=np.asarray(graph.molecule_ptr, dtype=np.int64),
            query_ptr=np.asarray(graph.query_ptr, dtype=np.int64),
            molecule_label=np.asarray(graph.molecule_label, dtype=np.int8),
            query_formula=np.asarray([str(value) for value in graph.query_formula]),
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
