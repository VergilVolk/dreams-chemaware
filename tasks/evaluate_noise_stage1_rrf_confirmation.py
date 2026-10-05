#!/usr/bin/env python
"""Confirm fixed RRF(dreams, cosine, top-k; k=60) on Stage-1 geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from evaluate_gnps_gold_silver_10ppm_embeddings import paired_summary
from noise_corrected_fullgraph_evaluation import full_metrics, score_embeddings
from noise_final_core import CandidateGraph, stable_fold
from noise_stage1_rrf_core import MATCH_TOL_DA, RRF_K, TOPK, build_rrf_graph_scores


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    return value


def selected_spectra(graph, queries: np.ndarray, data: Path):
    needed = set(map(int, graph.query_row[queries]))
    for query_value in queries:
        query = int(query_value)
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        needed.update(map(int, graph.pair_candidate_row[pair_left:pair_right]))
    rows = np.asarray(sorted(needed), dtype=np.int64)
    with h5py.File(data, "r") as handle:
        spectra = np.asarray(handle["spectrum"][rows], dtype=np.float32)
    index = {int(row): position for position, row in enumerate(rows)}
    counts = np.sum(
        (spectra[:, 0, :] > 0) & (spectra[:, 1, :] > 0), axis=1,
    ).astype(np.int64)
    peaks = [
        (
            np.ascontiguousarray(spectra[position, 0, :int(counts[position])]),
            np.ascontiguousarray(spectra[position, 1, :int(counts[position])]),
        )
        for position in range(len(rows))
    ]

    def get(row: int):
        try:
            return peaks[index[int(row)]]
        except KeyError as error:
            raise RuntimeError(f"spectrum row {row} was not registered") from error

    return get, int(len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage1-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.graph, args.embeddings, args.data, args.stage1_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph = CandidateGraph(args.graph)
    held = np.flatnonzero(np.asarray([
        stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold
        for formula in graph.query_formula
    ], dtype=bool)).astype(np.int64)
    if len(held) != 18333:
        raise RuntimeError(f"corrected held registry drifted: {len(held)} != 18333")
    with np.load(args.embeddings, allow_pickle=False) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    stage1_scores = score_embeddings(graph, rows, embeddings)
    getter, spectrum_rows = selected_spectra(graph, held, args.data)
    rrf_scores, naive_audit = build_rrf_graph_scores(
        graph, stage1_scores, held, getter, k=RRF_K,
    )
    with h5py.File(args.data, "r") as handle:
        raw_adduct = np.asarray(handle["adduct"][:])
    query_adduct = np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_adduct[graph.query_row]
    ], dtype=str)
    stage1_metrics, stage1_table = full_metrics(graph, stage1_scores, query_adduct, held)
    rrf_metrics, rrf_table = full_metrics(graph, rrf_scores, query_adduct, held)
    paired, outcome = paired_summary(
        stage1_table, rrf_table, args.bootstrap_resamples,
        args.bootstrap_seed, hypotheses=24,
    )
    prior = json.loads(args.stage1_report.read_text(encoding="utf-8"))
    expected = prior.get("candidate", {}).get("retrieval", {})
    reproduced = {
        "queries": expected.get("queries") == stage1_metrics["retrieval"]["queries"],
        "recall@1": abs(
            float(expected.get("recall@1", float("nan")))
            - float(stage1_metrics["retrieval"]["recall@1"])
        ) <= 1e-12,
        "mrr": abs(
            float(expected.get("mrr", float("nan")))
            - float(stage1_metrics["retrieval"]["mrr"])
        ) <= 1e-7,
    }
    if not all(reproduced.values()):
        raise RuntimeError(f"Stage-1 checkpoint does not reproduce its sealed report: {reproduced}")
    report = {
        "status": "noise_stage1_fixed_rrf_confirmation_complete",
        "recipe": {
            "fusion": "RRF(dreams_stage1, raw_cosine_max, topk_overlap_max)",
            "rrf_k": RRF_K,
            "fragment_tolerance_da": MATCH_TOL_DA,
            "topk_peaks": TOPK,
            "aggregation": "maximum measured-spectrum score within each molecule",
            "self_spectrum_excluded": True,
            "tie_policy": "strict: every score tie with a rival counts against the positive rank",
            "label_blind_scoring": True,
        },
        "held_queries": int(len(held)),
        "stage1_reproduction": reproduced,
        "stage1": stage1_metrics,
        "fixed_rrf": rrf_metrics,
        "fixed_rrf_vs_stage1": paired,
        "naive_spectrum_audit": {
            **naive_audit.__dict__, "loaded_unique_rows": spectrum_rows,
        },
        "score_semantics": {
            "retrieval_micro_macro": "candidate-molecule decision score",
            "pooled_pairwise_auc": (
                "molecule-level RRF score repeated over that molecule's spectrum edges; "
                "reported for completeness, not comparable to NIST20 cosine pair AUROC"
            ),
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embeddings),
            "data_sha256": sha256_file(args.data),
            "stage1_report_sha256": sha256_file(args.stage1_report),
        },
        "claim_limit": (
            "The fixed RRF recipe was selected after inspection of this same MassSpecGym held fold. "
            "This run measures incremental overlap against Stage-1 but is exploratory; only the "
            "independent frozen GNPS panels can confirm transfer. RRF is a reranker, not an encoder gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        outcome.to_csv(staging / "paired_queries.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(
            json.dumps(json_native(report), indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(json_native(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
