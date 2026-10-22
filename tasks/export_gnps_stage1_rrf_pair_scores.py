#!/usr/bin/env python
"""Export the already-fixed Stage-1 RRF recipe on the frozen GNPS edge axis.

This is an adapter, not a new reranker.  It reuses the constants and scoring
implementation from the historical confirmation experiment and writes the
standard immutable pair-score cache consumed by the unified integration job.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from encode_gnps_gold_silver_10ppm_checkpoint import load_selected_spectra, required_rows
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel
from gnps_pair_score_cache import PANELS, write_pair_score_cache
from noise_corrected_fullgraph_evaluation import score_embeddings
from noise_stage1_rrf_core import MATCH_TOL_DA, RRF_K, TOPK, build_rrf_graph_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = required_rows(args.benchmark)
    loaded = load_selected_spectra(args.benchmark / "spectra.mgf", selected)
    spectrum_index = {int(row): pos for pos, row in enumerate(selected)}

    def spectrum(row: int):
        peaks, _ = loaded[spectrum_index[int(row)]]
        valid = (peaks[0] > 0) & (peaks[1] > 0)
        return np.ascontiguousarray(peaks[0, valid]), np.ascontiguousarray(peaks[1, valid])

    with np.load(args.stage1_embeddings, allow_pickle=False) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if not np.array_equal(rows, selected):
        raise RuntimeError("Stage-1 embedding rows differ from the frozen two-panel union")

    scores: dict[str, np.ndarray] = {}
    for panel in PANELS:
        graph = graph_from_panel(args.benchmark / f"panel_{panel}.npz")
        stage1 = score_embeddings(graph, rows, embeddings)
        rrf, _ = build_rrf_graph_scores(graph, stage1, np.arange(graph.n_queries), spectrum, k=RRF_K)
        scores[panel] = np.asarray(rrf.pair, dtype=np.float32)

    write_pair_score_cache(
        args.output,
        args.benchmark,
        {
            "name": "noise_rrf_1_7",
            "status": "retired_audit_only",
            "recipe": "RRF(stage1, raw_cosine_max, topk_overlap_max)",
            "rrf_k": RRF_K,
            "fragment_tolerance_da": MATCH_TOL_DA,
            "topk_peaks": TOPK,
            "qualification": "historical fixed recipe; included only for explicit ablation",
        },
        scores,
    )
    print(f"GNPS_STAGE1_RRF_PAIR_CACHE_COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
