#!/usr/bin/env python
"""Independent fixed-RRF confirmation on both frozen GNPS panels."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from encode_gnps_gold_silver_10ppm_checkpoint import load_selected_spectra, required_rows
from evaluate_gnps_gold_silver_10ppm_embeddings import (
    graph_from_panel,
    paired_summary,
    rename_pairwise_metric,
)
from noise_corrected_fullgraph_evaluation import full_metrics, score_embeddings
from noise_stage1_rrf_core import MATCH_TOL_DA, RRF_K, TOPK, build_rrf_graph_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.benchmark / "spectra.mgf",
        args.benchmark / "panel_identity_disjoint.npz",
        args.benchmark / "panel_formula_disjoint.npz",
        args.stage1_embeddings,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected = required_rows(args.benchmark)
    loaded = load_selected_spectra(args.benchmark / "spectra.mgf", selected)
    spectrum_index = {int(row): position for position, row in enumerate(selected)}

    def spectrum(row: int):
        peaks, _precursor = loaded[spectrum_index[int(row)]]
        valid = (peaks[0] > 0) & (peaks[1] > 0)
        return (
            np.ascontiguousarray(peaks[0, valid]),
            np.ascontiguousarray(peaks[1, valid]),
        )

    with np.load(args.stage1_embeddings, allow_pickle=False) as body:
        embedding_rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if not np.array_equal(embedding_rows, selected):
        raise RuntimeError("GNPS embedding row registry differs from frozen two-panel union")
    report = {
        "status": "gnps_stage1_fixed_rrf_confirmation_complete",
        "recipe": {
            "fusion": "RRF(dreams_stage1, raw_cosine_max, topk_overlap_max)",
            "rrf_k": RRF_K,
            "fragment_tolerance_da": MATCH_TOL_DA,
            "topk_peaks": TOPK,
            "selected_without_GNPS_labels": True,
            "tie_policy": "strict: ties with any rival count against positive rank",
        },
        "panels": {},
        "score_semantics": {
            "pooled_pairwise_auc": (
                "molecule-level RRF score repeated over spectrum edges; not a NIST20 "
                "cosine pair AUROC and not directly comparable to 0.85"
            ),
        },
        "claim_limit": (
            "Independent GNPS Gold/Silver identity/formula-disjoint confirmation of a "
            "recipe frozen on MassSpecGym; RRF is a reranker and does not alter embeddings."
        ),
    }
    outcomes: dict[str, pd.DataFrame] = {}
    for panel_index, name in enumerate(("identity_disjoint", "formula_disjoint")):
        graph = graph_from_panel(args.benchmark / f"panel_{name}.npz")
        queries = np.arange(graph.n_queries, dtype=np.int64)
        stage1_scores = score_embeddings(graph, embedding_rows, embeddings)
        rrf_scores, audit = build_rrf_graph_scores(
            graph, stage1_scores, queries, spectrum, k=RRF_K,
        )
        adduct = np.full(graph.n_queries, "[M+H]+", dtype="U6")
        stage1_metrics, stage1_table = full_metrics(graph, stage1_scores, adduct)
        rrf_metrics, rrf_table = full_metrics(graph, rrf_scores, adduct)
        paired, outcome = paired_summary(
            stage1_table, rrf_table, args.bootstrap_resamples,
            args.bootstrap_seed + panel_index * 1000, hypotheses=24,
        )
        report["panels"][name] = {
            "stage1": rename_pairwise_metric(stage1_metrics, name),
            "fixed_rrf": rename_pairwise_metric(rrf_metrics, name),
            "fixed_rrf_vs_stage1": paired,
            "naive_spectrum_audit": audit.__dict__,
        }
        outcomes[name] = outcome
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for name, outcome in outcomes.items():
            outcome.to_csv(
                staging / f"paired_queries_{name}.csv.gz", index=False, compression="gzip",
            )
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
