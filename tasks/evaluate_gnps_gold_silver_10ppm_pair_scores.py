#!/usr/bin/env python
"""Evaluate arbitrary spectral pair scores on the frozen GNPS benchmark.

Both methods must provide an immutable ``gnps_pair_score_cache``.  The script
uses exactly the same molecule-max aggregation, tie policy, metric family and
formula-cluster paired confidence intervals as the embedding evaluator.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from evaluate_gnps_gold_silver_10ppm_embeddings import (
    graph_from_panel,
    paired_summary,
    rename_pairwise_metric,
)
from gnps_pair_score_cache import PANELS, load_pair_score_cache
from noise_corrected_fullgraph_evaluation import GraphScores, full_metrics


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--baseline-score-cache", type=Path, required=True)
    parser.add_argument("--candidate-score-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    return parser.parse_args()


def graph_scores(graph, pair: np.ndarray) -> GraphScores:
    pair = np.asarray(pair, dtype=np.float32)
    if pair.ndim != 1 or len(pair) != len(graph.pair_candidate_row):
        raise RuntimeError("pair scores do not align to the frozen panel")
    if not np.all(np.isfinite(pair)):
        raise RuntimeError("pair scores are non-finite")
    return GraphScores(
        pair=pair,
        molecule=np.maximum.reduceat(pair, graph.molecule_ptr[:-1]),
    )


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    baseline_report, baseline = load_pair_score_cache(
        args.baseline_score_cache, args.benchmark,
    )
    candidate_report, candidate = load_pair_score_cache(
        args.candidate_score_cache, args.benchmark,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    report: dict[str, object] = {
        "status": "gnps_gold_silver_10ppm_pair_score_evaluation_complete",
        "baseline_method": baseline_report["method"],
        "candidate_method": candidate_report["method"],
        "exact_nist20_replication": False,
        "aggregation": "maximum pair score within each candidate molecule",
        "panels": {},
    }
    try:
        for panel_index, name in enumerate(PANELS):
            graph = graph_from_panel(args.benchmark / f"panel_{name}.npz")
            baseline_scores = graph_scores(graph, baseline[name])
            candidate_scores = graph_scores(graph, candidate[name])
            adduct = np.full(graph.n_queries, "[M+H]+", dtype="U6")
            baseline_metrics, baseline_table = full_metrics(graph, baseline_scores, adduct)
            candidate_metrics, candidate_table = full_metrics(graph, candidate_scores, adduct)
            paired, outcome = paired_summary(
                baseline_table,
                candidate_table,
                args.bootstrap_resamples,
                args.bootstrap_seed + panel_index * 1000,
                hypotheses=24,
            )
            report["panels"][name] = {
                "baseline": rename_pairwise_metric(baseline_metrics, name),
                "candidate": rename_pairwise_metric(candidate_metrics, name),
                "paired": paired,
            }
            outcome.to_csv(
                staging / f"paired_queries_{name}.csv.gz",
                index=False,
                compression="gzip",
            )
        report["claim_limit"] = (
            "Frozen GNPS Gold/Silver identity/formula-disjoint transfer benchmark; "
            "identical graph and aggregation across methods; not an exact NIST20 replication."
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
