#!/usr/bin/env python
"""Evaluate a multi-method score bundle on both sealed GNPS panels."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from evaluate_gnps_gold_silver_10ppm_embeddings import (
    graph_from_panel,
    paired_summary,
    rename_pairwise_metric,
)
from noise_corrected_fullgraph_evaluation import GraphScores, full_metrics


ROOT = Path(__file__).resolve().parents[1]
PANELS = ("identity_disjoint", "formula_disjoint")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-method", default="official_dreams")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    return parser.parse_args()


def curve_rows(method: str, labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    rows = [
        {"method": method, "curve": "roc", "x": float(x), "y": float(y),
         "threshold": float(t) if np.isfinite(t) else None}
        for x, y, t in zip(fpr, tpr, thresholds)
    ]
    rows.extend(
        {"method": method, "curve": "precision_recall", "x": float(x), "y": float(y),
         "threshold": float(pr_thresholds[index]) if index < len(pr_thresholds) else None}
        for index, (x, y) in enumerate(zip(recall, precision))
    )
    return rows


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.score_bundle, allow_pickle=False) as bundle:
        methods = list(map(str, bundle["method_names"]))
        scores_by_panel = {
            name: np.asarray(bundle[f"scores_{name}"], dtype=np.float32) for name in PANELS
        }
    if len(set(methods)) != len(methods) or args.baseline_method not in methods:
        raise RuntimeError("score bundle has duplicate methods or misses the baseline")
    metadata_path = args.score_bundle.with_suffix(args.score_bundle.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    report = {
        "status": "noise_gnps_article_benchmark_complete",
        "baseline_method": args.baseline_method,
        "methods": methods,
        "score_bundle_metadata": metadata,
        "panels": {},
    }
    try:
        for panel_index, name in enumerate(PANELS):
            graph = graph_from_panel(args.benchmark / f"panel_{name}.npz")
            matrix = scores_by_panel[name]
            if matrix.shape != (len(methods), len(graph.pair_candidate_row)):
                raise RuntimeError(f"score bundle does not align to {name}: {matrix.shape}")
            tables: dict[str, pd.DataFrame] = {}
            panel_report = {"absolute": {}, "vs_official_dreams": {}}
            pair_labels = np.repeat(graph.molecule_label, np.diff(graph.molecule_ptr)).astype(np.int8)
            curve_records: list[dict] = []
            for method_index, method in enumerate(methods):
                pair = matrix[method_index]
                graph_scores = GraphScores(
                    pair=pair,
                    molecule=np.maximum.reduceat(pair, graph.molecule_ptr[:-1]),
                )
                metrics, table = full_metrics(
                    graph, graph_scores, np.full(graph.n_queries, "[M+H]+", dtype="U6"),
                )
                tables[method] = table
                panel_report["absolute"][method] = rename_pairwise_metric(metrics, name)
                curve_records.extend(curve_rows(method, pair_labels, pair))
                table.to_csv(
                    staging / f"queries_{name}_{method}.csv.gz", index=False, compression="gzip",
                )
            baseline = tables[args.baseline_method]
            for method_index, method in enumerate(methods):
                if method == args.baseline_method:
                    continue
                paired, outcomes = paired_summary(
                    baseline, tables[method], args.bootstrap_resamples,
                    args.bootstrap_seed + panel_index * 1000 + method_index * 10,
                    hypotheses=24,
                )
                panel_report["vs_official_dreams"][method] = paired
                outcomes.to_csv(
                    staging / f"paired_{name}_{method}.csv.gz", index=False, compression="gzip",
                )
            panel_report["rankings"] = {
                "recall_at_1": sorted(
                    [
                        {"method": method, "value": block["retrieval"]["recall@1"]}
                        for method, block in panel_report["absolute"].items()
                    ],
                    key=lambda row: (-row["value"], row["method"]),
                ),
                "pooled_pairwise_auroc": sorted(
                    [
                        {"method": method, "value": block["gnps_10ppm_pooled_pairwise"]["auroc"]}
                        for method, block in panel_report["absolute"].items()
                    ],
                    key=lambda row: (-row["value"], row["method"]),
                ),
            }
            pd.DataFrame(curve_records).to_csv(
                staging / f"curves_{name}.csv.gz", index=False, compression="gzip",
            )
            report["panels"][name] = panel_report
        report["claim_limit"] = (
            "Noise/P2b GNPS article benchmark. Methods at different information levels are "
            "reported in separate strata; this is not an exact NIST20 replication."
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
