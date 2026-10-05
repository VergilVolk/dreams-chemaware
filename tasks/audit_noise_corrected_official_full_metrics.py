"""Freeze the complete official baseline on the corrected noise candidate graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from noise_corrected_fullgraph_evaluation import full_metrics, official_scores
from noise_final_core import CandidateGraph, sha256_file, stable_fold


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    manifest_path = args.source_manifest_dir / "manifest.npz"
    manifest_report_path = args.source_manifest_dir / "manifest.json"
    for path in (graph_path, graph_report_path, manifest_path, manifest_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    if (
        graph_report.get("status") != "noise_corrected_candidate_graph_complete"
        or graph_report.get("formal_training_authorized") is not True
        or graph_report.get("provenance", {}).get("candidate_graph_sha256")
        != sha256_file(graph_path)
        or graph_report.get("provenance", {}).get("source_manifest_sha256")
        != sha256_file(manifest_path)
        or manifest_report.get("data_contract") != "train_primary_all_p3_disjoint_v1"
    ):
        raise RuntimeError("corrected graph/source manifest provenance is invalid")
    graph = CandidateGraph(graph_path)
    with np.load(manifest_path, allow_pickle=False) as body:
        source_query_row = np.asarray(body["query_row"], dtype=np.int64)
        query_adduct = np.asarray(body["query_adduct"], dtype=str)
    if not np.array_equal(source_query_row, graph.query_row):
        raise RuntimeError("source manifest query order differs from corrected graph")
    metrics, table = full_metrics(graph, official_scores(graph), query_adduct=query_adduct)
    rank = table["rank"].to_numpy(np.int64)
    folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed)
        for value in graph.query_formula
    ], dtype=np.int8)
    report = {
        "status": "noise_corrected_official_full_metrics_complete",
        "formal": True,
        "development_graph_only": True,
        "data_contract": "train_primary_all_p3_disjoint_v1",
        "metrics": metrics,
        "error_counts": {
            "errors": int(np.sum(rank != 1)),
            "near_errors": int(np.sum((rank != 1) & table["near"].to_numpy(bool))),
        },
        "formula_folds": {
            str(fold): {
                "queries": int(np.sum(folds == fold)),
                "formulas": int(len(np.unique(graph.query_formula[folds == fold]))),
                "recall@1": float(np.mean(rank[folds == fold] == 1)),
            }
            for fold in range(5)
        },
        "contracts": {
            "all_corrected_graph_queries_evaluated": bool(len(table) == graph.n_queries),
            "molecule_score_is_spectrum_max": True,
            "strict_ties_count_against_positive": True,
            "same_adduct_strict_10ppm_spectrum_edges": True,
            "massspecgym_pairwise_not_nist20_replication": True,
            "action_outcomes_consumed": False,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_manifest_report_sha256": sha256_file(manifest_report_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Official baseline on the corrected train-side development graph; this is "
            "not a fine-tuned checkpoint result and not a P3 test result."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_official_metrics_", dir=args.output_dir.parent))
    try:
        table.to_csv(staging / "per_query.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
