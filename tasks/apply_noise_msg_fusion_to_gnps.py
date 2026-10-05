#!/usr/bin/env python
"""Apply the frozen fusion ranker to the sealed GNPS panels.

Every input already exists as a frozen artifact: the article score bundle
(classical channels per sealed edge), the official and Noise V1 embedding
caches, and the joblib-frozen fusion model.  This script only assembles the
identical feature matrix, scores each sealed edge once, and writes a
``gnps_pair_score_cache`` that the existing evaluator consumes unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from gnps_pair_score_cache import PANELS, write_pair_score_cache
from noise_final_core import sha256_file
from noise_msg_pair_evidence_core import MOLECULE_FEATURES, build_molecule_features

ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--official-embeddings", type=Path, required=True)
    parser.add_argument("--v1-embeddings", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = joblib.load(args.model / "fusion_model.joblib")
    model_report = json.loads((args.model / "report.json").read_text(encoding="utf-8"))
    if model_report.get("status") != "NOISE_MSG_FUSION_FROZEN":
        raise RuntimeError("fusion model artifact is not frozen")
    if tuple(frozen.get("feature_names", ())) != tuple(MOLECULE_FEATURES):
        raise RuntimeError("frozen fusion model feature contract drifted")
    with np.load(args.score_bundle, allow_pickle=False) as body:
        methods = list(map(str, body["method_names"]))
        scores = {
            name: np.asarray(body[f"scores_{name}"], dtype=np.float32)
            for name in PANELS
        }
    required_methods = (
        "official_dreams", "noise_v1", "weighted_spectral_entropy",
        "p2b_sqrt_cosine", "p2b_unweighted_entropy",
        "neutral_loss_sqrt_cosine", "p2b_noise_v1_frozen",
    )
    missing = [name for name in required_methods if name not in methods]
    if missing:
        raise RuntimeError(f"score bundle misses frozen methods: {missing}")

    manifest_rows = len(pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"]))
    official_rows, official_embeddings = load_embeddings(
        args.official_embeddings, manifest_rows,
    )
    v1_rows, v1_embeddings = load_embeddings(args.v1_embeddings, manifest_rows)
    if not np.array_equal(official_rows, v1_rows):
        raise RuntimeError("official and V1 embedding registries differ")

    fused: dict[str, np.ndarray] = {}
    for panel in PANELS:
        graph = graph_from_panel(args.benchmark / f"panel_{panel}.npz")
        panel_scores = scores[panel]
        candidate_rows = np.asarray(graph.pair_candidate_row, dtype=np.int64)
        # query_ptr spans molecules; derive the edge-level query prefix.
        query_edge_ptr = np.asarray(graph.molecule_ptr, dtype=np.int64)[
            np.asarray(graph.query_ptr, dtype=np.int64)
        ]
        # Embedding cosines, one query block at a time (bounded memory).
        v1_cosine = np.empty(len(candidate_rows), dtype=np.float32)
        official_cosine = np.empty(len(candidate_rows), dtype=np.float32)
        lookup = {int(row): index for index, row in enumerate(official_rows)}
        for query in range(graph.n_queries):
            left, right = int(query_edge_ptr[query]), int(query_edge_ptr[query + 1])
            rows = candidate_rows[left:right]
            query_row = int(graph.query_row[query])
            v1_cosine[left:right] = (
                v1_embeddings[[lookup[int(r)] for r in rows]] @ v1_embeddings[lookup[query_row]]
            )
            official_cosine[left:right] = (
                official_embeddings[[lookup[int(r)] for r in rows]]
                @ official_embeddings[lookup[query_row]]
            )
        edges = {
            "v1_cosine": v1_cosine,
            "official_cosine": official_cosine,
            "weighted_entropy": panel_scores[methods.index("weighted_spectral_entropy")],
            "sqrt_cosine": panel_scores[methods.index("p2b_sqrt_cosine")],
            "entropy_similarity": panel_scores[methods.index("p2b_unweighted_entropy")],
            "neutral_loss_sqrt_cosine": panel_scores[methods.index("neutral_loss_sqrt_cosine")],
            "p2b_fused": panel_scores[methods.index("p2b_noise_v1_frozen")],
        }
        features = build_molecule_features(
            edges, graph.query_ptr, graph.molecule_ptr,
        )
        molecule_probabilities = frozen["model"].predict_proba(features)[:, 1].astype(np.float32)
        probabilities = np.repeat(
            molecule_probabilities,
            np.diff(np.asarray(graph.molecule_ptr, dtype=np.int64)),
        )
        if len(probabilities) != len(candidate_rows) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("fused GNPS probabilities are non-finite")
        fused[panel] = probabilities
        print(
            f"[fusion-apply] {panel}: {len(probabilities):,} edges scored", flush=True,
        )

    method = {
        "name": "noise_msg_fusion_stack_v1",
        "input_contract": "candidate-molecule max over spectrum-pair features",
        "feature_names": list(frozen["feature_names"]),
        "model": "HistGradientBoostingClassifier (frozen, formula-OOF validated)",
        "gnps_used_in_training": False,
        "sources": {
            "score_bundle_sha256": sha256_file(args.score_bundle),
            "official_embeddings_sha256": sha256_file(args.official_embeddings),
            "v1_embeddings_sha256": sha256_file(args.v1_embeddings),
            "model_report_sha256": sha256_file(args.model / "report.json"),
        },
    }
    write_pair_score_cache(args.output, args.benchmark, method, fused)
    print(json.dumps({"status": "gnps_fusion_pair_scores_complete"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
