#!/usr/bin/env python
"""Build the unified-benchmark manifest and baseline predictions from the
corrected MassSpecGym development graph.

This is the Track A implementation gate.  The official DreaMS molecule-max
aggregation must reproduce the frozen corrected-graph Recall@1 exactly, and
the mean aggregation must reproduce the known large degradation reported in
the B47-U1 result.  No new truth is opened: the corrected graph is an already
opened development denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_u1_core import CandidateGraph  # noqa: E402

FROZEN_OFFICIAL_RECALL_AT_1 = 0.928760
EXPECTED_MEAN_DELTA_PP_RANGE = (-12.0, -5.0)


def molecule_reduce(pair_score: np.ndarray, molecule_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.maximum.reduceat(pair_score, molecule_ptr[:-1])
    count = np.diff(molecule_ptr)
    total = np.add.reduceat(pair_score, molecule_ptr[:-1])
    return maximum.astype(np.float64), (total / np.maximum(count, 1)).astype(np.float64)


def build_frames(graph: CandidateGraph) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_score = graph.features[:, graph.dreams_column].astype(np.float64)
    maximum, mean = molecule_reduce(pair_score, graph.molecule_ptr)

    query_ptr = graph.query_ptr
    n_queries = graph.n_queries
    query_ids = np.empty(n_queries, dtype=object)
    starts = query_ptr[:-1]
    ends = query_ptr[1:]
    sizes = ends - starts
    query_id_per_row = np.repeat(np.arange(n_queries, dtype=np.int64), sizes)

    rows = {
        "query_id": query_id_per_row.astype(str),
        "candidate_id": graph.molecule_ik14.astype(str),
        "is_truth": graph.molecule_label.astype(int),
        "formula_cluster": graph.query_formula[query_id_per_row].astype(str),
        "source": np.full(len(query_id_per_row), "massspecgym_corrected_graph_dev", dtype=object),
        "polarity": np.full(len(query_id_per_row), "not_recorded", dtype=object),
        "near_query": graph.query_has_near[query_id_per_row].astype(bool),
    }
    manifest = pd.DataFrame(rows)

    duplicate_slots = int(manifest.duplicated(["query_id", "candidate_id"]).sum())
    if duplicate_slots:
        raise RuntimeError(
            f"corrected graph exposes {duplicate_slots} duplicate query/identity "
            "candidate slots; the manifest contract requires unique candidates"
        )

    predictions_max = pd.DataFrame({
        "query_id": rows["query_id"],
        "candidate_id": rows["candidate_id"],
        "score": maximum,
    })
    predictions_mean = pd.DataFrame({
        "query_id": rows["query_id"],
        "candidate_id": rows["candidate_id"],
        "score": mean,
    })
    return manifest, predictions_max, predictions_mean


def strict_recall_at_1(manifest: pd.DataFrame, score: np.ndarray) -> float:
    frame = manifest.copy()
    frame["score"] = score
    hits = []
    for _, group in frame.groupby("query_id", sort=False):
        truth_score = float(group.loc[group["is_truth"].astype(bool), "score"].iloc[0])
        negatives = group.loc[~group["is_truth"].astype(bool), "score"].to_numpy(float)
        hits.append(float(np.sum(negatives >= truth_score) == 0))
    return float(np.mean(hits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    graph_path = args.graph_dir.resolve() / "candidate_graph.npz"
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite replay output: {output}")
    output.mkdir(parents=True)

    graph = CandidateGraph(graph_path)
    manifest, predictions_max, predictions_mean = build_frames(graph)

    replay = strict_recall_at_1(manifest, predictions_max["score"].to_numpy(float))
    mean_replay = strict_recall_at_1(manifest, predictions_mean["score"].to_numpy(float))
    delta_pp = (mean_replay - replay) * 100.0

    manifest.to_csv(output / "candidate_manifest.csv.gz", index=False, compression="gzip")
    predictions_max.to_csv(
        output / "predictions_official_dreams.csv.gz", index=False, compression="gzip"
    )
    predictions_mean.to_csv(
        output / "predictions_dreams_mean.csv.gz", index=False, compression="gzip"
    )

    report = {
        "status": "bioaware_tracka_corrected_replay_inputs",
        "graph_path": str(graph_path),
        "queries": int(len(manifest["query_id"].unique())),
        "candidate_rows": int(len(manifest)),
        "truth_rows": int(manifest["is_truth"].sum()),
        "official_max_recall_at_1_replay": replay,
        "frozen_official_recall_at_1": FROZEN_OFFICIAL_RECALL_AT_1,
        "official_replay_matches_frozen": bool(
            abs(replay - FROZEN_OFFICIAL_RECALL_AT_1) < 5e-7
        ),
        "mean_recall_at_1_replay": mean_replay,
        "mean_minus_max_delta_pp": delta_pp,
        "mean_degradation_in_expected_range": bool(
            EXPECTED_MEAN_DELTA_PP_RANGE[0] <= delta_pp <= EXPECTED_MEAN_DELTA_PP_RANGE[1]
        ),
        "gate": "PASS"
        if abs(replay - FROZEN_OFFICIAL_RECALL_AT_1) < 5e-7
        and EXPECTED_MEAN_DELTA_PP_RANGE[0] <= delta_pp <= EXPECTED_MEAN_DELTA_PP_RANGE[1]
        else "FAIL",
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(output / "replay_gate.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
