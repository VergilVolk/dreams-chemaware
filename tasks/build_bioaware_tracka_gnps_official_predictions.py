#!/usr/bin/env python
"""Build unified-benchmark official-DreaMS predictions from GNPS embeddings.

Input is an embedding archive (``rows`` aligned to the benchmark manifest,
``embeddings`` L2-renormalized here) produced by
``tasks/encode_gnps_gold_silver_10ppm_checkpoint.py``.  For every query the
candidate-molecule score is the maximum cosine between the query embedding and
the embeddings of the molecule's frozen reference rows, matching the deployment
molecule-max rule.  No benchmark label is read beyond the frozen panel fields.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def build_frames(
    panel_path: Path, embeddings_path: Path, source_label: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    panel = np.load(panel_path, allow_pickle=False)
    with np.load(embeddings_path, allow_pickle=False) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    row_position = {int(row): index for index, row in enumerate(rows)}
    query_ptr = panel["query_ptr"]
    molecule_ptr = panel["molecule_ptr"]
    candidate_row = panel["candidate_row"]
    n_queries = len(panel["query_row"])

    def embedding_of(row_index: int) -> np.ndarray:
        try:
            return embeddings[row_position[int(row_index)]]
        except KeyError as error:
            raise RuntimeError(
                f"embedding archive misses manifest row {row_index} required by the panel"
            ) from error

    query_slot = np.repeat(np.arange(n_queries, dtype=np.int64), np.diff(query_ptr))
    records: list[dict[str, object]] = []
    for molecule_slot in range(len(panel["molecule_ik14"])):
        left, right = int(molecule_ptr[molecule_slot]), int(molecule_ptr[molecule_slot + 1])
        if right <= left:
            continue
        query_index = int(query_slot[molecule_slot])
        query_vector = embedding_of(panel["query_row"][query_index])
        best = -1.0
        for position in range(left, right):
            reference_vector = embedding_of(candidate_row[position])
            cosine = float(np.dot(query_vector, reference_vector))
            if cosine > best:
                best = cosine
        records.append({
            "query_id": str(query_index),
            "candidate_id": str(panel["molecule_ik14"][molecule_slot]),
            "is_truth": int(bool(panel["molecule_label"][molecule_slot])),
            "formula_cluster": str(panel["query_formula"][query_index]),
            "source": source_label,
            "polarity": "positive",
            "near_query": bool(panel["near_query"][query_index]),
            "score": best,
        })
    frame = pd.DataFrame(records)
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("panel exposes duplicate query/identity candidate slots")
    manifest = frame[[
        "query_id", "candidate_id", "is_truth", "formula_cluster",
        "source", "polarity", "near_query",
    ]]
    predictions = frame[["query_id", "candidate_id", "score"]]
    summary = {
        "queries": int(frame["query_id"].nunique()),
        "candidate_rows": int(len(frame)),
        "embedding_rows": int(len(rows)),
        "panel": str(panel_path),
        "embeddings": str(embeddings_path),
        "method": "official_dreams",
        "aggregation": "molecule_max",
    }
    return manifest, predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-file", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--expected-candidate-rows", type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite official prediction output: {output}")
    output.mkdir(parents=True)

    manifest, predictions, summary = build_frames(
        args.panel_file.resolve(), args.embeddings.resolve(), args.source_label
    )
    if args.expected_queries is not None and summary["queries"] != args.expected_queries:
        raise RuntimeError(
            f"query mismatch: expected {args.expected_queries}, observed {summary['queries']}"
        )
    if (
        args.expected_candidate_rows is not None
        and summary["candidate_rows"] != args.expected_candidate_rows
    ):
        raise RuntimeError(
            "candidate-row mismatch: expected "
            f"{args.expected_candidate_rows}, observed {summary['candidate_rows']}"
        )
    manifest.to_csv(output / "candidate_manifest.csv.gz", index=False, compression="gzip")
    predictions.to_csv(
        output / "predictions_official_dreams.csv.gz", index=False, compression="gzip"
    )
    payload = {"status": "bioaware_tracka_gnps_official_prediction_inputs", **summary}
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(output / "report.json")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
