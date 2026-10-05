#!/usr/bin/env python
"""Fail-closed validator for the B47 truth-blind DreaMS embedding cache."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    paths = {
        "report": directory / "report.json",
        "query_embeddings": directory / "query_embeddings.npy",
        "reference_embeddings": directory / "reference_embeddings.npy",
        "reference_rows": directory / "reference_rows.npy",
        "query_index": directory / "query_index.csv",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b47_truthblind_embeddings_complete":
        raise RuntimeError("unexpected B47 embedding status")
    if not report.get("formal") or not report.get("pass_to_truthblind_seed_construction"):
        raise RuntimeError("B47 embedding cache did not pass")
    expected_contract = {
        "one_shared_official_encoder": True,
        "truth_opened": False,
        "phenotype_used": False,
        "algorithm_outputs_opened": False,
        "candidate_scores_computed": False,
        "seed_selection_performed": False,
        "model_fitted": False,
        "P2b_used": False,
    }
    if report.get("contracts") != expected_contract:
        raise RuntimeError("B47 embedding contracts changed")
    provenance = report.get("provenance", {})
    for key in ("query_embeddings", "reference_embeddings", "reference_rows", "query_index"):
        if provenance.get(f"{key}_sha256") != sha256_file(paths[key]):
            raise RuntimeError(f"B47 output SHA256 mismatch: {key}")

    query = np.load(paths["query_embeddings"], mmap_mode="r", allow_pickle=False)
    reference = np.load(paths["reference_embeddings"], mmap_mode="r", allow_pickle=False)
    rows = np.load(paths["reference_rows"], mmap_mode="r", allow_pickle=False)
    expected_query_shape = (
        int(report["queries"]["rows"]), int(report["queries"]["dimension"])
    )
    expected_reference_shape = (
        int(report["references"]["rows"]), int(report["references"]["dimension"])
    )
    if query.shape != expected_query_shape or reference.shape != expected_reference_shape:
        raise RuntimeError("B47 embedding array shape mismatch")
    if rows.shape != (expected_reference_shape[0],):
        raise RuntimeError("B47 reference-row shape mismatch")
    if rows.dtype.kind not in "iu" or np.any(np.diff(rows) <= 0):
        raise RuntimeError("B47 reference rows are not strictly increasing integers")
    if query.dtype != np.float32 or reference.dtype != np.float32:
        raise RuntimeError("B47 embeddings are not float32")

    with paths["query_index"].open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != expected_query_shape[0]:
        raise RuntimeError("B47 query index length mismatch")
    if len({row["query_id"] for row in index_rows}) != len(index_rows):
        raise RuntimeError("B47 query index contains duplicate IDs")
    positions = np.asarray([int(row["embedding_position"]) for row in index_rows])
    if not np.array_equal(positions, np.arange(len(index_rows))):
        raise RuntimeError("B47 query embedding positions are not canonical")
    print(
        f"[validate_bioaware_b47_truthblind_embeddings] PASS "
        f"queries={query.shape[0]:,} references={reference.shape[0]:,} dim={query.shape[1]}"
    )


if __name__ == "__main__":
    main()

