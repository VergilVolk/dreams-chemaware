"""Audit whether the ChemAware retrieval graph supports multi-positive training.

The deployable retrieval score aggregates reference spectra within a candidate
molecule.  Before describing an auxiliary objective as "multi-positive", this
audit counts both the positive references present in each query candidate group
and every graph-reachable reference row belonging to the same molecular
identity.  The report is deliberately independent of phenotype and model
outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            ROOT / "data/validation/chemaware_positive_reference_multiplicity_audit_v1/report.json"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in sorted(Counter(values.tolist()).items())}


def main() -> None:
    args = arguments()
    graph = np.load(args.graph, allow_pickle=True)
    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    candidate_rows = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
    molecule_identity = np.asarray(graph["molecule_ik14"]).astype(str)
    query_identity = np.asarray(graph["query_ik14"]).astype(str)
    query_rows = np.asarray(graph["query_row"], dtype=np.int64)

    direct_counts = np.asarray([
        molecule_ptr[query_ptr[index] + 1] - molecule_ptr[query_ptr[index]]
        for index in range(len(query_rows))
    ], dtype=np.int64)

    identity_rows: dict[str, set[int]] = defaultdict(set)
    for molecule_index, identity in enumerate(molecule_identity):
        left = int(molecule_ptr[molecule_index])
        right = int(molecule_ptr[molecule_index + 1])
        identity_rows[identity].update(map(int, candidate_rows[left:right]))
    global_counts = np.asarray([
        len(identity_rows[identity] - {int(query_row)})
        for identity, query_row in zip(query_identity, query_rows)
    ], dtype=np.int64)

    report = {
        "status": "chemaware_positive_reference_multiplicity_audit_complete",
        "graph": str(args.graph.resolve()),
        "graph_sha256": sha256_file(args.graph),
        "queries": int(len(query_rows)),
        "unique_query_identities": int(len(set(query_identity.tolist()))),
        "direct_positive_candidate_reference_count": {
            "histogram": histogram(direct_counts),
            "mean": float(np.mean(direct_counts)),
            "maximum": int(np.max(direct_counts)),
            "queries_with_at_least_two": int(np.sum(direct_counts >= 2)),
        },
        "global_same_identity_reference_count_after_query_exclusion": {
            "histogram": histogram(global_counts),
            "mean": float(np.mean(global_counts)),
            "maximum": int(np.max(global_counts)),
            "queries_with_at_least_two": int(np.sum(global_counts >= 2)),
            "query_fraction_with_at_least_two": float(np.mean(global_counts >= 2)),
        },
        "decision": (
            "The current graph does not support broad multi-positive training: every direct "
            "positive candidate group contains one reference spectrum, and only a small minority "
            "of queries have multiple graph-reachable same-identity references. An objective "
            "using the current batch positives must be named a positive-pair increment unless a "
            "separate reference pool is constructed and its sparse coverage is reported."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
