#!/usr/bin/env python
"""Deterministic unit checks for BioAware Action Atlas v3."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

try:
    from develop_bioaware_action_atlas_v3 import (
        NETWORK_BLOCK_FOR_NULL,
        aggregate_rich_edge,
        degree_conditioned_permutation,
    )
except ModuleNotFoundError:
    from tasks.develop_bioaware_action_atlas_v3 import (
        NETWORK_BLOCK_FOR_NULL,
        aggregate_rich_edge,
        degree_conditioned_permutation,
    )


def test_edge_aggregation(tmp: Path) -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q", "q", "q"],
            "candidate_id": ["c", "c", "c"],
            "maximum_depth": [2, 2, 3],
            "identity_paths": [2, 0, 9],
            "complete_ms2_paths": [1, 0, 9],
            "node_combinations_evaluated": [4, 0, 9],
            "path_truncated": [False, False, True],
            "best_bottleneck": [0.8, np.nan, 1.0],
            "median_bottleneck": [0.6, np.nan, 1.0],
        }
    )
    path = tmp / "edge.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    result = aggregate_rich_edge(path, "edge0").iloc[0]
    assert np.isclose(result["edge0_path_available_fraction"], 0.5)
    assert np.isclose(result["edge0_complete_path_fraction"], 0.5)
    assert np.isclose(result["edge0_completion_ratio_mean"], 0.25)
    assert np.isclose(result["edge0_best_bottleneck_mean"], 0.4)
    assert np.isclose(result["edge0_median_bottleneck_mean"], 0.3)
    assert np.isclose(result["edge0_bottleneck_spread_mean"], 0.1)


def test_conditioned_null() -> None:
    rows = []
    for index in range(20):
        row = {
            "unit_id": "u",
            "spectral_score": index / 100,
            "known_log_degree": float(index // 4),
            "known_path_fraction": 1.0,
            "edge0_complete_fraction": 1.0,
            "edge1_complete_fraction": 1.0,
        }
        for offset, name in enumerate(NETWORK_BLOCK_FOR_NULL):
            row.setdefault(name, float(index * 100 + offset))
        rows.append(row)
    frame = pd.DataFrame(rows)
    permuted, audit = degree_conditioned_permutation(
        frame, np.random.default_rng(7)
    )
    assert np.array_equal(permuted["spectral_score"], frame["spectral_score"])
    assert np.array_equal(permuted["known_log_degree"], frame["known_log_degree"])
    before = np.sort(frame[NETWORK_BLOCK_FOR_NULL].to_numpy(), axis=0)
    after = np.sort(permuted[NETWORK_BLOCK_FOR_NULL].to_numpy(), axis=0)
    assert np.array_equal(before, after)
    assert audit["moved_rows"] > 0


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        test_edge_aggregation(Path(directory))
    test_conditioned_null()
    print("[test_bioaware_action_atlas_v3] PASS")


if __name__ == "__main__":
    main()
