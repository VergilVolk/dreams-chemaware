"""Minimal candidate-graph projection that never loads evaluation truth."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def scoreblind_graph_from_panel(path: Path) -> SimpleNamespace:
    with np.load(path, allow_pickle=False) as body:
        query_row = np.asarray(body["query_row"], dtype=np.int64)
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(body["molecule_ptr"], dtype=np.int64)
        candidate_row = np.asarray(body["candidate_row"], dtype=np.int64)
    molecule_count = int(query_ptr[-1])
    if len(query_ptr) != len(query_row) + 1 or len(molecule_ptr) != molecule_count + 1:
        raise RuntimeError(f"score-blind panel pointers are malformed: {path}")
    if int(molecule_ptr[-1]) != len(candidate_row):
        raise RuntimeError(f"score-blind candidate axis is malformed: {path}")
    return SimpleNamespace(
        n_queries=len(query_row),
        query_row=query_row,
        query_ptr=query_ptr,
        molecule_ptr=molecule_ptr,
        # Existing scoring helpers use this array only to verify pointer length.
        # Zero placeholders prevent reading the frozen evaluation labels.
        molecule_label=np.zeros(molecule_count, dtype=np.int8),
        pair_candidate_row=candidate_row,
    )
