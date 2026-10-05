"""Strict graph loader for ChemAware retrieval tasks that do not use MCES."""
from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "feature_names", "features", "pair_candidate_row", "query_ptr",
    "molecule_ptr", "molecule_label", "molecule_ik14", "molecule_formula",
    "query_row", "query_ik14", "query_formula", "query_has_near",
}


class RetrievalGraph:
    """Validated candidate graph with exactly the fields used by direct training."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as body:
            missing = REQUIRED_ARRAYS - set(body.files)
            if missing:
                raise RuntimeError(f"retrieval graph missing arrays: {sorted(missing)}")
            for name in REQUIRED_ARRAYS:
                setattr(self, name, body[name])
        self.feature_names = list(map(str, self.feature_names))
        self.features = np.asarray(self.features, dtype=np.float32)
        self.pair_candidate_row = np.asarray(self.pair_candidate_row, dtype=np.int64)
        self.query_ptr = np.asarray(self.query_ptr, dtype=np.int64)
        self.molecule_ptr = np.asarray(self.molecule_ptr, dtype=np.int64)
        self.molecule_label = np.asarray(self.molecule_label, dtype=np.int8)
        self.molecule_ik14 = np.asarray(self.molecule_ik14, dtype=str)
        self.molecule_formula = np.asarray(self.molecule_formula, dtype=str)
        self.query_row = np.asarray(self.query_row, dtype=np.int64)
        self.query_ik14 = np.asarray(self.query_ik14, dtype=str)
        self.query_formula = np.asarray(self.query_formula, dtype=str)
        self.query_has_near = np.asarray(self.query_has_near, dtype=bool)
        self.n_queries = len(self.query_ptr) - 1
        self._validate()

    def _validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(self.feature_names):
            raise RuntimeError("feature matrix/name mismatch")
        if self.query_ptr.ndim != 1 or self.molecule_ptr.ndim != 1:
            raise RuntimeError("candidate graph pointers must be one-dimensional")
        if self.query_ptr[0] != 0 or self.query_ptr[-1] != len(self.molecule_label):
            raise RuntimeError("query_ptr does not span candidate molecules")
        if self.molecule_ptr[0] != 0 or self.molecule_ptr[-1] != len(self.features):
            raise RuntimeError("molecule_ptr does not span spectrum pairs")
        if len(self.pair_candidate_row) != len(self.features):
            raise RuntimeError("candidate rows do not align to spectrum pairs")
        if not (
            len(self.query_row) == len(self.query_ik14)
            == len(self.query_formula) == len(self.query_has_near) == self.n_queries
        ):
            raise RuntimeError("query metadata is not aligned")
        if not (
            len(self.molecule_label) == len(self.molecule_ik14) == len(self.molecule_formula)
        ):
            raise RuntimeError("molecule metadata is not aligned")
        if np.any(np.diff(self.query_ptr) < 2) or np.any(np.diff(self.molecule_ptr) < 1):
            raise RuntimeError("each query needs >=2 molecules; each molecule >=1 spectrum")
        for left, right in zip(self.query_ptr[:-1], self.query_ptr[1:]):
            labels = self.molecule_label[left:right]
            if labels[0] != 1 or int(labels.sum()) != 1:
                raise RuntimeError("positive molecule must be unique and first")
            if len(set(self.molecule_ik14[left:right])) != int(right - left):
                raise RuntimeError("candidate identities must be unique within each query")

    @property
    def dreams_column(self) -> int:
        try:
            return self.feature_names.index("dreams_similarity")
        except ValueError as error:
            raise RuntimeError("retrieval graph has no dreams_similarity column") from error

    def query_block(self, query: int) -> tuple[slice, np.ndarray, np.ndarray, int]:
        molecule_left, molecule_right = map(int, self.query_ptr[query:query + 2])
        pair_left = int(self.molecule_ptr[molecule_left])
        pair_right = int(self.molecule_ptr[molecule_right])
        local_ptr = self.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        return (
            slice(pair_left, pair_right),
            self.pair_candidate_row[pair_left:pair_right],
            local_ptr.astype(np.int64, copy=False),
            molecule_left,
        )

    def official_molecule_scores(self, query: int) -> np.ndarray:
        pair_slice, _, ptr, _ = self.query_block(query)
        pair_scores = self.features[pair_slice, self.dreams_column]
        return np.maximum.reduceat(pair_scores, ptr[:-1])
