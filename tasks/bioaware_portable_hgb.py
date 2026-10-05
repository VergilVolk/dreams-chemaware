#!/usr/bin/env python
"""Version-independent NumPy inference for binary sklearn HistGradientBoosting trees."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_binary_hist_gradient_boosting(
    model: object, path: Path, feature_names: list[str], source_sha256: str,
) -> None:
    if getattr(model, "n_trees_per_iteration_", None) != 1:
        raise RuntimeError("portable exporter supports binary one-tree-per-iteration HGB only")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise RuntimeError("portable exporter requires classes [0, 1]")
    if int(getattr(model, "n_features_in_", -1)) != len(feature_names):
        raise RuntimeError("portable exporter feature count mismatch")
    predictors = [iteration[0] for iteration in model._predictors]
    if not predictors:
        raise RuntimeError("portable exporter received no trees")
    if any(np.any(predictor.nodes["is_categorical"]) for predictor in predictors):
        raise RuntimeError("portable exporter does not support categorical tree nodes")
    offsets = [0]
    for predictor in predictors:
        offsets.append(offsets[-1] + len(predictor.nodes))
    nodes = np.concatenate([predictor.nodes for predictor in predictors])
    _atomic_npz(
        path,
        format=np.asarray("bioaware_binary_hgb_v1"),
        feature_names=np.asarray(feature_names),
        source_joblib_sha256=np.asarray(source_sha256),
        baseline=np.asarray(float(np.ravel(model._baseline_prediction)[0]), np.float64),
        tree_offsets=np.asarray(offsets, np.int64),
        value=nodes["value"].astype(np.float64),
        feature_idx=nodes["feature_idx"].astype(np.int64),
        num_threshold=nodes["num_threshold"].astype(np.float64),
        missing_go_to_left=nodes["missing_go_to_left"].astype(np.uint8),
        left=nodes["left"].astype(np.int64),
        right=nodes["right"].astype(np.int64),
        is_leaf=nodes["is_leaf"].astype(np.uint8),
    )


class PortableBinaryHGB:
    def __init__(self, path: Path, expected_features: list[str]):
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["format"].item()) != "bioaware_binary_hgb_v1":
                raise RuntimeError("unknown portable HGB format")
            self.feature_names = payload["feature_names"].astype(str).tolist()
            self.source_joblib_sha256 = str(payload["source_joblib_sha256"].item())
            self.baseline = float(payload["baseline"].item())
            self.tree_offsets = payload["tree_offsets"].astype(np.int64)
            self.value = payload["value"].astype(np.float64)
            self.feature_idx = payload["feature_idx"].astype(np.int64)
            self.num_threshold = payload["num_threshold"].astype(np.float64)
            self.missing_go_to_left = payload["missing_go_to_left"].astype(bool)
            self.left = payload["left"].astype(np.int64)
            self.right = payload["right"].astype(np.int64)
            self.is_leaf = payload["is_leaf"].astype(bool)
        if self.feature_names != list(expected_features):
            raise RuntimeError(
                f"portable HGB feature order mismatch: {self.feature_names} != {expected_features}"
            )
        if self.tree_offsets[0] != 0 or self.tree_offsets[-1] != len(self.value):
            raise RuntimeError("portable HGB tree offsets are invalid")
        if np.any(np.diff(self.tree_offsets) <= 0):
            raise RuntimeError("portable HGB contains an empty tree")

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"portable HGB input shape mismatch: {x.shape}")
        raw = np.full(len(x), self.baseline, dtype=np.float64)
        sample_index = np.arange(len(x))
        for start, end in zip(self.tree_offsets[:-1], self.tree_offsets[1:]):
            positions = np.zeros(len(x), dtype=np.int64)
            for _ in range(128):
                indices = start + positions
                active = ~self.is_leaf[indices]
                if not np.any(active):
                    break
                active_samples = sample_index[active]
                active_nodes = indices[active]
                feature = self.feature_idx[active_nodes]
                observed = x[active_samples, feature]
                go_left = np.where(
                    np.isnan(observed),
                    self.missing_go_to_left[active_nodes],
                    observed <= self.num_threshold[active_nodes],
                )
                positions[active] = np.where(
                    go_left, self.left[active_nodes], self.right[active_nodes]
                )
            else:
                raise RuntimeError("portable HGB traversal exceeded 128 levels")
            leaf_indices = start + positions
            if np.any(leaf_indices < start) or np.any(leaf_indices >= end):
                raise RuntimeError("portable HGB traversal escaped a tree")
            if not np.all(self.is_leaf[leaf_indices]):
                raise RuntimeError("portable HGB traversal did not terminate at leaves")
            raw += self.value[leaf_indices]
        return raw

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        raw = self.decision_function(values)
        probability = np.empty_like(raw)
        positive = raw >= 0
        probability[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
        exponential = np.exp(raw[~positive])
        probability[~positive] = exponential / (1.0 + exponential)
        return np.column_stack((1.0 - probability, probability))
