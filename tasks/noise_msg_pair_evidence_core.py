"""Pure helpers for the formula-isolated molecule-evidence fusion stack.

Everything here is deterministic and label-blind at the edge level: an edge
feature never reads ``molecule_label``.  Labels are joined only inside the
trainer, and folds are assigned by molecular formula so no formula family is
shared between training and validation molecules.
"""
from __future__ import annotations

import numpy as np

MOLECULE_FEATURES = (
    "v1_cosine",
    "official_cosine",
    "weighted_entropy",
    "sqrt_cosine",
    "entropy_similarity",
    "neutral_loss_sqrt_cosine",
    "p2b_fused",
    "v1_percentile",
    "official_percentile",
    "wse_percentile",
    "sqrt_percentile",
    "entropy_percentile",
    "neutral_loss_percentile",
    "p2b_percentile",
    "query_molecules",
    "query_edges",
    "v1_top1_gap",
    "wse_top1_gap",
    "p2b_top1_gap",
    "v1_wse_top1_agree",
    "v1_p2b_top1_agree",
    "log_query_edges",
)


def _percentile_within_groups(values: np.ndarray, group_ptr: np.ndarray) -> np.ndarray:
    """Percentile of each value inside its contiguous group, ties averaged."""
    values = np.asarray(values, dtype=np.float64)
    group_ptr = np.asarray(group_ptr, dtype=np.int64)
    output = np.empty(len(values), dtype=np.float64)
    for left, right in zip(group_ptr[:-1], group_ptr[1:]):
        block = values[left:right]
        order = np.argsort(block, kind="stable")
        ranks = np.empty(len(block), dtype=np.float64)
        ranks[order] = np.arange(len(block), dtype=np.float64)
        # Average ranks of tied values so identical scores share a percentile.
        sorted_block = block[order]
        start = 0
        while start < len(block):
            stop = start + 1
            while stop < len(block) and sorted_block[stop] == sorted_block[start]:
                stop += 1
            if stop - start > 1:
                average = (ranks[order[start:stop]].sum()) / (stop - start)
                ranks[order[start:stop]] = average
            start = stop
        maximum = max(len(block) - 1, 1)
        output[left:right] = ranks / maximum
    return output


def _molecule_max(value: np.ndarray, group_ptr: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(
        np.asarray(value, dtype=np.float64), np.asarray(group_ptr, dtype=np.int64)[:-1],
    )


def _unique_top(values: np.ndarray) -> int:
    maximum = float(np.max(values))
    winners = np.flatnonzero(values == maximum)
    return int(winners[0]) if len(winners) == 1 else -1


def query_balanced_molecule_weights(
    molecule_label: np.ndarray,
    query_ptr: np.ndarray,
) -> np.ndarray:
    """Give every query equal dose and balance its positive/negative sides."""
    molecule_label = np.asarray(molecule_label, dtype=np.int8)
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        query_ptr.ndim != 1 or len(query_ptr) < 2
        or query_ptr[0] != 0 or query_ptr[-1] != len(molecule_label)
        or np.any(np.diff(query_ptr) < 2)
    ):
        raise ValueError("invalid query/molecule label ledger")
    weights = np.empty(len(molecule_label), dtype=np.float64)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        block = molecule_label[left:right]
        if int(block.sum()) != 1 or np.any(~np.isin(block, (0, 1))):
            raise RuntimeError("query must contain exactly one positive molecule")
        weights[left:right] = 0.5 / float(len(block) - 1)
        weights[left + int(np.flatnonzero(block == 1)[0])] = 0.5
    return weights


def build_molecule_features(
    edges: dict[str, np.ndarray],
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
) -> np.ndarray:
    """Aggregate pair evidence and assemble one feature row per molecule.

    Training and evaluation both rank candidate molecules after max pooling
    their reference spectra.  Aggregating before fitting prevents molecules
    with many spectra from receiving more supervision merely because their
    library coverage is larger.
    """
    required = ("v1_cosine", "official_cosine", "weighted_entropy",
                "sqrt_cosine", "entropy_similarity",
                "neutral_loss_sqrt_cosine", "p2b_fused")
    missing = [name for name in required if name not in edges]
    if missing:
        raise RuntimeError(f"pair evidence misses edge scores: {missing}")
    edge_count = len(edges["v1_cosine"])
    for name in required:
        if len(edges[name]) != edge_count or not np.all(np.isfinite(edges[name])):
            raise RuntimeError(f"edge score {name} is malformed")
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    molecule_ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if (
        molecule_ptr[0] != 0 or molecule_ptr[-1] != edge_count
        or query_ptr[0] != 0 or query_ptr[-1] != len(molecule_ptr) - 1
        or np.any(np.diff(query_ptr) < 1) or np.any(np.diff(molecule_ptr) < 1)
    ):
        raise RuntimeError("graph prefix arrays do not span molecules and edges")

    base_names = required
    base = np.stack([
        _molecule_max(np.asarray(edges[name], dtype=np.float64), molecule_ptr)
        for name in base_names
    ], axis=1)
    percentiles = np.stack([
        _percentile_within_groups(base[:, column], query_ptr)
        for column in range(len(base_names))
    ], axis=1)
    query_rows: list[np.ndarray] = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        n_molecules = right - left
        edge_count_query = int(molecule_ptr[right] - molecule_ptr[left])
        v1_molecule = base[left:right, 0]
        wse_molecule = base[left:right, 2]
        p2b_molecule = base[left:right, 6]
        v1_top, wse_top, p2b_top = _unique_top(v1_molecule), _unique_top(wse_molecule), _unique_top(p2b_molecule)

        def gap(scores: np.ndarray) -> float:
            order = np.sort(scores)[::-1]
            return float(order[0] - order[1]) if len(order) > 1 else 0.0

        query_rows.append(np.tile(np.asarray([
            float(n_molecules), float(edge_count_query), gap(v1_molecule), gap(wse_molecule),
            gap(p2b_molecule),
            1.0 if v1_top >= 0 and v1_top == wse_top else 0.0,
            1.0 if v1_top >= 0 and v1_top == p2b_top else 0.0,
            float(np.log(max(edge_count_query, 1))),
        ]), (n_molecules, 1)))
    context = np.concatenate(query_rows, axis=0)
    features = np.concatenate([base, percentiles, context], axis=1)
    if features.shape != (len(molecule_ptr) - 1, len(MOLECULE_FEATURES)):
        raise RuntimeError(f"feature matrix shape drifted: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise RuntimeError("non-finite feature reached the fusion stack")
    return features


# Backward-readable name for frozen artifact metadata; values are unchanged.
EDGE_FEATURES = MOLECULE_FEATURES
