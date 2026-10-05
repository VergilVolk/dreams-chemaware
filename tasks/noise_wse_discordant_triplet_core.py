"""Pure selection helpers for WSE-supported native DreaMS triplets.

Weighted spectral entropy is used only to choose labelled relations on the
MassSpecGym training corpus.  Its score is never a target and never enters the
model loss.  The returned relation is therefore an ordinary native
``(anchor, positive, negative)`` triplet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WSETripletChoice:
    positive_row: int
    negative_row: int
    v1_positive: float
    v1_negative: float
    wse_positive: float
    wse_negative: float

    @property
    def v1_margin(self) -> float:
        return self.v1_positive - self.v1_negative

    @property
    def wse_margin(self) -> float:
        return self.wse_positive - self.wse_negative

    @property
    def stratum(self) -> str:
        return "misranked" if self.v1_margin <= 0.0 else "native_hinge_boundary"


def choose_wse_supported_triplet(
    positive_rows: np.ndarray,
    positive_v1: np.ndarray,
    positive_wse: np.ndarray,
    negative_rows: np.ndarray,
    negative_v1: np.ndarray,
    negative_wse: np.ndarray,
    *,
    native_margin: float = 0.1,
) -> WSETripletChoice | None:
    """Choose one informative relation without using an evaluation label.

    A relation qualifies exactly when WSE orders the labelled positive above
    the labelled negative while the current V1 encoder still has active native
    triplet loss.  Misranked V1 relations are preferred; ties then favour the
    smallest V1 margin and the largest independent WSE margin.
    """
    arrays = tuple(np.asarray(value) for value in (
        positive_rows, positive_v1, positive_wse,
        negative_rows, negative_v1, negative_wse,
    ))
    p_rows, p_v1, p_wse, n_rows, n_v1, n_wse = arrays
    if (
        p_rows.ndim != 1 or n_rows.ndim != 1
        or p_v1.shape != p_rows.shape or p_wse.shape != p_rows.shape
        or n_v1.shape != n_rows.shape or n_wse.shape != n_rows.shape
        or native_margin <= 0
    ):
        raise ValueError("invalid WSE triplet candidate arrays")
    if not len(p_rows) or not len(n_rows):
        return None
    numeric = np.concatenate((p_v1, p_wse, n_v1, n_wse)).astype(np.float64)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("non-finite score reached WSE triplet selection")

    choices: list[WSETripletChoice] = []
    for p_index, p_row in enumerate(p_rows):
        for n_index, n_row in enumerate(n_rows):
            choice = WSETripletChoice(
                positive_row=int(p_row),
                negative_row=int(n_row),
                v1_positive=float(p_v1[p_index]),
                v1_negative=float(n_v1[n_index]),
                wse_positive=float(p_wse[p_index]),
                wse_negative=float(n_wse[n_index]),
            )
            if choice.wse_margin > 0.0 and choice.v1_margin < native_margin:
                choices.append(choice)
    if not choices:
        return None
    return min(
        choices,
        key=lambda choice: (
            choice.v1_margin > 0.0,
            choice.v1_margin,
            -choice.wse_margin,
            choice.positive_row,
            choice.negative_row,
        ),
    )


def choose_v1_preservation_triplet(
    positive_rows: np.ndarray,
    positive_v1: np.ndarray,
    negative_rows: np.ndarray,
    negative_v1: np.ndarray,
) -> tuple[int, int, float]:
    """Return the largest-margin measured relation for a non-WSE query."""
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    positive_v1 = np.asarray(positive_v1, dtype=np.float64)
    negative_rows = np.asarray(negative_rows, dtype=np.int64)
    negative_v1 = np.asarray(negative_v1, dtype=np.float64)
    if (
        not len(positive_rows) or not len(negative_rows)
        or positive_v1.shape != positive_rows.shape
        or negative_v1.shape != negative_rows.shape
        or not np.all(np.isfinite(np.concatenate((positive_v1, negative_v1))))
    ):
        raise ValueError("invalid V1 preservation candidates")
    p_index = int(np.argmax(positive_v1))
    n_index = int(np.argmin(negative_v1))
    return (
        int(positive_rows[p_index]),
        int(negative_rows[n_index]),
        float(positive_v1[p_index] - negative_v1[n_index]),
    )


def candidate_molecule_max_rows(
    pair_rows: np.ndarray,
    local_molecule_ptr: np.ndarray,
    pair_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one max-scoring spectrum row for every candidate molecule."""
    pair_rows = np.asarray(pair_rows, dtype=np.int64)
    local_molecule_ptr = np.asarray(local_molecule_ptr, dtype=np.int64)
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    if (
        pair_rows.ndim != 1 or pair_scores.shape != pair_rows.shape
        or local_molecule_ptr.ndim != 1 or len(local_molecule_ptr) < 3
        or local_molecule_ptr[0] != 0 or local_molecule_ptr[-1] != len(pair_rows)
        or np.any(np.diff(local_molecule_ptr) < 1)
        or not np.all(np.isfinite(pair_scores))
    ):
        raise ValueError("invalid candidate molecule block")
    rows: list[int] = []
    scores: list[float] = []
    for left, right in zip(local_molecule_ptr[:-1], local_molecule_ptr[1:]):
        left, right = int(left), int(right)
        local = int(np.argmax(pair_scores[left:right])) + left
        rows.append(int(pair_rows[local]))
        scores.append(float(pair_scores[local]))
    return np.asarray(rows, dtype=np.int64), np.asarray(scores, dtype=np.float64)
