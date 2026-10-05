"""Frozen, label-blind RRF helpers for Noise Stage-1 confirmation.

The positive label is never used to form a score.  Candidate spectra are
first aggregated to molecules with max similarity; all three molecule score
vectors are converted to tie-aware midranks and fused with fixed RRF k=60.
Final retrieval ranks remain strict (a positive tie is not a win).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from noise_corrected_fullgraph_evaluation import GraphScores


RRF_K = 60
MATCH_TOL_DA = 0.01
TOPK = 20


def _greedy_match_sum(
    weights_q: np.ndarray,
    mz_q: np.ndarray,
    weights_c: np.ndarray,
    mz_c: np.ndarray,
    tol: float,
) -> float:
    if len(mz_q) == 0 or len(mz_c) == 0:
        return 0.0
    ii, jj = np.nonzero(np.abs(mz_q[:, None] - mz_c[None, :]) <= tol)
    if not len(ii):
        return 0.0
    products = weights_q[ii] * weights_c[jj]
    order = np.argsort(-products, kind="stable")
    used_q = np.zeros(len(mz_q), dtype=bool)
    used_c = np.zeros(len(mz_c), dtype=bool)
    total = 0.0
    for position in order:
        i, j = int(ii[position]), int(jj[position])
        if not used_q[i] and not used_c[j]:
            used_q[i] = True
            used_c[j] = True
            total += float(products[position])
    return total


def cosine_similarity(
    mz_q: np.ndarray,
    intensity_q: np.ndarray,
    mz_c: np.ndarray,
    intensity_c: np.ndarray,
    tol: float = MATCH_TOL_DA,
) -> float:
    weights_q = np.sqrt(np.asarray(intensity_q, dtype=np.float64))
    weights_c = np.sqrt(np.asarray(intensity_c, dtype=np.float64))
    denominator = float(np.linalg.norm(weights_q) * np.linalg.norm(weights_c))
    if denominator <= 0.0:
        return 0.0
    return _greedy_match_sum(weights_q, mz_q, weights_c, mz_c, tol) / denominator


def topk_overlap(
    mz_q: np.ndarray,
    intensity_q: np.ndarray,
    mz_c: np.ndarray,
    intensity_c: np.ndarray,
    tol: float = MATCH_TOL_DA,
    k: int = TOPK,
) -> float:
    if len(mz_q) == 0 or len(mz_c) == 0:
        return 0.0
    order = np.lexsort((mz_q, -np.asarray(intensity_q)))
    selected = order[:min(k, len(mz_q))]
    candidate_mz = np.sort(np.asarray(mz_c, dtype=np.float64))
    lower, upper = candidate_mz - tol, candidate_mz + tol
    matched = 0
    for index in selected:
        mz = float(mz_q[index])
        position = int(np.searchsorted(lower, mz, side="right") - 1)
        if position >= 0 and mz <= upper[position]:
            matched += 1
    return matched / len(selected)


@dataclass(frozen=True)
class NaiveAudit:
    queries: int
    spectrum_comparisons: int
    excluded_self_edges: int
    empty_after_self_exclusion: int


def tie_aware_midranks(scores: np.ndarray) -> np.ndarray:
    """Return descending average ranks without using candidate order."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("RRF rank input must be a finite vector of length >=2")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = ((left + 1) + right) / 2.0
        left = right
    return ranks


def fixed_rrf_score(*score_vectors: np.ndarray, k: int = RRF_K) -> np.ndarray:
    if k != RRF_K:
        raise RuntimeError(f"confirmation recipe is frozen at RRF k={RRF_K}")
    if len(score_vectors) != 3:
        raise RuntimeError("confirmation recipe requires exactly DreaMS, cosine, top-k")
    lengths = {len(np.asarray(value)) for value in score_vectors}
    if len(lengths) != 1:
        raise RuntimeError("RRF score vectors are not aligned")
    output = np.zeros(next(iter(lengths)), dtype=np.float64)
    for values in score_vectors:
        output += 1.0 / (k + tie_aware_midranks(values))
    return output.astype(np.float32)


def build_rrf_graph_scores(
    graph,
    dreams_scores: GraphScores,
    queries: np.ndarray,
    spectrum: Callable[[int], tuple[np.ndarray, np.ndarray]],
    *,
    k: int = RRF_K,
    progress_every: int = 1000,
) -> tuple[GraphScores, NaiveAudit]:
    """Build fixed three-way RRF scores on complete candidate blocks.

    The query row is excluded from its own candidate spectrum pool.  A
    molecule with no remaining measured spectrum is a malformed benchmark and
    fails closed rather than receiving an invented score.
    """
    selected = np.asarray(queries, dtype=np.int64)
    if selected.ndim != 1 or not len(selected) or len(np.unique(selected)) != len(selected):
        raise RuntimeError("RRF query registry is empty or duplicated")
    if len(dreams_scores.molecule) != len(graph.molecule_label):
        raise RuntimeError("DreaMS molecule scores do not align with graph")
    molecule = np.zeros(len(graph.molecule_label), dtype=np.float32)
    pair = np.zeros(len(graph.pair_candidate_row), dtype=np.float32)
    comparisons = 0
    excluded_self = 0
    empty = 0
    for number, query_value in enumerate(selected, start=1):
        query = int(query_value)
        qrow = int(graph.query_row[query])
        query_mz, query_intensity = spectrum(qrow)
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        cosine_values = np.empty(molecule_right - molecule_left, dtype=np.float64)
        topk_values = np.empty_like(cosine_values)
        for local, candidate_molecule in enumerate(range(molecule_left, molecule_right)):
            candidate_cosine: list[float] = []
            candidate_topk: list[float] = []
            for pair_index in range(
                int(graph.molecule_ptr[candidate_molecule]),
                int(graph.molecule_ptr[candidate_molecule + 1]),
            ):
                row = int(graph.pair_candidate_row[pair_index])
                if row == qrow:
                    excluded_self += 1
                    continue
                candidate_mz, candidate_intensity = spectrum(row)
                candidate_cosine.append(float(cosine_similarity(
                    query_mz, query_intensity, candidate_mz, candidate_intensity,
                )))
                candidate_topk.append(float(topk_overlap(
                    query_mz, query_intensity, candidate_mz, candidate_intensity,
                )))
                comparisons += 1
            if not candidate_cosine:
                empty += 1
                raise RuntimeError(
                    f"query {query} molecule {candidate_molecule} has no non-self spectrum"
                )
            cosine_values[local] = max(candidate_cosine)
            topk_values[local] = max(candidate_topk)
        dreams_values = np.asarray(
            dreams_scores.molecule[molecule_left:molecule_right], dtype=np.float64,
        )
        fused = fixed_rrf_score(dreams_values, cosine_values, topk_values, k=k)
        molecule[molecule_left:molecule_right] = fused
        for local, candidate_molecule in enumerate(range(molecule_left, molecule_right)):
            left = int(graph.molecule_ptr[candidate_molecule])
            right = int(graph.molecule_ptr[candidate_molecule + 1])
            pair[left:right] = fused[local]
        if progress_every and (number % progress_every == 0 or number == len(selected)):
            print(f"[fixed-RRF] {number:,}/{len(selected):,} queries", flush=True)
    return GraphScores(pair=pair, molecule=molecule), NaiveAudit(
        queries=int(len(selected)),
        spectrum_comparisons=int(comparisons),
        excluded_self_edges=int(excluded_self),
        empty_after_self_exclusion=int(empty),
    )
