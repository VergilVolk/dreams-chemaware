"""M3 -- Target-decoy false discovery rate (FDR) estimation.

Spectral-matching annotations are scored by cosine, but a raw cosine threshold
has no false-positive rate attached to it. The target-decoy approach estimates
that FDR: generate shuffled "decoy" spectra that share the same precursor m/z but
whose fragment structure is destroyed, embed them identically, and let decoy hits
compete with target hits (Elias & Gygi, Nat Methods 2007, DOI 10.1038/nmeth1013;
Scheubert et al. passatutto, Nat Commun 2017, DOI 10.1038/s41467-017-01318-5).

A target hit at score ``s`` receives q = min over s' >= s of
(N_decoy(s') + 1) / (N_target(s') + 1), the standard TDA q-value.

**Calibration caveat (measured on library self-retrieval, neg mode, 2026-08-19):**
shuffle decoys are NOT competitive for the DreaMS embedder. They keep the source
compound's fragment m/z axis (only intensities are permuted), so a decoy of the
*true* compound still reaches cosine ~0.92 and inflates N_decoy -> TDA
**over-estimates** FDR ~4x (0.67 estimated vs 0.12-0.16 actual). The opposite
extreme, precursor-swap decoys (real spectra re-labelled with a random precursor
m/z), breaks the mass-structure correlation and **under-estimates** ~5x (0.03 vs
0.16). Neither decoy is calibrated; the defensible confidence scale is the direct
self-retrieval FDR curve (~12% FDR, flat across cosine 0.5-0.95 -- m/z does the
discrimination, not cosine). Do NOT report decoy-TDA q-values as a 1%-FDR claim.
"""
from __future__ import annotations

import numpy as np

from .params import Params, source


def make_shuffle_decoys(
    records: list[dict],
    n_decoys_per_target: int = 1,
    seed: int = 0,
) -> list[dict]:
    """Generate shuffle decoys: permute fragment intensities in place, keeping
    the m/z axis and precursor m/z intact. Returns ``n_decoys_per_target * len``
    decoy records, each with the same ``{"peaks": (2,n), "precursor_mz": ...}``
    schema as the targets."""
    rng = np.random.default_rng(seed)
    decoys: list[dict] = []
    for _ in range(n_decoys_per_target):
        for r in records:
            peaks = np.asarray(r["peaks"], dtype=np.float32).copy()
            perm = rng.permutation(peaks.shape[1])
            peaks[1] = peaks[1][perm]
            decoys.append({
                "peaks": peaks,
                "precursor_mz": r["precursor_mz"],
            })
    return decoys


def top1_scores(query: np.ndarray, library: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Cosine of each query's best library hit (chunked matmul)."""
    q = query
    if not np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-3):
        q = q / np.linalg.norm(q, axis=1)[:, None]
    lib = library
    if not np.allclose(np.linalg.norm(lib, axis=1), 1.0, atol=1e-3):
        lib = lib / np.linalg.norm(lib, axis=1)[:, None]
    best = np.empty(q.shape[0], dtype=np.float32)
    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        best[start:stop] = (q[start:stop] @ lib.T).max(axis=1)
    return best


def mz_masked_top1_scores(
    query: np.ndarray,
    q_mz: np.ndarray,
    library: np.ndarray,
    l_mz: np.ndarray,
    ppm_tolerance: float,
    chunk: int = 512,
) -> np.ndarray:
    """Cosine of each query's best library hit **subject to the precursor-m/z hard
    constraint** (|q_mz - l_mz|/l_mz*1e6 <= ppm_tolerance). Returns -inf when a
    query has no m/z-matching library entry.

    This is the score that the annotation pipeline actually applies (cosine >=
    cutoff AND m/z pass); the plain :func:`top1_scores` ignores m/z and therefore
    does not match the confident-annotation set the FDR is meant to describe.
    Decoys share the precursor m/z of their source spectrum (shuffle keeps it), so
    the same ``l_mz`` mask applies to target and decoy libraries alike.
    """
    q = query
    if not np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-3):
        q = q / np.linalg.norm(q, axis=1)[:, None]
    lib = library
    if not np.allclose(np.linalg.norm(lib, axis=1), 1.0, atol=1e-3):
        lib = lib / np.linalg.norm(lib, axis=1)[:, None]
    q_mz = np.asarray(q_mz, dtype=np.float64)
    l_mz = np.asarray(l_mz, dtype=np.float64)
    best = np.full(q.shape[0], -np.inf, dtype=np.float32)
    for start in range(0, q.shape[0], chunk):
        stop = min(start + chunk, q.shape[0])
        sim = q[start:stop] @ lib.T
        dppm = np.abs(q_mz[start:stop, None] - l_mz[None, :]) / np.maximum(
            np.abs(l_mz[None, :]), 1e-9) * 1e6
        best[start:stop] = np.where(dppm <= ppm_tolerance, sim, -np.inf).max(axis=1)
    return best


def target_decoy_qvalues(
    target_scores: np.ndarray,
    decoy_scores: np.ndarray,
) -> np.ndarray:
    """Classic TDA q-values: q(s) = min_{s' >= s} (N_decoy(s') + 1) / (N_target(s') + 1)."""
    # ``order`` sorts target_scores DESCENDING (argsort of the negation), so
    # sorted_scores[0] is the max score and sorted_scores[-1] the min.
    order = np.argsort(-target_scores, kind="stable")
    sorted_scores = target_scores[order]
    # Number of target scores >= each sorted score: 1 at the max, len at the min.
    # (BUGFIX: was arange(len, 0, -1) = [len, ..., 1], i.e. inverted, which made
    # fdr(top) = 1/(N_query+1) and dragged every q-value to that floor.)
    n_target_ge = np.arange(1, len(target_scores) + 1).astype(np.float64)
    # count of decoy scores >= each target score, via sort + searchsorted (O(n log n)).
    sorted_decoy = np.sort(decoy_scores)
    n_decoy_ge = len(decoy_scores) - np.searchsorted(sorted_decoy, sorted_scores, side="left")
    fdr = (n_decoy_ge + 1.0) / (n_target_ge + 1.0)
    # cumulative minimum from highest score down
    q = np.minimum.accumulate(fdr)
    out = np.empty_like(q)
    out[order] = q
    return out


def annotate_fdr(
    hits,
    target_top1_scores: np.ndarray,
    decoy_top1_scores: np.ndarray,
    params: Params,
):
    """Attach q-value + fdr_pass to a long annotation table.

    ``target_top1_scores`` / ``decoy_top1_scores`` are per-query best scores
    against the target and decoy libraries respectively (length == n_query).
    q-values are computed once per query and broadcast to that query's rows.
    """
    qvalues = target_decoy_qvalues(target_top1_scores, decoy_top1_scores)
    out = hits.copy()
    out["qvalue"] = qvalues[out["query_idx"].to_numpy()]
    out["fdr_pass"] = out["qvalue"] <= params.qvalue_threshold
    return out


FDR_CITATIONS = {
    "decoy": source("passatutto") + "; " + source("elias_gygi"),
    "qvalue": source("elias_gygi"),
    "rescoring": source("mokapot"),
}
