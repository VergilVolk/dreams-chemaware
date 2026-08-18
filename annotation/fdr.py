"""M3 -- Target-decoy false discovery rate (FDR) estimation.

Spectral-matching annotations are scored by cosine, but a raw cosine threshold
has no false-positive rate attached to it. The target-decoy approach estimates
that FDR: generate shuffled "decoy" spectra that share the same precursor m/z but
whose fragment structure is destroyed, embed them identically, and let decoy hits
compete with target hits (Elias & Gygi, Nat Methods 2007, DOI 10.1038/nmeth1013;
Scheubert et al. passatutto, Nat Commun 2017, DOI 10.1038/s41467-017-01318-5).

A target hit at score ``s`` receives q = min over s' >= s of
(N_decoy(s') + 1) / (N_target(s') + 1), the standard TDA q-value.
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


def target_decoy_qvalues(
    target_scores: np.ndarray,
    decoy_scores: np.ndarray,
) -> np.ndarray:
    """Classic TDA q-values: q(s) = min_{s' >= s} (N_decoy(s') + 1) / (N_target(s') + 1)."""
    order = np.argsort(-target_scores, kind="stable")
    sorted_scores = target_scores[order]
    n_target_ge = np.arange(len(target_scores), 0, -1).astype(np.float64)
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
