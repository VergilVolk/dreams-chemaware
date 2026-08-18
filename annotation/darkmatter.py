"""M7 -- Dark-matter candidate mining.

"Dark matter" = MS2 spectra with no confident library annotation (in our Met/neg
baseline, 94%). This module turns them into *candidates*, not confirmations:

  1. identify dark spectra (no confident top-1 hit),
  2. greedily cluster them by embedding cosine (spectra of the same unknown
     compound should co-cluster),
  3. report structural *leads* for each cluster from its low-cosine library hits
     and (when available) rule-based diagnostic evidence.

Confidence framing follows the dark-metabolome perspective (Cao et al., JACS Au
2025, DOI 10.1021/jacsau.5c01063) and the level-3 "tentative candidate" tier of
Schymanski et al. 2014 -- these are hypotheses to validate, never claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, source


def dark_mask(hits: pd.DataFrame, params: Params) -> np.ndarray:
    """Boolean mask over *query* indices whose top-1 hit is not confident
    (cosine below cutoff OR m/z mismatch)."""
    top1 = hits[hits["rank"] == 1].set_index("query_idx")
    qidx = top1.index.to_numpy()
    confident = (top1["cosine"] >= params.cosine_confident) & top1["mz_pass"]
    dark = np.ones(hits["query_idx"].nunique(), dtype=bool)
    for i, ok in zip(qidx, confident.to_numpy()):
        dark[i] = not bool(ok)
    return dark


def cluster_dark(
    query_emb: np.ndarray,
    dark_indices: np.ndarray,
    cosine_threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy single-pass clustering of dark spectra by embedding cosine.

    Returns (labels over dark_indices, cluster_centers). Each spectrum joins the
    cluster of the nearest existing center if cosine >= threshold, else seeds a
    new cluster. O(n_dark * n_centers) dot products; fine for ~10k spectra."""
    n = len(dark_indices)
    labels = np.full(n, -1, dtype=np.int64)
    centers: list[np.ndarray] = []
    for i, idx in enumerate(dark_indices):
        q = query_emb[idx]
        if centers:
            sims = np.asarray([float(q @ c) for c in centers])
            best = int(np.argmax(sims))
            if sims[best] >= cosine_threshold:
                labels[i] = best
            else:
                labels[i] = len(centers)
                centers.append(q)
        else:
            labels[i] = 0
            centers.append(q)
    return labels, np.asarray(centers)


def candidate_leads(
    hits: pd.DataFrame,
    dark_indices: np.ndarray,
    params: Params,
    top_n_leads: int = 3,
) -> pd.DataFrame:
    """For each dark query, return its top-N library leads (low-cosine hits used
    as structural hints) with the query's cluster label if provided."""
    rows = []
    qidx_set = set(int(i) for i in dark_indices)
    dark_hits = hits[hits["query_idx"].isin(qidx_set)].copy()
    dark_hits = dark_hits.sort_values(["query_idx", "rank"])
    for _, g in dark_hits.groupby("query_idx"):
        leads = g.head(top_n_leads)
        for _, r in leads.iterrows():
            rows.append({
                "query_idx": int(r["query_idx"]),
                "query_file": r["query_file"],
                "query_precursor_mz": r["query_precursor_mz"],
                "lead_rank": int(r["rank"]),
                "lead_cosine": r["cosine"],
                "lead_name": r["lib_name"],
                "lead_inchikey": r["lib_inchikey"],
                "lead_precursor_mz": r["lib_precursor_mz"],
                "dppm": r["dppm"],
            })
    return pd.DataFrame(rows)


def summarize_clusters(
    labels: np.ndarray,
    dark_indices: np.ndarray,
    query_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Cluster summary: size, mean precursor m/z, and member query indices."""
    rows = []
    for c in np.unique(labels):
        members = dark_indices[labels == c]
        mzs = query_manifest["precursor_mz"].to_numpy()[members]
        rows.append({
            "cluster": int(c),
            "n_spectra": int(len(members)),
            "mean_precursor_mz": float(np.mean(mzs)),
            "precursor_mz_std": float(np.std(mzs)),
        })
    return pd.DataFrame(rows).sort_values("n_spectra", ascending=False).reset_index(drop=True)


DARK_CITATIONS = {
    "framing": source("darkmatter"),
    "level": source("schymanski"),
    "class_prior": source("canopus") + " (optional class-level prior, not yet wired)",
}
