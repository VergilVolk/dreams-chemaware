"""M1 -- Cosine top-k retrieval with a precursor m/z hard constraint.

The DreaMS embedding encodes fragmentation (structure) far more than mass --
the precursor m/z is only a single prepended peak -- so raw cosine alone
over-annotates ~4x (measured on Met/neg: 73% of top-1 hits are >1000 ppm off).
We therefore require precursor-m/z agreement (same adduct, tolerance in ppm)
as a hard constraint, matching the DreaMS retrieval evaluation which also
combines cosine ranking with a precursor-m/z tolerance
(Bushuiev et al., Nat Biotechnol 2025, DOI 10.1038/s41587-025-02663-3).

This module produces the standard annotation table that every downstream module
(M2 confidence levels, M3 FDR, M4 calibration) appends columns to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Params, source


def load_embedding_set(d: Path) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(d / "embeddings.npy")
    manifest = pd.read_csv(d / "manifest.csv")
    assert len(emb) == len(manifest), f"{d}: embedding/manifest length mismatch"
    return emb, manifest


def chunked_topk(
    query: np.ndarray, library: np.ndarray, k: int, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Return (topk_vals, topk_idx) each (n_query, k), chunked to bound memory."""
    n_query = query.shape[0]
    topk_vals = np.empty((n_query, k), dtype=np.float32)
    topk_idx = np.empty((n_query, k), dtype=np.int64)
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        sim = query[start:stop] @ library.T  # both L2-normalized -> cosine
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        topk_idx[start:stop] = np.take_along_axis(idx, order, axis=1)
        topk_vals[start:stop] = np.take_along_axis(vals, order, axis=1)
    return topk_vals, topk_idx


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        arr = arr / norms[:, None]
    return arr


def dppm(query_mz: np.ndarray, lib_mz: np.ndarray) -> np.ndarray:
    """Precursor-m/z offset in ppm, |q - l| / l * 1e6."""
    return np.abs(query_mz - lib_mz) / np.maximum(np.abs(lib_mz), 1e-9) * 1e6


def retrieve(
    query_dir: Path,
    library_dir: Path,
    params: Params,
    group_by: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run top-k retrieval and return (annotations, report).

    ``annotations`` is a long table, one row per (query spectrum, rank k), with
    columns: query metadata, cosine, lib metadata, dppm, mz_pass. ``group_by``
    optionally names a query-manifest column (e.g. a group/condition label) to be
    carried into the table for downstream differential analysis.

    ``report`` records the annotation rate ladder both with and without the m/z
    constraint -- the two numbers that the whole pipeline is benchmarked on.
    """
    query, q_manifest = load_embedding_set(query_dir)
    library, l_manifest = load_embedding_set(library_dir)
    query = _normalize(query)
    library = _normalize(library)

    topk_vals, topk_idx = chunked_topk(query, library, params.topk)

    q_pmz = q_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_pmz = l_manifest["precursor_mz"].to_numpy(dtype=np.float64)

    # long table
    rows = []
    l_smiles = l_manifest["smiles"].tolist()
    l_inchikey = l_manifest["inchikey"].tolist()
    l_name = l_manifest["name"].tolist()
    l_pmz_list = l_manifest["precursor_mz"].tolist()
    q_group = q_manifest[group_by].tolist() if group_by else [""] * len(query)
    for i in range(len(query)):
        for r in range(params.topk):
            j = int(topk_idx[i, r])
            rows.append({
                "query_idx": i,
                "query_file": q_manifest["file_name"].iloc[i],
                "query_scan": int(q_manifest["scan_number"].iloc[i]),
                "query_precursor_mz": float(q_pmz[i]),
                "query_group": q_group[i],
                "rank": r + 1,
                "cosine": float(topk_vals[i, r]),
                "lib_smiles": l_smiles[j],
                "lib_inchikey": l_inchikey[j],
                "lib_name": l_name[j],
                "lib_precursor_mz": l_pmz_list[j],
            })
    hits = pd.DataFrame(rows)
    top1_lib = topk_idx[:, 0]
    # dppm / mz_pass must be per-row (this rank's library hit), not the top-1 hit.
    hits["dppm"] = dppm(
        hits["query_precursor_mz"].to_numpy(dtype=np.float64),
        hits["lib_precursor_mz"].to_numpy(dtype=np.float64),
    )
    hits["mz_pass"] = hits["dppm"] <= params.ppm_tolerance

    top1 = topk_vals[:, 0]
    rates_cos = {str(t): float((top1 >= t).mean()) for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
    top1_dppm = dppm(q_pmz, l_pmz[top1_lib])
    rates_mz = {
        str(t): float(((top1 >= t) & (top1_dppm <= params.ppm_tolerance)).mean())
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]
    }

    report = {
        "n_query_spectra": int(len(query)),
        "n_library_spectra": int(len(library)),
        "topk": params.topk,
        "ppm_tolerance": params.ppm_tolerance,
        "cosine_confident": params.cosine_confident,
        "annotation_rate_cosine_only": rates_cos,
        "annotation_rate_cosine_and_mz": rates_mz,
        "sources": {"ppm": source("dreams"), "cosine": source("dreams")},
    }
    return hits, report


def save(hits: pd.DataFrame, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hits.to_csv(out_dir / "annotations.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
