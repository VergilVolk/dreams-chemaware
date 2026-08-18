"""Cosine top-k retrieval: tissue MS2 spectra (MSV000100574) vs MONA-negative library.

Loads the two precomputed embedding sets, computes chunked cosine similarity
(tissue x library), and reports:
  * top-k library hits per tissue spectrum (SMILES / InChIKey / name / precursor_mz)
  * annotation rate = fraction of tissue spectra whose top-1 cosine is >= a threshold
  * top-1 similarity histogram (how many spectra are "dark matter" vs confidently matched)

The embedding space is L2-normalized, so cosine similarity = dot product.

Usage (CPU):
    python tasks/retrieve_mona_neg.py \
        --tissue data/msv100574/embeddings/met_neg \
        --library data/models/mona_neg_dreams_emb \
        --out data/msv100574/retrieval/met_neg \
        --topk 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Similarity thresholds for the annotation-rate report. 0.7 is the standard
# "confident" cutoff in DreaMS retrieval evaluations; we report a ladder.
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# A top-k hit only counts as a *structural annotation* if BOTH the cosine is high
# AND the precursor m/z agrees (same adduct, e.g. [M-H]-). The DreaMS embedding
# encodes fragmentation (structure) far more than mass, so raw cosine alone
# over-annotates ~4x: measured on Met/neg, 73% of top-1 hits are >1000 ppm off.
DEFAULT_PPM_TOL = 20.0


def load_set(d: Path) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(d / "embeddings.npy")
    manifest = pd.read_csv(d / "manifest.csv")
    assert len(emb) == len(manifest), f"{d}: embedding/manifest length mismatch"
    return emb, manifest


def chunked_topk(
    tissue: np.ndarray,
    library: np.ndarray,
    k: int,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (topk_vals, topk_idx) each shaped (n_tissue, k).

    Processes tissue rows in chunks to keep peak memory bounded (library is 37k x 1024,
    so one full similarity matrix would be ~2 GB at float32).
    """
    n_tissue = tissue.shape[0]
    topk_vals = np.empty((n_tissue, k), dtype=np.float32)
    topk_idx = np.empty((n_tissue, k), dtype=np.int64)
    for start in range(0, n_tissue, chunk):
        stop = min(start + chunk, n_tissue)
        sim = tissue[start:stop] @ library.T  # (chunk, n_lib) cosine (both L2-normed)
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]  # (chunk, k) candidate idx
        # gather + sort the top-k within each row
        vals = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        topk_idx[start:stop] = np.take_along_axis(idx, order, axis=1)
        topk_vals[start:stop] = np.take_along_axis(vals, order, axis=1)
        if (stop // chunk) % 20 == 0:
            print(f"    {stop}/{n_tissue} rows", flush=True)
    return topk_vals, topk_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--ppm-tolerance", type=float, default=DEFAULT_PPM_TOL,
                        help="max |tissue - library| precursor m/z offset (ppm) for a hit to count")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[1] loading embeddings...", flush=True)
    t0 = time.time()
    tissue, t_manifest = load_set(args.tissue)
    library, l_manifest = load_set(args.library)
    print(f"    tissue {tissue.shape} | library {library.shape} | {time.time()-t0:.0f}s", flush=True)

    # cosine similarity assumes unit-norm; verify and normalize defensively.
    for name, arr in (("tissue", tissue), ("library", library)):
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(f"    [warn] {name} not unit-norm; re-normalizing", flush=True)
            arr /= norms[:, None]

    print("[2] chunked cosine top-k...", flush=True)
    t0 = time.time()
    topk_vals, topk_idx = chunked_topk(tissue, library, args.topk, args.chunk)
    print(f"    done {time.time()-t0:.0f}s", flush=True)

    top1 = topk_vals[:, 0]

    # precursor m/z consistency of the top-1 hit (same-adduct requirement).
    t_pmz = t_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_pmz = l_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    top1_lib = topk_idx[:, 0]
    dppm = np.abs(t_pmz - l_pmz[top1_lib]) / np.maximum(np.abs(l_pmz[top1_lib]), 1e-9) * 1e6

    # annotation rate at each threshold (raw cosine only)
    rates = {str(t): float((top1 >= t).mean()) for t in THRESHOLDS}
    # annotation rate requiring BOTH cosine AND precursor-m/z agreement
    mz_rates = {
        str(t): float(((top1 >= t) & (dppm <= args.ppm_tolerance)).mean())
        for t in THRESHOLDS
    }

    # histogram of top-1 similarity (coarse bins for a quick dark-matter readout)
    bins = np.arange(0.0, 1.01, 0.1)
    hist, _ = np.histogram(top1, bins=bins)
    hist_report = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}

    report = {
        "status": "msv100574_mona_neg_retrieval",
        "n_tissue_spectra": int(len(tissue)),
        "n_library_spectra": int(len(library)),
        "topk": args.topk,
        "ppm_tolerance": args.ppm_tolerance,
        "annotation_rate_cosine_only": rates,
        "annotation_rate_cosine_and_mz": mz_rates,
        "top1_similarity": {
            "mean": float(top1.mean()),
            "median": float(np.median(top1)),
            "max": float(top1.max()),
            "p90": float(np.percentile(top1, 90)),
        },
        "top1_histogram": hist_report,
    }
    print(json.dumps(report, indent=2), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # save top-k hits as a long table: one row per (tissue_spectrum, rank)
    print("[3] writing top-k hit table...", flush=True)
    rows = []
    l_smiles = l_manifest["smiles"].tolist()
    l_inchikey = l_manifest["inchikey"].tolist()
    l_name = l_manifest["name"].tolist()
    l_pmz = l_manifest["precursor_mz"].tolist()
    for i in range(len(tissue)):
        for r in range(args.topk):
            j = int(topk_idx[i, r])
            rows.append({
                "tissue_idx": i,
                "tissue_file": t_manifest["file_name"].iloc[i],
                "tissue_scan": int(t_manifest["scan_number"].iloc[i]),
                "tissue_precursor_mz": float(t_manifest["precursor_mz"].iloc[i]),
                "rank": r + 1,
                "cosine": float(topk_vals[i, r]),
                "lib_smiles": l_smiles[j],
                "lib_inchikey": l_inchikey[j],
                "lib_name": l_name[j],
                "lib_precursor_mz": l_pmz[j],
            })
    hits = pd.DataFrame(rows)
    # dppm for each hit's top-1 precursor-m/z offset (same value per tissue spectrum)
    hits["dppm"] = dppm[hits["tissue_idx"].to_numpy()]
    hits.to_csv(args.out / "topk_hits.csv", index=False)
    print(f"    wrote {len(hits)} rows to topk_hits.csv", flush=True)

    # confident structural annotations: top-1 cosine >= 0.7 AND precursor m/z agrees.
    conf = hits[hits["rank"] == 1]
    conf = conf[(conf["cosine"] >= 0.7) & (conf["dppm"] <= args.ppm_tolerance)]
    conf = conf.reset_index(drop=True)
    conf.to_csv(args.out / "annotations_confident.csv", index=False)
    print(f"    confident annotations (top1>=0.7 & dppm<={args.ppm_tolerance}): {len(conf)}", flush=True)


if __name__ == "__main__":
    main()
