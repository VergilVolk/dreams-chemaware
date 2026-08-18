"""M3 -- full target-decoy FDR on Met/neg vs MONA-neg (CPU, resumable).

Usage:
    nohup python tasks/run_fdr_met_neg.py > data/models/fdr_met_neg.log 2>&1 &

Chunked + checkpointed decoy embedding: shuffle decoys are generated and embedded
in chunks of 2000, each chunk saved to ``data/models/mona_neg_decoy_chunks/chunk_*.npy``.
A re-run skips cached chunks, so an interruption resumes instead of restarting.
Final concatenation is cached to ``data/models/mona_neg_decoy_emb.npy``.

Appends qvalue / fdr_pass to the annotation table.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation import embed, retrieve, fdr  # noqa: E402
from annotation.params import DEFAULT  # noqa: E402
from annotation.cli import parse_mgf  # noqa: E402

DECOY_CACHE = ROOT / "data/models/mona_neg_decoy_emb.npy"
DECOY_CHUNK_DIR = ROOT / "data/models/mona_neg_decoy_chunks"
CHUNK = 2000


def _make_chunk_decoys(records: list[dict], start: int, stop: int) -> list[dict]:
    """Shuffle decoys for records[start:stop]. Permutation seed = record index, so
    each decoy is deterministic and independent -> chunking/resume is exact."""
    decoys: list[dict] = []
    for i in range(start, stop):
        r = records[i]
        peaks = np.asarray(r["peaks"], dtype=np.float32).copy()
        perm = np.random.default_rng(i).permutation(peaks.shape[1])
        peaks[1] = peaks[1][perm]
        decoys.append({"peaks": peaks, "precursor_mz": r["precursor_mz"]})
    return decoys


def embed_decoys_chunked(records: list[dict], device: str) -> np.ndarray:
    """Chunked + checkpointed decoy embedding. Returns [n x 1024]."""
    DECOY_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    n = len(records)
    n_chunks = (n + CHUNK - 1) // CHUNK
    model, weight, bias = embed.load_embedder(device)
    chunk_files: list[Path] = []
    for ci in range(n_chunks):
        cf = DECOY_CHUNK_DIR / f"chunk_{ci:04d}.npy"
        chunk_files.append(cf)
        if cf.exists():
            print(f"[FDR] chunk {ci+1}/{n_chunks} cached, skip", flush=True)
            continue
        start = ci * CHUNK
        stop = min(start + CHUNK, n)
        t0 = time.time()
        decoys = _make_chunk_decoys(records, start, stop)
        emb = embed.embed_records(decoys, model, weight, bias, device, batch_size=128)
        np.save(cf, emb.astype(np.float32))
        print(f"[FDR] chunk {ci+1}/{n_chunks} ({stop-start} decoys) "
              f"{time.time()-t0:.0f}s -> {cf.name}", flush=True)
    return np.concatenate([np.load(cf) for cf in chunk_files])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    args = parser.parse_args()

    t0 = time.time()
    print("[FDR] parsing MGF...", flush=True)
    records = parse_mgf(ROOT / "data/models/mona_neg_full.mgf")
    print(f"[FDR] {len(records)} records in {time.time()-t0:.0f}s", flush=True)

    if DECOY_CACHE.exists():
        decoy_emb = np.load(DECOY_CACHE)
        print(f"[FDR] loaded cached decoy embeddings {decoy_emb.shape}", flush=True)
    else:
        t0 = time.time()
        decoy_emb = embed_decoys_chunked(records, args.device)
        np.save(DECOY_CACHE, decoy_emb.astype(np.float32))
        print(f"[FDR] cached -> {DECOY_CACHE} ({time.time()-t0:.0f}s)", flush=True)

    query, _ = retrieve.load_embedding_set(ROOT / "data/msv100574/embeddings/met_neg")
    library, _ = retrieve.load_embedding_set(ROOT / "data/models/mona_neg_dreams_emb")
    query = retrieve._normalize(query)
    library = retrieve._normalize(library)
    t0 = time.time()
    target_scores = fdr.top1_scores(query, library)
    decoy_scores = fdr.top1_scores(query, decoy_emb)
    print(f"[FDR] top1 scores in {time.time()-t0:.0f}s", flush=True)

    hits = pd.read_csv(ROOT / "data/msv100574/annotation/met_neg/annotations.csv")
    hits = fdr.annotate_fdr(hits, target_scores, decoy_scores, DEFAULT)

    n_query = len(query)
    n_fdr = int(hits[hits["rank"] == 1]["fdr_pass"].sum())
    print(f"[FDR] top1 passing q<=0.01: {n_fdr} / {n_query}", flush=True)
    q = hits[hits["rank"] == 1]["qvalue"].to_numpy()
    print("[FDR] top1 qvalue percentiles [50,75,90,95,99]:",
          np.percentile(q, [50, 75, 90, 95, 99]).round(4), flush=True)
    conf = hits[(hits["rank"] == 1) & (hits["cosine"] >= 0.7) & (hits["mz_pass"])]
    print(f"[FDR] confident top1 = {len(conf)}, of which fdr_pass = {int(conf['fdr_pass'].sum())}",
          flush=True)

    out = ROOT / "data/msv100574/annotation/met_neg/annotations_fdr.csv"
    hits.to_csv(out, index=False)
    print(f"[FDR] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
