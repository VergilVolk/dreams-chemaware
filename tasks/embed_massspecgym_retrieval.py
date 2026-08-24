"""Embed MassSpecGym spectra in the official DreaMS *retrieval* space.

Why this script exists
----------------------
The COSMIC Layer-1 confidence must live in the SAME embedding space the retrieval
pipeline uses, otherwise the score/probe/decoy/self-retrieval are not comparable.

There are TWO different models in this repo and they must not be mixed:

  * ``ssl_model_server.pt``      (raw_ssl)           -> SSL-pretrained backbone,
    NO head, 128 peaks. Produced ``data/validation/e0_baseline/e0_embeddings.npy``
    and the original ``frozen_concept_probe`` (whose report claimed "official DreaMS
    embedding" -- that claim was false).
  * ``official_embedding_slim.pt`` (official_embedding) -> contrastive retrieval
    embedding (backbone + 1024->1024 head, 100 peaks). This is what
    ``annotation.embed.load_embedder`` (the whole M1/M2 retrieval pipeline) uses.

Measured: the two backbones differ ~1e-3 per layer (both fp32) and, even matched on
no-head + 128 peaks, their precursor embeddings are only cosine ~0.46-0.69. They are
different models, not different preprocessings.

Output layout
-------------
Cache order: ``retrieval_embeddings.npy[k]`` is the retrieval-space embedding of the
spectrum at ``spectrum_rule_labels.npz -> hdf5_row[k]`` (i.e. aligned row-for-row with
``labels[k]`` / ``ik14[k]``). No ``embedding_idx`` permutation -- that indirection only
existed because ``e0_embeddings.npy`` was written in a different order.

Fail-fast checks
----------------
  1. Outputs are finite and L2-normalized (the retrieval embedding is normalized).

Usage (conda dreams_env):
    python tasks/embed_massspecgym_retrieval.py --n 3000            # local smoke
    python tasks/embed_massspecgym_retrieval.py --device cuda       # full 45k (GPU)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation import embed  # noqa: E402

DEFAULT_HDF5 = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_LABELS = ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz"
DEFAULT_OUT = ROOT / "data/validation/cosmic_retrieval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=0, help="embed first N cache rows (0 = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = np.load(args.labels, allow_pickle=False)
    hdf5_row = cache["hdf5_row"].astype(np.int64)
    ik14 = cache["ik14"].astype(str)
    m_total = len(hdf5_row)

    n = args.n if args.n > 0 else m_total
    rows = hdf5_row[:n]

    model, w_head, b_head = embed.load_embedder(args.device)
    if args.device == "cpu":
        # load_embedder caps torch threads at 8; use more cores for the CPU forward.
        torch.set_num_threads(max(8, min(os.cpu_count() or 8, 16)))
    print(f"[embed] loaded retrieval-space embedder ({time.time()-t0:.0f}s)", flush=True)

    # Vectorized read of only the needed rows. We pass the RAW padded (2,128) rows
    # directly: preprocess_spectrum keeps top-100 by intensity, which excludes
    # zero-intensity padding either way, so stripping is a no-op (verified against
    # the full SpectrumPreprocessor path on padded vs stripped inputs).
    with h5py.File(args.data, "r") as h:
        spec_subset = np.asarray(h["spectrum"][rows], dtype=np.float32)       # (n, 2, 128)
        prec_subset = np.asarray(h["precursor_mz"][rows], dtype=np.float64)   # (n,)
    records = [
        {"peaks": spec_subset[i], "precursor_mz": float(prec_subset[i])}
        for i in range(len(rows))
    ]

    print(f"[embed] embedding {len(records)} spectra (n_highest_peaks=100) ...", flush=True)
    out_emb = embed.embed_records(
        records, model, w_head, b_head, args.device, batch_size=args.batch_size
    )  # [n, 1024] L2-normalized, cache order

    # ── fail-fast: finite + L2-normalized ──
    if not np.isfinite(out_emb).all():
        raise RuntimeError("non-finite embedding values")
    norms = np.linalg.norm(out_emb, axis=1)
    if float(np.abs(norms - 1.0).max()) > 1e-4:
        raise RuntimeError(
            f"embeddings not L2-normalized (max |norm-1| = {np.abs(norms-1.0).max():.2e})"
        )

    emb_path = args.output_dir / "retrieval_embeddings.npy"
    np.save(emb_path, out_emb)
    np.savez(
        args.output_dir / "embed_manifest.npz",
        hdf5_row=hdf5_row[:n],
        ik14=ik14[:n],
        n_embedded=np.int64(n),
        n_total=np.int64(m_total),
    )
    print(f"[embed] wrote {emb_path} shape={out_emb.shape} "
          f"(embedded {n}/{m_total}, L2-norm ok, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
