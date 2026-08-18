"""Encode MSV000100574 tissue MS2 spectra with the official DreaMS embedding.

Input:  .hdf5 files produced by `MSData.load('<sample>.mzML')` (via pyteomics).
Output: embeddings.npy (N x 1024, L2-normalized) + manifest.csv (per-spectrum metadata)
        + report.json in the output dir.

Reuses the lightweight inference backbone (LightweightDreaMS) + the slim official
embedding checkpoint, exactly as the E0/E1 audit scripts do (no Lightning, no msml).

Usage (CPU):
    python tasks/encode_msv100574_spectra.py \
        --hdf5 data/msv100574/Metabolomics/neg/PF_1.hdf5 data/msv100574/Metabolomics/neg/HF_1.hdf5 \
        --out data/msv100574/embeddings/met_neg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import (  # noqa: E402
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    torch_load_compat,
)
from pilot_paired_layer_cka import (  # noqa: E402
    DEFAULT_RAW,
    SpectrumRows,
    reconstruct_backbone,
)

DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
N_HIGHEST_PEAKS = 100
BATCH_SIZE = 32


def load_model(device: torch.device):
    raw_package = torch_load_compat(DEFAULT_RAW, map_location="cpu")
    if checkpoint_kind(raw_package) != "raw_ssl":
        raise ValueError("--raw-checkpoint must be ssl_model_server.pt format")
    official_package = torch_load_compat(DEFAULT_OFFICIAL, map_location="cpu")
    if checkpoint_kind(official_package) != "official_embedding_slim":
        raise ValueError("official embedding slim checkpoint not found; run prepare_official_embedding_checkpoint.py first")
    model = reconstruct_backbone(
        raw_package, official_backbone_state(official_package), N_HIGHEST_PEAKS, device
    )
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    return model, weight, bias


def embed_file(model, weight, bias, hdf5_path: Path, device: torch.device) -> tuple[np.ndarray, dict]:
    with h5py.File(hdf5_path, "r") as handle:
        n = int(handle["spectrum"].shape[0])
        meta = {c: np.asarray(handle[c][:]) for c in ("scan_number", "precursor_mz", "charge", "RT")}
    rows = np.arange(n, dtype=np.int64)
    loader = DataLoader(
        SpectrumRows(hdf5_path, rows, N_HIGHEST_PEAKS),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    embs = []
    t0 = time.time()
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = batch.to(device=device, dtype=dtype)
            precursor = model(batch, None)[:, 0]
            e = F.normalize(F.linear(precursor, weight, bias), dim=-1).float().cpu().numpy()
            embs.append(e)
            if (i + 1) % 40 == 0:
                print(f"    {hdf5_path.stem}: {min((i+1)*BATCH_SIZE, n)}/{n} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    embs = np.concatenate(embs)
    meta["file_name"] = np.full(n, hdf5_path.stem, dtype=object)
    meta["row_in_file"] = rows
    return embs, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    args.out.mkdir(parents=True, exist_ok=True)

    print("[1] loading official DreaMS embedding model...", flush=True)
    t0 = time.time()
    model, weight, bias = load_model(device)
    print(f"    ok {time.time()-t0:.1f}s", flush=True)

    all_embs, all_meta = [], {k: [] for k in ("file_name", "scan_number", "precursor_mz", "charge", "RT", "row_in_file")}
    for hdf5_path in args.hdf5:
        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        print(f"[2] embedding {hdf5_path} ...", flush=True)
        t0 = time.time()
        embs, meta = embed_file(model, weight, bias, hdf5_path, device)
        print(f"    {len(embs)} spectra in {time.time()-t0:.0f}s", flush=True)
        all_embs.append(embs)
        for k in all_meta:
            all_meta[k].append(meta[k])

    embeddings = np.concatenate(all_embs)
    for k in all_meta:
        all_meta[k] = np.concatenate(all_meta[k])

    if not np.isfinite(embeddings).all():
        raise RuntimeError("non-finite embedding values detected")
    np.save(args.out / "embeddings.npy", embeddings.astype(np.float32))

    import pandas as pd
    manifest = pd.DataFrame({
        "file_name": all_meta["file_name"],
        "scan_number": all_meta["scan_number"].astype(np.int64),
        "precursor_mz": all_meta["precursor_mz"].astype(np.float64),
        "charge": all_meta["charge"].astype(np.int64),
        "RT": all_meta["RT"].astype(np.float64),
        "row_in_file": all_meta["row_in_file"].astype(np.int64),
    })
    manifest.to_csv(args.out / "manifest.csv", index=False)

    report = {
        "status": "msv100574_official_dreams_embeddings",
        "embedding_definition": "L2-normalized linear-head(frozen official backbone precursor token)",
        "n_spectra": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "files": [str(p) for p in args.hdf5],
        "n_highest_peaks": N_HIGHEST_PEAKS,
        "raw_checkpoint": str(DEFAULT_RAW),
        "official_checkpoint": str(DEFAULT_OFFICIAL),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
