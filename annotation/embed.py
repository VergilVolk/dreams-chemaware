"""M1 -- DreaMS embedding of query spectra and library spectra.

Implements the official embedding formula (Bushuiev et al., Nat Biotechnol 2025,
DOI 10.1038/s41587-025-02663-3):

    precursor = backbone(batch, None)[:, 0]      # position-0 precursor token
    embedding = L2_normalize( linear(precursor, weight, bias) )

Both query (tissue) and reference (library) spectra are embedded with the *same*
frozen backbone + linear head, so their cosine similarity is meaningful.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ._inference import (
    SpectrumRows,
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    preprocess_spectrum,
    reconstruct_backbone,
    torch_load_compat,
)
from .params import Params

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"

BATCH_SIZE = 32


def load_embedder(
    device: str | torch.device = "cpu",
    raw_path: Path = DEFAULT_RAW,
    official_path: Path = DEFAULT_OFFICIAL,
    n_highest_peaks: int = 100,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Return (backbone, head_weight, head_bias) ready for embedding."""
    device = torch.device(device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    raw_package = torch_load_compat(raw_path, map_location="cpu")
    if checkpoint_kind(raw_package) != "raw_ssl":
        raise ValueError("raw checkpoint is not ssl_model_server.pt format")
    official_package = torch_load_compat(official_path, map_location="cpu")
    if checkpoint_kind(official_package) != "official_embedding_slim":
        raise ValueError("official checkpoint is not official_embedding_slim format")

    model = reconstruct_backbone(
        raw_package, official_backbone_state(official_package), n_highest_peaks, device
    )
    head = official_head_state(official_package)
    dtype = next(model.parameters()).dtype
    weight = head["weight"].to(device=device, dtype=dtype)
    bias = head["bias"].to(device=device, dtype=dtype)
    model.eval()
    return model, weight, bias


def _embed_batches(loader: DataLoader, model, weight, bias, device, n_expected: int):
    embs = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device=device, dtype=dtype)
            precursor = model(batch, None)[:, 0]
            e = F.normalize(F.linear(precursor, weight, bias), dim=-1).float().cpu().numpy()
            embs.append(e)
    return np.concatenate(embs).astype(np.float32)


def embed_hdf5(
    hdf5_path: Path,
    model,
    weight,
    bias,
    device: str | torch.device = "cpu",
    n_highest_peaks: int = 100,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed every MS2 spectrum in an hdf5 file (written by dreams.utils.data.MSData).

    Returns (embeddings [N x 1024], manifest DataFrame with scan_number,
    precursor_mz, charge, RT, file_name, row_in_file)."""
    device = torch.device(device)
    with h5py.File(hdf5_path, "r") as handle:
        n = int(handle["spectrum"].shape[0])
        meta = {c: np.asarray(handle[c][:]) for c in ("scan_number", "precursor_mz", "charge", "RT")}
    rows = np.arange(n, dtype=np.int64)
    loader = DataLoader(
        SpectrumRows(hdf5_path, rows, n_highest_peaks),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    embs = _embed_batches(loader, model, weight, bias, device, n)
    if not np.isfinite(embs).all():
        raise RuntimeError("non-finite embedding values")
    manifest = pd.DataFrame({
        "file_name": np.full(n, hdf5_path.stem, dtype=object),
        "scan_number": meta["scan_number"].astype(np.int64),
        "precursor_mz": meta["precursor_mz"].astype(np.float64),
        "charge": meta["charge"].astype(np.int64),
        "RT": meta["RT"].astype(np.float64),
        "row_in_file": rows,
    })
    return embs, manifest


def embed_records(
    records: list[dict],
    model,
    weight,
    bias,
    device: str | torch.device = "cpu",
    n_highest_peaks: int = 100,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed a list of library records, each ``{"peaks": (2,n) ndarray,
    "precursor_mz": float}``. Returns [N x 1024] L2-normalized embeddings."""
    device = torch.device(device)

    class _Ds(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, i):
            r = records[i]
            return preprocess_spectrum(r["peaks"], r["precursor_mz"], n_highest_peaks)

    loader = DataLoader(_Ds(), batch_size=batch_size, shuffle=False, num_workers=0)
    embs = _embed_batches(loader, model, weight, bias, device, len(records))
    if not np.isfinite(embs).all():
        raise RuntimeError("non-finite embedding values")
    return embs
