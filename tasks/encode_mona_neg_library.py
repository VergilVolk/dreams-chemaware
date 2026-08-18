"""Embed the MONA-negative reference library with the official DreaMS embedding.

Reads data/models/mona_neg_full.mgf (standard MGF: SMILES, INCHIKEY, PEPMASS, peaks),
embeds each spectrum with the same official backbone + linear head used for the tissue
data, and caches embeddings.npy + manifest.csv for cosine top-k retrieval.

Usage (CPU):
    python tasks/encode_mona_neg_library.py --mgf data/models/mona_neg_full.mgf \
        --out data/models/mona_neg_dreams_emb --limit 200   # quick smoke
    python tasks/encode_mona_neg_library.py --mgf data/models/mona_neg_full.mgf \
        --out data/models/mona_neg_dreams_emb               # full library
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
    preprocess_spectrum,
    reconstruct_backbone,
)

DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
N_HIGHEST_PEAKS = 100
BATCH_SIZE = 64


def parse_mgf(path: Path) -> list[dict]:
    """Parse a standard MGF into per-record dicts (precursor_mz, smiles, inchikey, peaks)."""
    records: list[dict] = []
    cur: dict | None = None
    peaks: list[tuple[float, float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line == "BEGIN IONS":
                cur = {}
                peaks = []
            elif line == "END IONS":
                if cur is not None and peaks and cur.get("precursor_mz"):
                    arr = np.asarray(peaks, dtype=np.float32)  # (n, 2)
                    cur["peaks"] = arr.T  # (2, n)
                    records.append(cur)
                cur = None
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if k == "PEPMASS":
                    cur["precursor_mz"] = float(v.split()[0])
                elif k == "SMILES":
                    cur["smiles"] = v
                elif k == "INCHIKEY":
                    cur["inchikey"] = v
                elif k == "NAME":
                    cur["name"] = v
            elif cur is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return records


class LibraryDataset(Dataset):
    def __init__(self, records: list[dict], n_highest: int):
        self.records = records
        self.n_highest = n_highest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> torch.Tensor:
        r = self.records[i]
        return preprocess_spectrum(r["peaks"], r["precursor_mz"], self.n_highest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--out", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb")
    parser.add_argument("--limit", type=int, default=0, help="0 = full library")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    print("[1] parsing MGF...", flush=True)
    t0 = time.time()
    records = parse_mgf(args.mgf)
    print(f"    {len(records)} records in {time.time()-t0:.0f}s", flush=True)
    if args.limit:
        records = records[: args.limit]
        print(f"    limited to {len(records)}", flush=True)

    print("[2] loading official DreaMS embedding model...", flush=True)
    raw_package = torch_load_compat(DEFAULT_RAW, map_location="cpu")
    assert checkpoint_kind(raw_package) == "raw_ssl"
    official_package = torch_load_compat(DEFAULT_OFFICIAL, map_location="cpu")
    assert checkpoint_kind(official_package) == "official_embedding_slim"
    model = reconstruct_backbone(
        raw_package, official_backbone_state(official_package), N_HIGHEST_PEAKS, device
    )
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()

    print("[3] embedding library...", flush=True)
    loader = DataLoader(LibraryDataset(records, N_HIGHEST_PEAKS), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    embs = []
    dtype = next(model.parameters()).dtype
    t0 = time.time()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = batch.to(device=device, dtype=dtype)
            precursor = model(batch, None)[:, 0]
            e = F.normalize(F.linear(precursor, weight, bias), dim=-1).float().cpu().numpy()
            embs.append(e)
            if (i + 1) % 100 == 0:
                print(f"    {min((i+1)*BATCH_SIZE, len(records))}/{len(records)} ({time.time()-t0:.0f}s)", flush=True)
    embs = np.concatenate(embs)
    if not np.isfinite(embs).all():
        raise RuntimeError("non-finite embedding values")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", embs.astype(np.float32))
    import pandas as pd
    manifest = pd.DataFrame({
        "smiles": [r.get("smiles", "") for r in records],
        "inchikey": [r.get("inchikey", "") for r in records],
        "name": [r.get("name", "") for r in records],
        "precursor_mz": [r.get("precursor_mz", float("nan")) for r in records],
    })
    manifest.to_csv(args.out / "manifest.csv", index=False)
    report = {
        "status": "mona_neg_dreams_embeddings",
        "n_spectra": int(len(embs)),
        "embedding_dim": int(embs.shape[1]),
        "limit": args.limit,
        "source": str(args.mgf),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
