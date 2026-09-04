#!/usr/bin/env python
"""Encode all ambiguity-free MetDNA3 Level-1 feature spectra for data-layer edges."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotation.embed import load_embedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/bioaware_metdna3_data_layer_embeddings.npz"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = json.loads((args.cache_dir / "report.json").read_text(encoding="utf-8"))
    if not report.get("formal") or report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("invalid data-layer cache contract")
    tensors = np.load(args.cache_dir / "external_tensors.npz", allow_pickle=False)["external_tensor"]
    if len(tensors) != report["external_level1_spectra"]:
        raise RuntimeError("external tensor count mismatch")
    if args.output.exists():
        raise RuntimeError(f"fail-closed: {args.output}")
    device = torch.device(args.device)
    model, weight, bias = load_embedder(device=device, n_highest_peaks=100)
    dtype = next(model.parameters()).dtype
    embeddings = []
    loader = DataLoader(TensorDataset(torch.from_numpy(tensors)), batch_size=args.batch_size)
    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device=device, dtype=dtype)
            z = F.normalize(F.linear(model(x, None)[:, 0], weight, bias), dim=-1)
            embeddings.append(z.float().cpu().numpy())
    embedding = np.concatenate(embeddings).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise RuntimeError("non-finite data-layer embedding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embedding=embedding)
    print(json.dumps({
        "status": "bioaware_metdna3_data_layer_embeddings_complete",
        "spectra": int(len(embedding)), "identities": int(report["external_level1_identities"]),
        "source": "official DreaMS, same encoder as candidate-ranking baseline",
        "P2b": "forbidden",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

