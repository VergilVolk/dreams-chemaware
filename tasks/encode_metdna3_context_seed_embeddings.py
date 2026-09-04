#!/usr/bin/env python
"""Encode the frozen MetDNA3 Level-1 seed spectrum pool with official DreaMS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.embed import load_embedder  # noqa: E402
from encode_bioaware_metdna3_dreams import encode_loader  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_context_seed_embeddings_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest_path = args.cache_dir / "external_spectra.csv.gz"
    tensor_path = args.cache_dir / "external_tensors.npz"
    report_path = args.cache_dir / "report.json"
    for path in (manifest_path, tensor_path, report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    cache_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not cache_report.get("formal") or cache_report.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("invalid MetDNA3 frozen cache contract")
    manifest = pd.read_csv(manifest_path)
    tensors = np.load(tensor_path, allow_pickle=False)["external_tensor"].astype(np.float32)
    if len(manifest) != len(tensors) or manifest.truth_ik14.astype(str).str.len().lt(14).any():
        raise RuntimeError("external seed manifest/tensor mismatch")
    device = torch.device(args.device)
    model, weight, bias = load_embedder(device=device, n_highest_peaks=args.n_highest_peaks)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tensors)), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    embeddings = encode_loader(loader, model, weight, bias, device)
    if len(embeddings) != len(manifest) or not np.isfinite(embeddings).all():
        raise RuntimeError("invalid seed embeddings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = args.output_dir / "manifest.csv.gz"
    embedding_out = args.output_dir / "embeddings.npy"
    manifest[[
        "truth_ik14", "truth_formula", "adduct", "polarity", "source_file",
        "spectrum_id", "spectrum_key",
    ]].to_csv(manifest_out, index=False)
    np.save(embedding_out, embeddings.astype(np.float32))
    report = {
        "status": "bioaware_metdna3_context_seed_embeddings_complete",
        "formal": True,
        "spectra": int(len(manifest)),
        "identities": int(manifest.truth_ik14.astype(str).nunique()),
        "embedding_dim": int(embeddings.shape[1]),
        "contracts": {
            "official_shared_encoder": True,
            "identity_used_only_for_seed_prototype_grouping": True,
            "P2b": "forbidden",
            "internal_validation_or_external_test_opened": False,
        },
        "provenance": {
            "cache_report_sha256": sha256(report_path),
            "source_manifest_sha256": sha256(manifest_path),
            "source_tensor_sha256": sha256(tensor_path),
            "manifest_sha256": sha256(manifest_out),
            "embeddings_sha256": sha256(embedding_out),
        },
        "claim_limit": "Execution cache for consumed HILIC development contexts; not a performance result.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
