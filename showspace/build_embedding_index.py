"""Build a checkpoint-aligned reference embedding index for Showspace.

Run this on a CUDA host for the full 231k-spectrum library, then copy the
``.npy`` and adjacent ``.json`` manifest to the public inference machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from inference_utils import SharedDreaMSEncoderAdapter, load_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("official_dreams", "e4a_shared", "e8_shared"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke-only partial index; public service rejects it by default")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(args.model, args.device)
    if not isinstance(model, SharedDreaMSEncoderAdapter):
        raise RuntimeError("index builder requires the state-only shared encoder backend")
    with h5py.File(args.data, "r") as handle:
        total = len(handle["spectrum"])
        rows = min(total, args.max_rows) if args.max_rows else total
        output = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=(rows, 1024))
        for left in range(0, rows, args.batch_size):
            right = min(rows, left + args.batch_size)
            spectra = [np.asarray(handle["spectrum"][row]).T for row in range(left, right)]
            precursor = [float(handle["precursor_mz"][row]) for row in range(left, right)]
            output[left:right] = model.embed_batch(spectra, precursor)
            if right == rows or right % (args.batch_size * 20) == 0:
                print(f"[index] {right:,}/{rows:,}", flush=True)
        output.flush()
    manifest = {
        "status": "showspace_model_aligned_embedding_index",
        "model_type": args.model,
        "model_fingerprint": model.fingerprint,
        "checkpoint": model.checkpoint_path.name,
        "checkpoint_sha256": model.fingerprint,
        "source_hdf5": str(args.data.resolve()),
        "source_hdf5_sha256": sha256(args.data),
        "rows": rows,
        "source_rows": total,
        "complete": rows == total,
        "shape": [rows, 1024],
        "dtype": "float32",
        "preprocessing": "training-identical top-100 clean spectrum",
        "inference_candidate_independent": True,
        "p2b_used": False,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
