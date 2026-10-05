#!/usr/bin/env python
"""Encode one deterministic shard of all MassSpecGym spectra with one checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from evaluate_noise_dreams_native import encode_rows, sha256_file
from train_e1_identity import load_base_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full MassSpecGym encoding requires an allocated GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid MassSpecGym shard")
    for path in (args.data, args.checkpoint, args.architecture_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    with h5py.File(args.data, "r") as handle:
        total_rows = len(handle["spectrum"])
    rows = np.arange(args.shard_index, total_rows, args.shard_count, dtype=np.int64)
    model, checkpoint_kind = load_base_model(
        args.checkpoint, args.architecture_checkpoint, torch.device("cuda"), 100,
    )
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    embeddings = encode_rows(
        model, rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label=f"full-msg-shard-{args.shard_index}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.", suffix=".npz",
        dir=args.output.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez(temporary, rows=rows, embeddings=embeddings)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "noise_massspecgym_checkpoint_shard_complete",
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "total_rows": total_rows,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "rows": int(len(rows)),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
