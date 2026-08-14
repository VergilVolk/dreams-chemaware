"""Encode official DreaMS peak tokens into a fixed random projection.

The projection is label-free and frozen before reranker fitting.  It preserves
contextual token geometry approximately while keeping the CPU/disk footprint
small enough for the large formula-isolated cohort.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pilot_multilevel_factor_activations as multi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--output-root", type=Path, default=Path("data/validation/official_peak_tokens"))
    parser.add_argument("--data", type=Path, default=multi.DEFAULT_DATA)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw, multi.official_backbone_state(official), device)
    model.eval()
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    projection = torch.randn(
        model.d_model, args.projection_dim, generator=generator, dtype=torch.float32
    ) / np.sqrt(args.projection_dim)
    projection = projection.to(device=device, dtype=dtype)
    np.save(args.output_root / "projection.npy", projection.float().cpu().numpy())
    reports = {}
    for split in args.splits:
        directory = args.embedding_root / f"large_observability_embeddings_{split}"
        manifest = pd.read_csv(directory / "manifest.csv")
        rows = manifest["hdf5_row"].to_numpy(np.int64)
        loader = DataLoader(
            multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        token_parts, mz_parts, intensity_parts, valid_parts = [], [], [], []
        with torch.inference_mode():
            for position, batch in enumerate(loader, start=1):
                batch = batch.to(device=device, dtype=dtype)
                tokens = model(batch, None)[:, 1:, :]
                tokens = F.normalize(tokens, dim=-1)
                projected = F.normalize(tokens @ projection, dim=-1)
                valid = batch[:, 1:, 0] > 0
                projected = projected.masked_fill(~valid.unsqueeze(-1), 0)
                token_parts.append(projected.half().cpu().numpy())
                mz_parts.append(batch[:, 1:, 0].float().cpu().numpy())
                intensity_parts.append(batch[:, 1:, 1].float().cpu().numpy())
                valid_parts.append(valid.cpu().numpy())
                if position % 100 == 0:
                    print(f"  {split}: {position}/{len(loader)} batches", flush=True)
        output = args.output_root / split
        output.mkdir(parents=True, exist_ok=True)
        token_values = np.concatenate(token_parts)
        mz_values = np.concatenate(mz_parts)
        intensity_values = np.concatenate(intensity_parts)
        valid_values = np.concatenate(valid_parts)
        np.save(output / "peak_tokens_f16.npy", token_values)
        np.save(output / "peak_mz.npy", mz_values)
        np.save(output / "peak_intensity.npy", intensity_values)
        np.save(output / "peak_valid.npy", valid_values)
        manifest.to_csv(output / "manifest.csv", index=False)
        reports[split] = {
            "spectra": len(manifest), "shape": list(token_values.shape),
            "valid_peak_tokens": int(valid_values.sum()),
        }
        del token_parts, mz_parts, intensity_parts, valid_parts, token_values
        gc.collect()
    report = {
        "status": "official_peak_token_random_projection",
        "checkpoint": str(args.official_checkpoint),
        "source_tokens": "official fine-tuned backbone final-layer fragment tokens",
        "projection": f"label-free Gaussian random projection, seed={args.seed}",
        "projection_dim": args.projection_dim, "splits": reports,
        "claim_limit": "Projected token similarity approximates final-layer geometry; it is not a learned chemical annotation.",
    }
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
