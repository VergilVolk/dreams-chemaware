"""Encode the external ring-balanced pilot with the official DreaMS weights."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

import pilot_multilevel_factor_activations as multi  # noqa: E402
from e1_checkpoint_io import official_head_state  # noqa: E402
from pilot_paired_layer_cka import preprocess_spectrum  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/external_ring_balanced_embeddings"))
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    head = official_head_state(official_package)
    head_weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    head_bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    dtype = next(model.parameters()).dtype
    reports = {}
    for split in ("discovery", "confirmation"):
        source = np.load(args.pilot_dir / f"{split}_spectra.npz")
        spectra = source["spectra"]
        precursor = source["precursor_mz"]
        tensors = []
        for unit in range(len(spectra)):
            for view in range(2):
                tensors.append(preprocess_spectrum(
                    spectra[unit, view], float(precursor[unit, view]), args.n_highest_peaks
                ))
        tensors = torch.stack(tensors)
        loader = DataLoader(TensorDataset(tensors), batch_size=args.batch_size, shuffle=False)
        outputs = []
        backbone_outputs = []
        with torch.inference_mode():
            for (batch,) in loader:
                precursor_token = model(batch.to(device=device, dtype=dtype), None)[:, 0]
                encoded = F.linear(precursor_token, head_weight, head_bias)
                backbone_outputs.append(precursor_token.float().cpu().numpy())
                outputs.append(encoded.float().cpu().numpy())
        embeddings = np.concatenate(outputs).reshape(len(spectra), 2, -1)
        backbone_tokens = np.concatenate(backbone_outputs).reshape(len(spectra), 2, -1)
        norms = np.linalg.norm(embeddings, axis=-1)
        normalized = embeddings / np.clip(norms[..., None], 1e-12, None)
        np.save(args.output_dir / f"{split}_official.npy", normalized.astype(np.float32))
        np.save(args.output_dir / f"{split}_backbone.npy", backbone_tokens.astype(np.float32))
        reports[split] = {
            "units": len(spectra), "spectra": 2 * len(spectra),
            "embedding_dim": embeddings.shape[-1],
            "norm_min": float(norms.min()), "norm_median": float(np.median(norms)),
            "all_finite": bool(np.isfinite(normalized).all()),
        }
    del model, official_package, raw_package
    gc.collect()
    report = {
        "status": "external_ring_balanced_official_embeddings_with_projection_head",
        "checkpoint": str(args.official_checkpoint),
        "embedding_definition": "L2-normalized official linear-head(backbone precursor token)",
        "preprocessing": f"Established DreaMS preprocessing with {args.n_highest_peaks} peaks",
        "splits": reports,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
