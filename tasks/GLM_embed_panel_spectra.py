"""Embed GNPS panel spectra with a trained arm checkpoint.

Two encoder modes:
  synthetic : the trainer's SyntheticEncoder state dict (logic smoke path)
  dreams    : the real DreaMS backbone (server; same loading contract as
              the trainer)

Input rows come from the frozen panels (query rows + candidate rows);
output is an npz {rows, embeddings} aligned to the manifest row space, in
the exact format the arm scoring step consumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
PANEL_DIR = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction"
PANELS = ("identity_disjoint", "formula_disjoint")


def needed_rows() -> np.ndarray:
    rows = set()
    for panel in PANELS:
        with np.load(PANEL_DIR / f"panel_{panel}.npz") as z:
            rows.update(map(int, z["query_row"]))
            rows.update(map(int, z["candidate_row"]))
    arr = np.asarray(sorted(rows), dtype=np.int64)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--encoder", choices=("synthetic", "dreams"),
                    default="synthetic")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    rows = needed_rows()
    n = len(rows)
    print(f"panel spectra to embed: {n:,}", flush=True)

    if args.encoder == "synthetic":
        from GLM_train_orbit_boundary_encoder import SyntheticEncoder
        state = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False)["encoder"]
        model = SyntheticEncoder()
        model.load_state_dict(state)
        model.eval()
        # feature layout must match the trainer's spectra_feats convention:
        # deterministic per-row 32-dim pseudo-features (seeded, row-indexed)
        gen = torch.Generator().manual_seed(3407)
        feats = torch.randn(int(rows.max()) + 1, 32, generator=gen)
        with torch.no_grad():
            embs = torch.cat([model(feats[i:i + args.batch_size])
                              for i in range(0, feats.shape[0],
                                             args.batch_size)])
    else:
        from GLM_train_orbit_boundary_encoder import load_dreams_encoder
        from types import SimpleNamespace
        from GLM_score_challenger_models_on_gnps import parse_mgf_used
        ns = SimpleNamespace(official_checkpoint=None,
                             architecture_checkpoint=None,
                             noise_v1_checkpoint=None,
                             n_highest_peaks=100, lr=0.0, data=None)
        # server path: the real arguments are provided by the sbatch
        head = load_dreams_encoder(ns)
        state = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False)["encoder"]
        head.load_state_dict(state, strict=True)
        head.eval()
        spectra = parse_mgf_used(BENCH / "spectra.mgf", set(map(int, rows)))
        embs = np.zeros((int(rows.max()) + 1, head.head.in_features
                         if hasattr(head.head, "in_features") else 256),
                        dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                chunk = rows[i:i + args.batch_size]
                batch = torch.stack([
                    torch.as_tensor(np.asarray(spectra[int(r)][0]),
                                    dtype=torch.float32) for r in chunk])
                out = head(batch, charge=None).cpu().numpy()
                embs[chunk] = out

    embs_np = embs.numpy() if isinstance(embs, torch.Tensor) else embs
    norms = np.linalg.norm(embs_np[rows], axis=1, keepdims=True)
    unit = embs_np[rows] / np.clip(norms, 1e-12, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rows=rows,
                        embeddings=unit.astype(np.float32))
    print(f"written: {args.output} ({len(rows):,} rows, "
          f"dim {unit.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
