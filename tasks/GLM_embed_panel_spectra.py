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
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="weights checkpoint overriding the backbone")
    ap.add_argument("--encoder", choices=("synthetic", "dreams"),
                    default="synthetic")
    ap.add_argument("--official-checkpoint", type=Path)
    ap.add_argument("--architecture-checkpoint", type=Path)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
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
        from train_e1_identity import load_base_model, preprocess_spectrum
        from dreams.models.heads.heads import ContrastiveHead
        import types
        # matchms shim (same as trainer)
        _shim = types.ModuleType("matchms.Spikes")
        class _Spikes:
            def __init__(self, mz=None, intensities=None):
                self.mz, self.intensities = mz, intensities
        _shim.Spikes = _Spikes
        sys.modules.setdefault("matchms.Spikes", _shim)
        import matchms.similarity as _ms
        if not hasattr(_ms, "ModifiedCosine"):
            _ms.ModifiedCosine = type("ModifiedCosine", (), {})

        if args.architecture_checkpoint is None:
            raise RuntimeError("dreams mode requires --official-checkpoint "
                               "and --architecture-checkpoint")
        initialized, _ = load_base_model(
            args.architecture_checkpoint, args.architecture_checkpoint,
            torch.device("cpu"), args.n_highest_peaks)
        state = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False)
        sd = state.get("encoder", state) if isinstance(state, dict) else state
        sd = sd.get("backbone", sd) if isinstance(sd, dict) else sd
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        if isinstance(sd, dict) and sd:
            initialized.backbone.load_state_dict(sd, strict=False)
            print(f"  weights override loaded from {args.checkpoint}")
        model = ContrastiveHead(initialized.backbone, 1e-6, 0.0,
                                triplet_loss_margin=0.1)
        if hasattr(initialized, 'head'):
            try:
                model.head.load_state_dict(initialized.head.state_dict(),
                                           strict=True)
            except Exception:
                pass
        model.eval()
        head = model
        from GLM_score_challenger_models_on_gnps import parse_mgf_used
        spectra = parse_mgf_used(BENCH / "spectra.mgf", set(map(int, rows)))
        from train_e1_identity import preprocess_spectrum  # noqa: PLC0415
        embs = []
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                chunk = rows[i:i + args.batch_size]
                batch = torch.stack([
                    preprocess_spectrum(
                        np.vstack([np.asarray(spectra[int(r)][0]),
                                   np.asarray(spectra[int(r)][1])]),
                        float(spectra[int(r)][2]), args.n_highest_peaks)
                    for r in chunk])
                out = head(batch, charge=None).cpu().numpy()
                embs.append(out)
        embs = np.concatenate(embs)
        full = np.zeros((int(rows.max()) + 1, embs.shape[1]),
                        dtype=np.float32)
        full[rows] = embs
        embs_np = full

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
