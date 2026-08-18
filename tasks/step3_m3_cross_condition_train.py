"""M3: head-only cross-condition FN training on the M2 cohort.

Freeze the DreaMS backbone at official weights and train ONLY the 1024->1024
projection head so that cross-condition SAME-molecule pairs move closer (toward
the same-condition floor), while condition-matched different-molecule negatives
stay put.

Loss (plan §4/§5 -- L1 consistency + FP margin guard + preserve, head-only):
  L_cond     = mean(1 - cos(anchor, positive))                 # cross-condition pull (FN)
  L_margin   = mean(relu(margin - (cos(pos) - cos(neg))))      # FP separation guard
  L_preserve = mean(relu(floor - cos(anchor, teacher)))        # stay near official

The FP guard is now a *relative* margin (pos must stay `margin` above neg), not
the earlier absolute ceiling. It directly optimises the benchmark's margin metric
and pushes the different-molecule negative DOWN while the consistency term pulls
the positive UP -- the pure `1-cos` consistency over-generalised (seed 2: val neg
cosine 0.327->0.408, margin 0.496->0.448) because it had no negative-push term.

margin (default 0.40) sits above the train cohort's official margin (~0.29) so
the term has real gradient at init, and below the val fold's official margin
(~0.50) so it does not over-squeeze a separation the official head already has.
preserve_weight is bumped to 2.0 because seed 2 (weight 1.0) let preserve fall to
0.953 (< 0.995 floor) -- that head drift is what pulled the different-molecule
negatives up on val.

The backbone is frozen, so its precursor embeddings are computed ONCE per unique
spectrum and cached to disk (signature excludes --seed), making the 3-seed sweep
pay the forward cost once. The final gate is the LOCKED val-fold benchmark, run
separately with --head-checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from pilot_paired_layer_cka import preprocess_spectrum
from train_e1_identity import cpu_state_dict, load_base_model


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_MANIFEST = ROOT / "data/validation/cross_condition_m3/train_pairs.json"
DEFAULT_OUTPUT = ROOT / "data/validation/cross_condition_m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # loss weights / knobs
    parser.add_argument("--cond-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.40,
                        help="required pos-neg separation (relative FP guard)")
    parser.add_argument("--preserve-weight", type=float, default=2.0)
    parser.add_argument("--preserve-floor", type=float, default=0.995)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subset (150 pairs, 1 epoch) for end-to-end check")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def backbone_embeddings(model, tokens: list[torch.Tensor], device: torch.device,
                        batch_size: int, label: str = "") -> torch.Tensor:
    model.backbone.eval()
    out = []
    n = len(tokens)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = torch.stack(tokens[start:start + batch_size]).to(device)
            out.append(model.backbone(batch, None)[:, 0, :].cpu())
            done = min(start + batch_size, n)
            if done % (batch_size * 5) == 0 or done == n:
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done) if done else 0.0
                print(f"  [{label}] {done}/{n} spectra ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    return torch.cat(out, dim=0)


def load_manifest(path: Path, smoke: bool) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    pairs = manifest["cross_pairs"]
    if smoke:
        pairs = pairs[:150]
    if not pairs:
        raise RuntimeError(f"Empty cohort in {path}; run M2 first")
    return {"pairs": pairs, "audit": manifest.get("audit", {}), "fold": manifest.get("fold")}


def unique_rows_and_index(pairs: list[dict]) -> tuple[list[int], dict[int, int]]:
    rows_set = set()
    for p in pairs:
        rows_set.add(p["rows"][0])
        rows_set.add(p["rows"][1])
        rows_set.add(p["negative_row"])
    unique = sorted(rows_set)
    idx_map = {int(r): k for k, r in enumerate(unique)}
    return unique, idx_map


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[m3] device={device} seed={args.seed}", flush=True)

    manifest = load_manifest(args.manifest, args.smoke)
    pairs = manifest["pairs"]
    unique_rows, idx_map = unique_rows_and_index(pairs)
    anchor_idx = [idx_map[p["rows"][0]] for p in pairs]
    pos_idx = [idx_map[p["rows"][1]] for p in pairs]
    neg_idx = [idx_map[p["negative_row"]] for p in pairs]
    print(f"[m3] {len(pairs)} pairs, {len(unique_rows)} unique spectra "
          f"(fold={manifest['fold']})", flush=True)

    # ---- load model (backbone frozen + official head init) ----
    model, kind = load_base_model(args.official_checkpoint, args.architecture_ckpt, device, args.n_highest_peaks)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    # ---- tokenise the unique spectra ----
    with h5py.File(args.data, "r") as handle:
        def tok(row: int) -> torch.Tensor:
            return preprocess_spectrum(
                np.asarray(handle["spectrum"][row]),
                float(handle["precursor_mz"][row]),
                args.n_highest_peaks,
            )
        tokens = [tok(r) for r in unique_rows]

    # ---- frozen backbone embeddings (cached; signature excludes --seed) ----
    manifest_bytes = args.manifest.read_bytes()
    sig = hashlib.md5(
        f"v1:{manifest_bytes.hex()}:{len(pairs)}:{args.n_highest_peaks}:{args.data.stat().st_size}".encode()
    ).hexdigest()[:16]
    cache_path = args.output_dir / f"backbone_embeddings_{sig}.npz"
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            Z = torch.from_numpy(data["Z_unique"]).clone()
            print(f"[embed] loaded cached embeddings ({cache_path.name})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[embed] cache unreadable ({exc}), recomputing", flush=True)
            Z = backbone_embeddings(model, tokens, device, args.batch_size, "backbone")
            np.savez(cache_path, Z_unique=Z.numpy())
    else:
        print(f"[embed] computing frozen backbone embeddings for {len(unique_rows)} spectra...", flush=True)
        t0 = time.time()
        Z = backbone_embeddings(model, tokens, device, args.batch_size, "backbone")
        np.savez(cache_path, Z_unique=Z.numpy())
        print(f"[embed] cached -> {cache_path.name} ({time.time() - t0:.0f}s)", flush=True)

    Zq = Z[anchor_idx].to(device)
    Zp = Z[pos_idx].to(device)
    Zn = Z[neg_idx].to(device)
    n_train = len(Zq)

    # ---- official-head reference (baseline + teacher) ----
    with torch.no_grad():
        official = lambda z: F.normalize(model.head(z), dim=-1)
        Q_official = official(Zq).clone()          # teacher for preserve
        pos_cos_official = (Q_official * official(Zp)).sum(1).mean().item()
        neg_cos_official = (Q_official * official(Zn)).sum(1).mean().item()

    print(f"[init] official head: cross_cos={pos_cos_official:.4f} "
          f"neg_cos={neg_cos_official:.4f} | margin={pos_cos_official - neg_cos_official:.4f} "
          f"| target margin={args.margin:.4f}", flush=True)

    # ---- train head ----
    head = model.head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n_train)
        losses, conds, margins, preserves = [], [], [], []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            q = F.normalize(head(Zq[idx]), dim=-1)
            p = F.normalize(head(Zp[idx]), dim=-1)
            n = F.normalize(head(Zn[idx]), dim=-1)
            pos_cos = (q * p).sum(1)
            neg_cos = (q * n).sum(1)
            pres_cos = (q * Q_official[idx]).sum(1)

            cond = (1.0 - pos_cos).mean()
            margin_loss = F.relu(args.margin - (pos_cos - neg_cos)).mean()
            preserve = F.relu(args.preserve_floor - pres_cos).mean()
            loss = (args.cond_weight * cond + args.margin_weight * margin_loss
                    + args.preserve_weight * preserve)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            conds.append(float(cond.detach()))
            margins.append(float(margin_loss.detach()))
            preserves.append(float(preserve.detach()))
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)), "cond": float(np.mean(conds)),
            "margin": float(np.mean(margins)), "preserve": float(np.mean(preserves)),
        })
        print(f"  epoch {epoch:02d} loss={history[-1]['loss']:.5f} cond={history[-1]['cond']:.5f} "
              f"margin={history[-1]['margin']:.5f} preserve={history[-1]['preserve']:.5f}", flush=True)

    # ---- in-run eval on the cohort (train fold; final gate is the val benchmark) ----
    with torch.no_grad():
        head.eval()
        q = F.normalize(head(Zq), dim=-1)
        p = F.normalize(head(Zp), dim=-1)
        n = F.normalize(head(Zn), dim=-1)
        cross_cos = (q * p).sum(1).mean().item()
        neg_cos = (q * n).sum(1).mean().item()
        preserve_cos = (q * Q_official).sum(1).mean().item()

    summary = {
        "status": "cross_condition_m3",
        "seed": args.seed, "kind": kind,
        "n_pairs": len(pairs), "n_unique_spectra": len(unique_rows),
        "fold": manifest["fold"],
        "config": {
            "lr": args.lr, "epochs": args.epochs, "weight_decay": args.weight_decay,
            "cond_weight": args.cond_weight, "margin_weight": args.margin_weight,
            "margin": args.margin,
            "preserve_weight": args.preserve_weight, "preserve_floor": args.preserve_floor,
        },
        "in_run_metrics": {
            "official": {"cross_cosine": pos_cos_official, "negative_cosine": neg_cos_official,
                         "margin": pos_cos_official - neg_cos_official},
            "trained": {"cross_cosine": cross_cos, "negative_cosine": neg_cos,
                        "margin": cross_cos - neg_cos, "preserve_cosine": preserve_cos},
        },
        "training_history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint = {
        "format": "e1_identity_v1",
        "stage": "cross_condition_m3",
        "epoch": args.epochs,
        "seed": args.seed,
        "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
        "base_checkpoint": str(args.official_checkpoint.resolve()),
        "backbone_state_dict": cpu_state_dict(model.backbone),
        "head_state_dict": cpu_state_dict(model.head),
        "config": {"n_highest_peaks": args.n_highest_peaks},
    }
    torch.save(checkpoint, run_dir / "best_m3.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved checkpoint: {run_dir / 'best_m3.pt'}", flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
