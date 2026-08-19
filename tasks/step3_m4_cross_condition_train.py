"""M4: unfreeze the last Transformer layer(s) for cross-condition FN training.

M3 (head-only) failed the FP guard: a 1024x1024 linear head cannot distinguish
"cross-condition same molecule" from "different molecule" because the
condition-specific information is entangled in the backbone's peak-level
features -- a linear transform moves everything together.

M4 unfreezes the last K Transformer layers (atts[-K:], ffs[-K:], and their
LayerNorms) so the backbone itself learns to suppress condition-specific peaks.
But the v2 loss (relative margin) still failed: unfreezing gave the model more
freedom to CONTRACT the whole space -- val cross 0.823->0.873 (+0.050) AND
val neg 0.327->0.424 (+0.097), so margin shrank 0.496->0.449. The relative term
relu(margin - (pos-neg)) stays satisfied while pos and neg rise together.

Loss v3 replaces the relative margin with ABSOLUTE anchors so the model cannot
contract:
  L_cond     = mean(relu(pos_floor - cos(pos)))             # pull cross UP, bounded at floor
  L_neg      = mean(relu(cos(neg) - neg_ceiling))           # push different-molecule DOWN (absolute)
  L_preserve = mean(relu(floor - cos(anchor, teacher)))     # stay near official

teacher = official frozen backbone + official head, computed ONCE before
unfreezing. The training loop forwards the FULL backbone each step (frozen
prefix + trainable tail) and backprops only through the tail + head (the frozen
prefix params have requires_grad=False).

The final gate is the LOCKED val-fold benchmark, run separately with
--head-checkpoint (which now ALSO reloads the modified backbone).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

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
DEFAULT_OUTPUT = ROOT / "data/validation/cross_condition_m4_v3"


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
    parser.add_argument("--lr", type=float, default=1e-4, help="head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-5,
                        help="learning rate for the unfrozen backbone tail")
    parser.add_argument("--unfreeze-layers", type=int, default=1,
                        help="number of last Transformer layers to unfreeze (1 or 2)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # loss weights / knobs (v3: absolute anchors)
    parser.add_argument("--cond-weight", type=float, default=1.0)
    parser.add_argument("--pos-floor", type=float, default=0.88,
                        help="pull cross-condition cosine up to this floor, then stop")
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--neg-ceiling", type=float, default=0.35,
                        help="push different-molecule cosine back down if it exceeds this (absolute FP guard)")
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


def backbone_embeddings(model, tokens, device, batch_size, label=""):
    """Forward all tokens through the (frozen) backbone; returns [N, d_model]."""
    model.backbone.eval()
    out = []
    n = len(tokens)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = torch.stack(tokens[start:start + batch_size]).to(device)
            out.append(model.backbone(batch, None)[:, 0, :].cpu())
            done = min(start + batch_size, n)
            if done % (batch_size * 20) == 0 or done == n:
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done) if done else 0.0
                print(f"  [{label}] {done}/{n} spectra ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    return torch.cat(out, dim=0)


def unfreeze_backbone_tail(model, k: int) -> int:
    """Unfreeze the last `k` Transformer layers (att/ff + their LayerNorms)."""
    enc = model.backbone.transformer_encoder
    n = enc.n_layers
    k = min(max(1, k), n)
    n_params = 0
    for i in range(n - k, n):
        for sub in (enc.atts[i], enc.ffs[i]):
            for p in sub.parameters():
                p.requires_grad = True
                n_params += p.numel()
        for s in (enc.scales[2 * i], enc.scales[2 * i + 1]):
            for p in s.parameters():
                p.requires_grad = True
                n_params += p.numel()
    if getattr(enc, "pre_norm", False):
        for p in enc.scales[-1].parameters():
            p.requires_grad = True
            n_params += p.numel()
    return n_params


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
    print(f"[m4] device={device} seed={args.seed} unfreeze_layers={args.unfreeze_layers} "
          f"backbone_lr={args.backbone_lr}", flush=True)

    manifest = load_manifest(args.manifest, args.smoke)
    pairs = manifest["pairs"]
    unique_rows, idx_map = unique_rows_and_index(pairs)
    anchor_idx = torch.tensor([idx_map[p["rows"][0]] for p in pairs], dtype=torch.long)
    pos_idx = torch.tensor([idx_map[p["rows"][1]] for p in pairs], dtype=torch.long)
    neg_idx = torch.tensor([idx_map[p["negative_row"]] for p in pairs], dtype=torch.long)
    print(f"[m4] {len(pairs)} pairs, {len(unique_rows)} unique spectra "
          f"(fold={manifest['fold']})", flush=True)

    # ---- load model (backbone frozen first, head trainable) ----
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
    T_all = torch.stack(tokens)  # [N_unique, peaks, 2], stays on CPU

    # ---- teacher: official frozen backbone + official head, computed ONCE ----
    Z_official = backbone_embeddings(model, tokens, device, args.batch_size, "teacher")
    with torch.inference_mode():
        Q_official = F.normalize(model.head(Z_official.to(device)), dim=-1).cpu()  # [N_unique, D]
    pos_cos_official = (Q_official[anchor_idx] * Q_official[pos_idx]).sum(1).mean().item()
    neg_cos_official = (Q_official[anchor_idx] * Q_official[neg_idx]).sum(1).mean().item()
    Q_teacher = Q_official[anchor_idx].to(device)  # [n_pairs, D] per-pair anchor teacher

    print(f"[init] official: cross_cos={pos_cos_official:.4f} neg_cos={neg_cos_official:.4f} "
          f"| margin={pos_cos_official - neg_cos_official:.4f} | pos_floor={args.pos_floor:.4f} "
          f"neg_ceiling={args.neg_ceiling:.4f}", flush=True)

    # ---- unfreeze last K backbone layers ----
    n_unfrozen = unfreeze_backbone_tail(model, args.unfreeze_layers)
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    print(f"[m4] unfroze last {args.unfreeze_layers} layer(s): {n_unfrozen:,} backbone params trainable",
          flush=True)

    head = model.head
    optimizer = torch.optim.AdamW(
        [
            {"params": list(head.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": trainable_backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
        ],
    )
    n_train = len(pairs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.backbone.train()
        head.train()
        perm = torch.randperm(n_train)
        losses, conds, negs, preserves = [], [], [], []
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            a = anchor_idx[idx]
            p = pos_idx[idx]
            n = neg_idx[idx]
            rows_local = torch.unique(torch.cat([a, p, n]))
            z_rows = model.backbone(T_all[rows_local].to(device), None)[:, 0, :]  # [B, d_model]
            pos_map = {int(r): j for j, r in enumerate(rows_local.tolist())}
            za = z_rows[[pos_map[int(r)] for r in a.tolist()]]
            zp = z_rows[[pos_map[int(r)] for r in p.tolist()]]
            zn = z_rows[[pos_map[int(r)] for r in n.tolist()]]

            q = F.normalize(head(za), dim=-1)
            p_emb = F.normalize(head(zp), dim=-1)
            n_emb = F.normalize(head(zn), dim=-1)
            pos_cos = (q * p_emb).sum(1)
            neg_cos = (q * n_emb).sum(1)
            pres_cos = (q * Q_teacher[idx]).sum(1)

            cond = F.relu(args.pos_floor - pos_cos).mean()
            neg_guard = F.relu(neg_cos - args.neg_ceiling).mean()
            preserve = F.relu(args.preserve_floor - pres_cos).mean()
            loss = (args.cond_weight * cond + args.neg_weight * neg_guard
                    + args.preserve_weight * preserve)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            conds.append(float(cond.detach()))
            negs.append(float(neg_guard.detach()))
            preserves.append(float(preserve.detach()))
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)), "cond": float(np.mean(conds)),
            "neg_guard": float(np.mean(negs)), "preserve": float(np.mean(preserves)),
        })
        print(f"  epoch {epoch:02d} loss={history[-1]['loss']:.5f} cond={history[-1]['cond']:.5f} "
              f"neg={history[-1]['neg_guard']:.5f} preserve={history[-1]['preserve']:.5f}", flush=True)

    # ---- in-run eval on the cohort (train fold; final gate is the val benchmark) ----
    #     Batch the backbone forward: the unbatched 4765-spectra call OOMs GPU
    #     (fourier features + graphormer dists blow up ~10 GiB in one shot).
    with torch.inference_mode():
        model.backbone.eval()
        head.eval()
        z_all = backbone_embeddings(model, tokens, device, args.batch_size, "eval")
        q_all = F.normalize(head(z_all.to(device)), dim=-1).cpu()
        cross_cos = (q_all[anchor_idx] * q_all[pos_idx]).sum(1).mean().item()
        neg_cos = (q_all[anchor_idx] * q_all[neg_idx]).sum(1).mean().item()
        preserve_cos = (q_all[anchor_idx] * Q_official[anchor_idx]).sum(1).mean().item()

    summary = {
        "status": "cross_condition_m4",
        "seed": args.seed, "kind": kind,
        "n_pairs": len(pairs), "n_unique_spectra": len(unique_rows),
        "fold": manifest["fold"],
        "config": {
            "lr": args.lr, "backbone_lr": args.backbone_lr,
            "unfreeze_layers": args.unfreeze_layers,
            "epochs": args.epochs, "weight_decay": args.weight_decay,
            "cond_weight": args.cond_weight, "pos_floor": args.pos_floor,
            "neg_weight": args.neg_weight, "neg_ceiling": args.neg_ceiling,
            "preserve_weight": args.preserve_weight, "preserve_floor": args.preserve_floor,
        },
        "n_unfrozen_backbone_params": n_unfrozen,
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
        "stage": "cross_condition_m4",
        "epoch": args.epochs,
        "seed": args.seed,
        "unfreeze_layers": args.unfreeze_layers,
        "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
        "base_checkpoint": str(args.official_checkpoint.resolve()),
        "backbone_state_dict": cpu_state_dict(model.backbone),
        "head_state_dict": cpu_state_dict(model.head),
        "config": {"n_highest_peaks": args.n_highest_peaks},
    }
    torch.save(checkpoint, run_dir / "best_m4.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved checkpoint: {run_dir / 'best_m4.pt'}", flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
