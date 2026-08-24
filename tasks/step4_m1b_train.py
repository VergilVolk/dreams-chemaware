"""Step 4-M1b: balanced local-ranking fine-tune (head-only), NO synthetic noise.

Per docs/G8R_M1_POSTMORTEM_AND_M1B_DECISION_20260822.md §6 (locked 2026-08-22).
M1b answers ONE question: can real cross-condition positives + real hard
negatives improve LOCAL ranking, under a head-only linear map?

Two-stream batch (easy samples must not dominate by count):
  hard-ranking stream  = anchors with BOTH a real cross-condition positive AND
                         >=1 hard negative (34.34% of the locked train set).
                         L_rank + lam_f * L_pos-floor.
  safety-preservation  = anchors cycled from the FULL train set.
                         lam_p * L_preserve.
  Each step takes one hard batch + one safety batch; the two are mean'd
  separately then combined.  Epoch is defined over the hard anchors
  (~108 steps/epoch at batch 32).

Locked objective:
  L_rank       = softplus((margin + s_n* - s_p) / tau)      margin=0.05, tau=0.1
                 n* = hardest fixed negative (max cosine).
  L_pos-floor  = relu(s_p^official - s_p^student)           positive only gets a
                 teacher lower bound (never pushed unconditionally toward 1).
  L_preserve   = 1 - cos(z, z_official)                     lam_p = 5.

NO synthetic noise: no peak deletion / intensity / m/z / addition, no synthetic
positive, no shared-main-peak counterfactual.  Those are a SEPARATE ablation
gated behind M1b passing.

Because M1b is noise-free and the backbone is frozen, all OFFICIAL embeddings are
constants: they are computed once up front (z_off cache), and the training loop
only runs the student forward.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import cpu_state_dict, load_base_model, preprocess_spectrum, seed_everything  # noqa: E402
from step5_gate_eval import embed  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/m1b_rank"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-subset", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.05, help="ranking margin m")
    p.add_argument("--tau", type=float, default=0.1, help="softplus temperature")
    p.add_argument("--lam-posfloor", type=float, default=1.0, help="lambda_f: teacher pos lower-bound")
    p.add_argument("--lam-preserve", type=float, default=5.0, help="lambda_p: official-representation preservation")
    p.add_argument("--unfreeze-layers", type=int, default=0, help="0=head-only (M1b default)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch (smoke)")
    p.add_argument("--smoke", action="store_true", help="tiny subset + 1 epoch, mechanism check")
    p.add_argument("--max-hard", type=int, default=120, help="smoke: cap hard anchors")
    p.add_argument("--max-safety", type=int, default=120, help="smoke: cap safety anchors")
    p.add_argument("--backbone-eval", action=argparse.BooleanOptionalAction, default=False,
                   help="P0 dropout single-variable control: set the FROZEN backbone to eval() "
                        "(dropout off) and keep only the head in train(). Default False keeps the "
                        "historical model.train() (frozen-backbone dropout ON).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  frozen-control (mirrors step4)
# --------------------------------------------------------------------------- #
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_layers(model, n: int):
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    n = max(0, min(n, L))
    for p in model.head.parameters():
        p.requires_grad = True
    for i in range(L - n, L):
        for p in enc.atts[i].parameters():
            p.requires_grad = True
        for p in enc.ffs[i].parameters():
            p.requires_grad = True
    for i in range(L - n, L):
        for p in enc.scales[2 * i].parameters():
            p.requires_grad = True
        for p in enc.scales[2 * i + 1].parameters():
            p.requires_grad = True
    for p in enc.scales[-1].parameters():
        p.requires_grad = True


def head_grad_norm(model) -> float:
    total = 0.0
    for p in model.head.parameters():
        if p.grad is not None:
            total += float(p.grad.float().norm())
    return total


def cpu_optimizer_state(optimizer):
    """Full optimizer state_dict with every tensor deep-copied to CPU.

    optimizer.state_dict()["state"] holds LIVE references; a shallow .cpu()
    would corrupt the live optimizer's exp_avg on GPU.  Rebuild both the state
    and param_groups so resume via optimizer.load_state_dict() actually works.
    """
    sd = optimizer.state_dict()
    return {
        "state": {
            pid: {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                  for k, v in ps.items()}
            for pid, ps in sd["state"].items()
        },
        "param_groups": sd["param_groups"],
    }


# --------------------------------------------------------------------------- #
#  sibling map (one cross-condition pair per (ik14, adduct) group)
# --------------------------------------------------------------------------- #
def build_sibling(entries: list[dict]) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[(e["ik14"], e["adduct"])].append(i)
    sib = [-1] * len(entries)
    for rows in groups.values():
        if len(rows) == 2:
            sib[rows[0]] = rows[1]
            sib[rows[1]] = rows[0]
        else:
            for a, b in zip(rows, rows[1:]):
                sib[a] = b
                sib[b] = a
    return sib


# --------------------------------------------------------------------------- #
#  datasets (index into precomputed spectra + official-embedding cache)
# --------------------------------------------------------------------------- #
class HardRankingDataset(Dataset):
    """One item per hard anchor: (anchor_spec, pos_spec, [neg_specs], off_pos_cos)."""

    def __init__(self, spec_idx, pos_spec_idx, neg_spec_idx, off_pos_cos):
        self.spec_idx = spec_idx              # list[int] local spec index of anchor
        self.pos_spec_idx = pos_spec_idx      # list[int]
        self.neg_spec_idx = neg_spec_idx      # list[list[int]]
        self.off_pos_cos = off_pos_cos        # torch.Tensor (n,)

    def __len__(self):
        return len(self.spec_idx)

    def __getitem__(self, k):
        return (self.spec_idx[k], self.pos_spec_idx[k],
                self.neg_spec_idx[k], self.off_pos_cos[k])


class SafetyDataset(Dataset):
    def __init__(self, spec_idx):
        self.spec_idx = spec_idx

    def __len__(self):
        return len(self.spec_idx)

    def __getitem__(self, k):
        return self.spec_idx[k]


def collate_hard(batch, specs):
    anchors = torch.stack([specs[b[0]] for b in batch])
    pos = torch.stack([specs[b[1]] for b in batch])
    neg_flat, neg_ptr = [], [0]
    off = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    anchor_idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos_idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    neg_idx_flat = []
    for b in batch:
        neg_flat.extend(specs[j] for j in b[2])
        neg_ptr.append(len(neg_flat))
        neg_idx_flat.extend(b[2])
    zero = torch.zeros(0, anchors.shape[1], anchors.shape[2])
    neg = torch.stack(neg_flat) if neg_flat else zero
    neg_idx = (torch.tensor(neg_idx_flat, dtype=torch.long)
               if neg_idx_flat else torch.tensor([], dtype=torch.long))
    return (anchors, pos, neg, torch.tensor(neg_ptr, dtype=torch.long), off,
            anchor_idx, pos_idx, neg_idx)


def collate_safety(batch, specs):
    return torch.stack([specs[j] for j in batch]), torch.tensor(batch, dtype=torch.long)


# --------------------------------------------------------------------------- #
#  loss
# --------------------------------------------------------------------------- #
def m1b_loss(model, z_off, hard_batch, safety_specs, safety_idx,
             margin, tau, lam_f, lam_p):
    anchors, pos, neg, neg_ptr, off_pos_cos, anchor_idx, pos_idx, neg_idx = hard_batch
    dev = anchors.device
    B = anchors.shape[0]
    z_a = model(anchors)
    z_p = model(pos)
    z_n = model(neg) if neg.shape[0] else torch.zeros(0, z_a.shape[1], device=dev)

    s_p = (z_a * z_p).sum(1)
    s_n = torch.empty(B, device=dev)
    for i in range(B):
        lo, hi = int(neg_ptr[i]), int(neg_ptr[i + 1])
        s_n[i] = (z_a[i] * z_n[lo:hi]).sum(1).max()

    L_rank = F.softplus((margin + s_n - s_p) / tau).mean()
    L_posfloor = F.relu(off_pos_cos.to(dev) - s_p).mean()

    # Issue A fix (§9): the hard-ranking stream must ALSO be preserved against
    # the official representation — anchor, positive, and every hard negative —
    # so the ranking update cannot freely rotate the hard subspace.  L_preserve
    # then covers hard + safety streams equally, exactly as pre-registered.
    off_a = z_off[anchor_idx.to(dev)]
    off_p = z_off[pos_idx.to(dev)]
    off_n = z_off[neg_idx.to(dev)]
    preserve_a = 1.0 - (z_a * off_a).sum(1)
    preserve_p = 1.0 - (z_p * off_p).sum(1)
    preserve_n = torch.empty(B, device=dev)
    for i in range(B):
        lo, hi = int(neg_ptr[i]), int(neg_ptr[i + 1])
        preserve_n[i] = (1.0 - (z_n[lo:hi] * off_n[lo:hi]).sum(1)).mean()
    L_preserve_hard = ((preserve_a + preserve_p + preserve_n) / 3.0).mean()

    z_s = model(safety_specs)
    z_off_s = z_off[safety_idx.to(dev)]
    L_preserve_safety = (1.0 - (z_s * z_off_s).sum(1)).mean()

    L_preserve = (L_preserve_hard + L_preserve_safety) / 2.0

    loss = L_rank + lam_f * L_posfloor + lam_p * L_preserve
    return loss, L_rank, L_posfloor, L_preserve, s_p.mean(), s_n.mean()


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    payload = json.loads(args.train_subset.read_text(encoding="utf-8"))
    entries = payload["entries"]
    n_all = len(entries)
    sibling = build_sibling(entries)
    hard_idx = [i for i, e in enumerate(entries) if e["neg"]]

    # ---- select active sets ----
    if args.smoke:
        hard_sel = hard_idx[: args.max_hard]
        safety_sel = list(range(min(args.max_safety, n_all)))
    else:
        hard_sel = hard_idx
        safety_sel = list(range(n_all))

    active_entries = set(hard_sel) | set(safety_sel)
    for i in hard_sel:
        if sibling[i] >= 0:
            active_entries.add(sibling[i])
    active_entries = sorted(active_entries)

    # ---- rows to embed ----
    with h5py.File(args.data, "r") as h:
        pmz_all = np.asarray(h["precursor_mz"][:], dtype=float)
        anchor_rows_active = [int(entries[i]["anchor_row"]) for i in active_entries]
        anchor_row_set = set(anchor_rows_active)
        neg_rows_active = sorted(
            {int(n["row"]) for i in hard_sel for n in entries[i]["neg"]} - anchor_row_set)
        rows = anchor_rows_active + neg_rows_active
        specs = torch.stack([
            preprocess_spectrum(np.asarray(h["spectrum"][r]), float(pmz_all[r]), args.n_highest_peaks)
            for r in rows
        ])
    row_to_spec = {r: k for k, r in enumerate(rows)}
    spec_of_entry = {i: row_to_spec[int(entries[i]["anchor_row"])] for i in active_entries}

    print(f"[data] full={n_all} hard={len(hard_idx)} active_entries={len(active_entries)} "
          f"hard_sel={len(hard_sel)} safety_sel={len(safety_sel)} rows={len(rows)}", flush=True)

    # ---- official-embedding cache (frozen constants) ----
    official, _ = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    official.eval()
    z_off = embed(official, [specs[k] for k in range(len(specs))], device, args.batch_size).to(device)
    del official
    gc.collect()

    # ---- precomputed per-hard-anchor indices + off_pos_cos ----
    hard_spec_idx = [spec_of_entry[i] for i in hard_sel]
    hard_pos_spec_idx = [spec_of_entry[sibling[i]] for i in hard_sel]
    hard_neg_spec_idx = [[row_to_spec[int(n["row"])] for n in entries[i]["neg"]] for i in hard_sel]
    off_pos_cos = [float((z_off[spec_of_entry[i]] * z_off[spec_of_entry[sibling[i]]]).sum())
                   for i in hard_sel]
    safety_spec_idx = [spec_of_entry[i] for i in safety_sel]

    hard_ds = HardRankingDataset(hard_spec_idx, hard_pos_spec_idx, hard_neg_spec_idx, off_pos_cos)
    safety_ds = SafetyDataset(safety_spec_idx)
    hard_loader = DataLoader(hard_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=lambda b: collate_hard(b, specs))
    safety_loader = DataLoader(safety_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: collate_safety(b, specs))

    model, kind = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    freeze_all(model)
    unfreeze_last_layers(model, args.unfreeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] init kind={kind}; trainable params={sum(p.numel() for p in trainable):,}", flush=True)
    if not trainable:
        raise RuntimeError("no trainable params (unfreeze failed)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "last.pt"

    if args.backbone_eval:
        # P0 control: frozen backbone in eval() (dropout off) so the head learns
        # a stable metric on a fixed coordinate system; only the head is train().
        model.backbone.eval()
        model.head.train()
    else:
        # historical default: whole model in train() => frozen-backbone dropout ON.
        model.train()
    losses, margins = [], []
    grad_flow = 0.0
    nan_detected = False
    start_epoch = 0

    if args.resume and resume_path.exists():
        st = torch.load(resume_path, map_location="cpu")
        model.backbone.load_state_dict(st["backbone_state_dict"])
        model.head.load_state_dict(st["head_state_dict"])
        try:
            optimizer.load_state_dict(st["optimizer_state_dict"])
        except Exception as e:
            print(f"[resume] optimizer state load failed (fresh optimizer): {e}", flush=True)
        start_epoch = int(st.get("epoch_completed", -1)) + 1
        losses = list(st.get("losses", []))
        margins = list(st.get("margins", []))
        grad_flow = float(st.get("grad_flow", 0.0))
        nan_detected = bool(st.get("nan_detected", False))
        print(f"[resume] from {resume_path.name} at epoch {start_epoch}", flush=True)

    t0 = time.time()
    from itertools import cycle

    safety_cycle = cycle(safety_loader)
    for epoch in range(start_epoch, args.epochs):
        steps = 0
        for step, (hard_batch, (safety_specs, safety_idx)) in enumerate(
                zip(hard_loader, safety_cycle)):
            if args.max_steps and step >= args.max_steps:
                break
            anchors, pos, neg, neg_ptr, off, anchor_idx, pos_idx, neg_idx = hard_batch
            hard_batch = (anchors.to(device), pos.to(device), neg.to(device),
                          neg_ptr.to(device), off,
                          anchor_idx.to(device), pos_idx.to(device), neg_idx.to(device))
            safety_specs = safety_specs.to(device)
            safety_idx = safety_idx.to(device)

            loss, Lr, Lf, Lp, sp_mean, sn_mean = m1b_loss(
                model, z_off, hard_batch, safety_specs, safety_idx,
                args.margin, args.tau, args.lam_posfloor, args.lam_preserve)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 0 and step == 0:
                grad_flow = head_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            li = float(loss.detach())
            if not np.isfinite(li):
                nan_detected = True
            losses.append(li)
            margins.append(float(sp_mean.detach() - sn_mean.detach()))
            steps += 1
            if step % 10 == 0 or step == 0:
                print(f"  epoch {epoch} step {step:4d} loss={li:.4f} "
                      f"L_rank={float(Lr.detach()):.4f} L_posfloor={float(Lf.detach()):.4f} "
                      f"L_preserve={float(Lp.detach()):.4f} margin={margins[-1]:.4f} "
                      f"grad={grad_flow:.4f}", flush=True)

        ckpt = {
            "format": "m1b_rank_v2",
            "seed": args.seed,
            "epoch_completed": epoch,
            "losses": losses,
            "margins": margins,
            "grad_flow": grad_flow,
            "nan_detected": nan_detected,
            "backbone_state_dict": cpu_state_dict(model.backbone),
            "head_state_dict": cpu_state_dict(model.head),
            "optimizer_state_dict": cpu_optimizer_state(optimizer),
            "config": {"margin": args.margin, "tau": args.tau,
                       "lam_posfloor": args.lam_posfloor, "lam_preserve": args.lam_preserve,
                       "hard_preserve": True,
                       "backbone_eval": args.backbone_eval,
                       "unfreeze_layers": args.unfreeze_layers, "lr": args.lr,
                       "batch_size": args.batch_size, "epochs": args.epochs,
                       "n_highest_peaks": args.n_highest_peaks,
                       "train_subset": str(args.train_subset)},
        }
        tmp = resume_path.with_name(resume_path.name + ".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, resume_path)
        print(f"[save] epoch {epoch} -> {resume_path.name} ({steps} steps)", flush=True)

    loss_first = losses[0] if losses else float("nan")
    loss_last = losses[-1] if losses else float("nan")
    checks = {
        "seed": args.seed,
        "kind": kind,
        "n_full": n_all,
        "n_hard": len(hard_sel),
        "n_safety": len(safety_sel),
        "unfreeze_layers": args.unfreeze_layers,
        "margin": args.margin, "tau": args.tau,
        "lam_posfloor": args.lam_posfloor, "lam_preserve": args.lam_preserve,
        "grad_flow_head": grad_flow,
        "gate1_unfreeze_ok": bool(grad_flow > 0),
        "loss_first": loss_first, "loss_last": loss_last,
        "gate2_loss_decreased": bool(np.isfinite(loss_first) and np.isfinite(loss_last) and loss_last < loss_first),
        "nan_detected": nan_detected,
        "gate3_no_nan": bool(not nan_detected),
        "margin_first": margins[0] if margins else float("nan"),
        "margin_last": margins[-1] if margins else float("nan"),
        "elapsed_seconds": time.time() - t0,
    }
    gate = checks["gate1_unfreeze_ok"] and checks["gate2_loss_decreased"] and checks["gate3_no_nan"]
    checks["smoke_pass"] = bool(gate)
    (run_dir / "summary.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    # NOTE: this is the FINAL-epoch checkpoint, NOT a validation-gate-selected
    # best.  Named "final.pt" to avoid the "best.pt" misnomer (master plan §9).
    final_ckpt = {"format": "m1b_rank_v2", "seed": args.seed,
                  "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
                  "base_checkpoint": str(args.base_ckpt.resolve()),
                  "backbone_state_dict": cpu_state_dict(model.backbone),
                  "head_state_dict": cpu_state_dict(model.head),
                  "config": checks}
    tmp = (run_dir / "final.pt").with_name("final.pt.tmp")
    torch.save(final_ckpt, tmp)
    os.replace(tmp, run_dir / "final.pt")

    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(f"\n=== M1b smoke: {'PASS' if gate else 'FAIL'} ===", flush=True)


if __name__ == "__main__":
    main()
