"""
Step 4: InfoNCE 对比训练（噪声不变性 + 异构体区分），解冻 backbone last-N 层。

依据 docs/NOISE_TASK_PLAN_20260818.md：
  anchor = 干净谱 A（CLS 经 head 投影 + L2 归一化）
  正例   = 噪声谱 noise(A)（唯一正例，四轴噪声）
  负例   = 异构体谱 {B, noise(B), C, noise(C), ...}（clean + 噪声版都放）
         ∪ batch 内其他 anchor 的干净谱 {A_j} 与噪声谱 {noise(A_j)}（j≠i）
  τ = 0.1；噪声每 step 现场重抽。

起点：官方微调模型 official_embedding_slim.pt（head(CLS) 约定，与 M3/M4/噪声 pilot 可比）。
解冻：last-N 层 transformer（atts+ffs+对应 scales）+ head，其余冻结（--unfreeze-layers，默认 2）。

G2 smoke 三件事（严格因果门）：
  1. 解冻生效：首步反传后 last-N 层 grad 范数 > 0（若冻结则 grad=None → 0）。
  2. loss 下降：首步 loss > 末步 loss。
  3. 无 NaN。

用法：
  # 本机 CPU smoke（tiny 子集，验证机制）
  python tasks/step4_infonce_train.py --smoke --device cpu
  # 服务器 GPU 全量（G3，后续接 sbatch）
  python tasks/step4_infonce_train.py --device cuda --epochs 8 --batch-size 32

输入：
  tasks/massspecgym_isomers/dataset_manifest.json   # Step 3 产物
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt                # 起点（官方微调）
  dreams/models/pretrained/ssl_model_server.pt      # 架构（重建 DreaMS args）
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import (  # noqa: E402
    cpu_state_dict,
    load_base_model,
    preprocess_spectrum,
    seed_everything,
)
from noise_augment import NoiseConfig, apply_noise  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/noise_isomer_infonce"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5, help="PartB/C 定案 3e-5；1e-4/3e-4 会坍缩+检索退化")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--unfreeze-layers", type=int, default=2, help="解冻 last-N 层")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--smoke", action="store_true", help="tiny 子集 + 1 epoch，仅验证机制")
    ap.add_argument("--max-anchors", type=int, default=0, help="训练子集大小（0=全量）")
    ap.add_argument("--max-steps", type=int, default=0, help="每 epoch 最多步数（0=全 epoch）")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
class NoiseContrastiveDataset(Dataset):
    """每个 item = 一个 anchor（谱行）；__getitem__ 现场施加四轴噪声。"""

    def __init__(self, entries, h5_path, precursor_mz_all, n_highest, noise_cfg, seed):
        self.entries = entries
        self.h5_path = str(h5_path)
        self.pmz = precursor_mz_all
        self.n_highest = n_highest
        self.noise_cfg = noise_cfg
        self.seed = seed
        self.epoch = 0
        self._h5 = None

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def _handle(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx)
        h = self._handle()

        def make(row: int, noisy: bool) -> torch.Tensor:
            raw = np.asarray(h["spectrum"][row])
            if noisy:
                raw = apply_noise(raw, rng, self.noise_cfg)
            return preprocess_spectrum(raw, float(self.pmz[row]), self.n_highest)

        anchor_clean = make(e["anchor_row"], False)
        anchor_noisy = make(e["anchor_row"], True)
        neg_clean = [make(n["row"], False) for n in e["neg"]]
        neg_noisy = [make(n["row"], True) for n in e["neg"]]
        return anchor_clean, anchor_noisy, neg_clean, neg_noisy

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def collate(batch):
    anchors = torch.stack([b[0] for b in batch])
    pos = torch.stack([b[1] for b in batch])
    nc_flat, nn_flat = [], []
    neg_ptr = [0]
    for b in batch:
        nc_flat.extend(b[2])
        nn_flat.extend(b[3])
        neg_ptr.append(len(nc_flat))
    neg_ptr = torch.tensor(neg_ptr, dtype=torch.long)
    zero = torch.zeros(0, anchors.shape[1], anchors.shape[2])
    nc = torch.stack(nc_flat) if nc_flat else zero
    nn = torch.stack(nn_flat) if nn_flat else zero
    return anchors, pos, nc, nn, neg_ptr


# --------------------------------------------------------------------------- #
# 解冻控制
# --------------------------------------------------------------------------- #
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_layers(model, n: int):
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    n = max(1, min(n, L))
    # head 永远训练
    for p in model.head.parameters():
        p.requires_grad = True
    # last-N 层 att + ff
    for i in range(L - n, L):
        for p in enc.atts[i].parameters():
            p.requires_grad = True
        for p in enc.ffs[i].parameters():
            p.requires_grad = True
    # 对应 scales（layer i 用 scales[2i], scales[2i+1]）+ 末层 norm scales[-1]
    for i in range(L - n, L):
        for p in enc.scales[2 * i].parameters():
            p.requires_grad = True
        for p in enc.scales[2 * i + 1].parameters():
            p.requires_grad = True
    for p in enc.scales[-1].parameters():
        p.requires_grad = True


def last_layer_grad_norm(model, n: int) -> float:
    """last-N 层梯度范数；若解冻失败（冻结）则 grad=None → 0。"""
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    total = 0.0
    for i in range(L - n, L):
        for p in list(enc.atts[i].parameters()) + list(enc.ffs[i].parameters()):
            if p.grad is not None:
                total += float(p.grad.float().norm())
    return total


def last_layer_weight_norm(model, n: int) -> float:
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    total = 0.0
    for i in range(L - n, L):
        for p in list(enc.atts[i].parameters()) + list(enc.ffs[i].parameters()):
            total += float(p.detach().float().norm())
    return total


# --------------------------------------------------------------------------- #
# InfoNCE loss
# --------------------------------------------------------------------------- #
def infonce_forward(model, anchors, pos, nc, nn, neg_ptr, tau):
    """返回 loss 与 (pos_cos, neg_cos_mean)。全部嵌入先归一化（model.forward 已归一化）。"""
    a = model(anchors)        # (B, d)
    p = model(pos)            # (B, d)
    nc_emb = model(nc) if nc.shape[0] else torch.zeros(0, a.shape[1], device=a.device)
    nn_emb = model(nn) if nn.shape[0] else torch.zeros(0, a.shape[1], device=a.device)

    pos_logit = (a * p).sum(1) / tau                       # (B,)
    sim_aa = a @ a.T / tau                                 # (B, B)
    sim_ap = a @ p.T / tau                                 # (B, B)
    sim_anc = a @ nc_emb.T / tau                           # (B, Nc)
    sim_ann = a @ nn_emb.T / tau                           # (B, Nn)

    B = a.shape[0]
    arange = torch.arange(B, device=a.device)
    losses = []
    for i in range(B):
        mask = arange != i
        negs = [sim_aa[i, mask], sim_ap[i, mask]]          # in-batch clean/noisy others
        lo, hi = int(neg_ptr[i]), int(neg_ptr[i + 1])
        if hi > lo:
            negs.append(sim_anc[i, lo:hi])                 # 干净异构体
            negs.append(sim_ann[i, lo:hi])                 # 噪声异构体
        all_logits = torch.cat([pos_logit[i:i + 1], *negs])
        losses.append(-all_logits[0] + torch.logsumexp(all_logits, dim=0))
    loss = torch.stack(losses).mean()
    return loss, a, p


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    train_entries = manifest["train"]
    if args.smoke:
        train_entries = train_entries[:200]
        args.epochs = 1
    elif args.max_anchors > 0:
        train_entries = train_entries[: args.max_anchors]
    print(f"[data] train anchors: {len(train_entries)}", flush=True)

    # 载入 precursor_mz（全量，一次读进内存）
    import h5py
    with h5py.File(args.data, "r") as f:
        precursor_mz_all = np.array(f["precursor_mz"][:], dtype=float)

    noise_cfg = NoiseConfig()
    dataset = NoiseContrastiveDataset(
        train_entries, args.data, precursor_mz_all, args.n_highest_peaks, noise_cfg, args.seed
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    model, kind = load_base_model(
        args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks
    )
    freeze_all(model)
    unfreeze_last_layers(model, args.unfreeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] init kind={kind}; trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    if not trainable:
        raise RuntimeError("没有可训练参数（解冻失败）")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    w0 = last_layer_weight_norm(model, args.unfreeze_layers)
    losses = []
    grad_flow = 0.0
    nan_detected = False
    t0 = time.time()

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        for step, (anchors, pos, nc, nn, neg_ptr) in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            anchors = anchors.to(device)
            pos = pos.to(device)
            nc = nc.to(device)
            nn = nn.to(device)
            neg_ptr = neg_ptr.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss, _, _ = infonce_forward(model, anchors, pos, nc, nn, neg_ptr, args.tau)
            scaler.scale(loss).backward()

            if epoch == 0 and step == 0:
                grad_flow = last_layer_grad_norm(model, args.unfreeze_layers)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            li = float(loss.detach())
            if not np.isfinite(li):
                nan_detected = True
            losses.append(li)
            if step % 10 == 0 or step == 0:
                print(f"  epoch {epoch} step {step:4d} loss={li:.4f} "
                      f"grad_flow={grad_flow:.4f}", flush=True)

    w1 = last_layer_weight_norm(model, args.unfreeze_layers)
    loss_first = losses[0] if losses else float("nan")
    loss_last = losses[-1] if losses else float("nan")

    checks = {
        "seed": args.seed,
        "kind": kind,
        "n_train_anchors": len(train_entries),
        "unfreeze_layers": args.unfreeze_layers,
        "tau": args.tau, "lr": args.lr, "batch_size": args.batch_size,
        "grad_flow_last_layers": grad_flow,
        "gate1_unfreeze_ok": bool(grad_flow > 0),
        "loss_first": loss_first, "loss_last": loss_last,
        "gate2_loss_decreased": bool(np.isfinite(loss_first) and np.isfinite(loss_last) and loss_last < loss_first),
        "nan_detected": nan_detected,
        "gate3_no_nan": bool(not nan_detected),
        "weight_change_last_layer": float(abs(w1 - w0)),
        "gate_weight_moved": bool(abs(w1 - w0) > 1e-6),
        "elapsed_seconds": time.time() - t0,
    }
    gate = checks["gate1_unfreeze_ok"] and checks["gate2_loss_decreased"] and checks["gate3_no_nan"]
    checks["G2_smoke_pass"] = bool(gate)

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存 checkpoint（供 G3 eval 加载；smoke 也存，便于检查）
    checkpoint = {
        "format": "noise_isomer_infonce_v1",
        "seed": args.seed,
        "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
        "base_checkpoint": str(args.base_ckpt.resolve()),
        "backbone_state_dict": cpu_state_dict(model.backbone),
        "head_state_dict": cpu_state_dict(model.head),
        "config": {"n_highest_peaks": args.n_highest_peaks, "unfreeze_layers": args.unfreeze_layers,
                   "tau": args.tau, "lr": args.lr, "batch_size": args.batch_size},
        "checks": checks,
    }
    torch.save(checkpoint, run_dir / "best_infonce.pt")

    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(f"\n=== G2 smoke: {'PASS' if gate else 'FAIL'} ===", flush=True)
    print(f"  gate1 解冻生效(grad>0): {checks['gate1_unfreeze_ok']}  (grad_flow={grad_flow:.4f})")
    print(f"  gate2 loss 下降: {checks['gate2_loss_decreased']}  ({loss_first:.4f} -> {loss_last:.4f})")
    print(f"  gate3 无 NaN: {checks['gate3_no_nan']}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
