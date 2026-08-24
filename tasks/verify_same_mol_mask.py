"""本地 CPU 端到端验证「显式同分子正例」工程正确性（真实分布采样）。

背景（G7 前审计，2026-08-21）：
  旧实现把同分子正例绑定到 batch 内 shuffle 偶遇，但真实 train 池 median 仅 3 谱/分子，
  batch=32 下同分子对期望 0.318/批 —— 偶遇方案几乎不触发（demo 8×16 密集构造高 750×，
  不可外推）。故 step4 已改为「显式同分子正例」：__getitem__ 直接携带同完整 SMILES 的另一条谱
  作为正例，独立于 batch 采样。

  同时判据从 ik14（InChIKey 前 14 位=连通性，会把立体异构体误并，实测 8.86% 混并）
  改为完整 SMILES（含 @/@@ 立体标记），promote 不再把对映体/非对映体错误拉近。

判据（A/B + 正例对冲，lr 无关）：
  mask 关（复现 G5 bug）：同分子被当负例推远 → 余弦下降。
  mask 开（修复）       ：同分子被排除出负例 → 余弦基本不降。
  promote（显式正例）   ：同分子除排除外还作正例 → 余弦不降甚至上升（拉到同位置）。
                          三种情况下异构体都应被推远。

审计 D（顺带）：baseline 的同分子余弦即「真同分子不同谱在官方 DreaMS 下的平均余弦」——
  若明显 <1 说明「多谱」是真变体而非逐位复制，显式正例有真实信号可学。

注意：本脚本用 lr=1e-3（机制验证用，放大梯度方向便于观测），
      **非**真实训练 lr（3e-5）。这里只验证工程方向对，不看绝对性能。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum, seed_everything  # noqa: E402
from noise_augment import NoiseConfig  # noqa: E402
from step4_infonce_train import (  # noqa: E402
    NoiseContrastiveDataset, collate, freeze_all, infonce_forward, unfreeze_last_layers,
)

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=512, help="训练子集 anchor 数（真实分布随机抽）")
    ap.add_argument("--steps", type=int, default=10, help="每配置训练步数")
    ap.add_argument("--lr", type=float, default=1e-3, help="机制验证用放大 lr（非真实 3e-5）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--unfreeze-layers", type=int, default=2)
    ap.add_argument("--n-highest", type=int, default=100)
    ap.add_argument("--max-pairs", type=int, default=120, help="同分子/异构体对度量上限")
    return ap.parse_args()


def embed(model, specs, device, batch=32):
    out = []
    for i in range(0, len(specs), batch):
        b = torch.stack(specs[i:i + batch]).to(device)
        with torch.no_grad():
            out.append(model(b).cpu())
    return torch.cat(out, dim=0)


def pair_cos(model, pairs, clean_spec, device, max_pairs):
    if not pairs:
        return float("nan")
    pairs = pairs[:max_pairs]
    ea = embed(model, [clean_spec(a) for a, _ in pairs], device)
    eb = embed(model, [clean_spec(b) for _, b in pairs], device)
    return float((ea * eb).sum(1).mean())


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    seed_everything(0)

    manifest = json.load(open(DEFAULT_MANIFEST))
    entries = manifest["train"]

    h5 = h5py.File(DEFAULT_DATA, "r")
    pmz_all = np.array(h5["precursor_mz"][:], dtype=float)
    smiles_all = np.array([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                           for x in h5["smiles"][:]])

    def clean_spec(row):
        return preprocess_spectrum(np.asarray(h5["spectrum"][row]), float(pmz_all[row]), args.n_highest)

    # ---- 1) 训练子集：真实分布随机抽（不是 8×16 密集），验证显式正例在真实密度下生效 ----
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(entries), size=args.n_train, replace=False)
    train_subset = [entries[i] for i in idxs]
    print(f"[data] 训练子集 {len(train_subset)} 锚（真实分布随机抽，非密集构造）", flush=True)

    # ---- 2) 度量对：密集抽（独立于训练子集），用完整 SMILES 判同分子（避免立体异构体混入）----
    # 同分子对：同完整 SMILES 的不同谱（真同分子，不含立体异构体）。
    by_smiles = defaultdict(list)
    for e in entries:
        by_smiles[smiles_all[e["anchor_row"]]].append(e["anchor_row"])
    same_pairs = []
    for rows in by_smiles.values():
        if len(rows) < 2:
            continue
        # 每分子取至多 3 对，跨足够多分子保证度量稳定
        for i in range(min(len(rows) - 1, 3)):
            same_pairs.append((rows[i], rows[i + 1]))
        if len(same_pairs) >= args.max_pairs * 3:
            break
    iso_pairs = []
    for e in entries:
        for n in e["neg"]:
            iso_pairs.append((e["anchor_row"], n["row"]))
            if len(iso_pairs) >= args.max_pairs * 3:
                break
        if len(iso_pairs) >= args.max_pairs * 3:
            break
    print(f"[pairs] same_mol={len(same_pairs)}  isomer={len(iso_pairs)}（SMILES 判同分子）", flush=True)

    # ---- 3) 载模型一次，快照权重用于重置 ----
    t0 = time.time()
    model, kind = load_base_model(DEFAULT_BASE, DEFAULT_ARCH, device, args.n_highest)
    freeze_all(model)
    unfreeze_last_layers(model, args.unfreeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"[model] loaded in {time.time()-t0:.0f}s; trainable={sum(p.numel() for p in trainable):,}", flush=True)

    def reset():
        model.load_state_dict(snapshot)

    def measure():
        return pair_cos(model, same_pairs, clean_spec, device, args.max_pairs), \
               pair_cos(model, iso_pairs, clean_spec, device, args.max_pairs)

    base_sm, base_iso = measure()
    print(f"[baseline] same_mol_cos={base_sm:.4f}  isomer_cos={base_iso:.4f}"
          f"  （审计 D：真同分子官方余弦 {base_sm:.4f}，" + ("<1=真变体" if base_sm < 0.95 else "≈1=近似重复") + "）", flush=True)

    # ---- 4) 训练（mask 关 / mask 开 / promote 显式正例）----
    noise_cfg = NoiseConfig()
    configs = [
        ("mask_off", False, False),   # 复现 bug
        ("mask_on", True, False),     # 只排除负例
        ("promote", True, True),      # 排除负例 + 显式同分子正例（主动拉近）
    ]
    results = {}
    for tag, use_mask, use_promote in configs:
        reset()
        opt = torch.optim.AdamW(trainable, lr=args.lr)
        ds = NoiseContrastiveDataset(train_subset, DEFAULT_DATA, pmz_all, args.n_highest, noise_cfg, 0,
                                     smiles_all=smiles_all)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        model.train()
        step = 0
        for epoch in range(3):
            ds.set_epoch(epoch)
            for anchors, pos, nc, nn, neg_ptr, same_mol, sm_pos, sm_mask in loader:
                if step >= args.steps:
                    break
                if not use_mask:
                    same_mol = torch.eye(anchors.shape[0], dtype=torch.bool)
                sp = sm_pos if use_promote else None
                sm = sm_mask if use_promote else None
                loss, _, _ = infonce_forward(model, anchors, pos, nc, nn, neg_ptr, 0.1, same_mol, sp, sm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
            if step >= args.steps:
                break
        model.eval()
        sm_cos, iso = measure()
        results[tag] = (sm_cos, iso)
        print(f"[{tag}] {step} 步后  same_mol_cos={sm_cos:.4f} (Δ={sm_cos-base_sm:+.4f})  "
              f"isomer_cos={iso:.4f} (Δ={iso-base_iso:+.4f})", flush=True)

    # ---- 5) 判定 ----
    moff_sm, moff_iso = results["mask_off"]
    mon_sm, mon_iso = results["mask_on"]
    prom_sm, prom_iso = results["promote"]
    drop_off = base_sm - moff_sm
    drop_on = base_sm - mon_sm
    drop_prom = base_sm - prom_sm
    print("\n===== 判定 =====")
    print(f"同分子余弦变化：mask_off Δ={-drop_off:+.4f}   mask_on Δ={-drop_on:+.4f}   promote Δ={-drop_prom:+.4f}")
    print(f"异构体余弦变化：mask_off Δ={moff_iso-base_iso:+.4f}   mask_on Δ={mon_iso-base_iso:+.4f}   promote Δ={prom_iso-base_iso:+.4f}")

    ok = True
    # 1) mask 开比 mask 关少伤同分子（A/B）
    if not (drop_off > drop_on):
        print("[FAIL] mask_off 未比 mask_on 更伤同分子")
        ok = False
    # 2) promote 比 mask_on 更保同分子（显式正例对冲在真实分布下生效）
    if not (drop_prom < drop_on):
        print(f"[FAIL] promote 未比 mask_on 更保同分子（{drop_prom:.4f} >= {drop_on:.4f}），"
              f"显式正例在真实分布下未生效")
        ok = False
    # 3) promote 下异构体仍被推远（正例没有把异构体也拉近）
    if not (prom_iso < base_iso - 0.01):
        print(f"[WARN] promote 异构体余弦未降 >0.01（分离可能被正例削弱，需关注）")
    print(f"\n=== verify_same_mol_mask: {'PASS' if ok else 'FAIL'} ===")
    h5.close()


if __name__ == "__main__":
    main()
