"""
规则驱动的对比学习微调脚本 [v4]

核心思路：
  DreaMS 用昂贵的结构标签（InChIKey）做 Triplet Loss 对比微调。
  我们用免费的化学规则重叠度替代结构标签——两张谱图共享的碎裂规则越多，
  化学上越相似，应该被拉近。

与 v1-v3 的本质区别：
  v1-v3: 规则→注意力偏置→间接影响嵌入（梯度链太长，断在 softmax）
  v4:    规则→谱图相似度→Triplet Loss→直接影响嵌入距离（梯度链短）

用法：
  python -m dreams.models.chem_aware.train_contrastive \
      --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
      --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
      --epochs 10 --batch_size 32

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple

import dreams.utils.data as du
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.definitions import PRETRAINED


def parse_args():
    p = argparse.ArgumentParser(description='Rule-driven contrastive fine-tuning [v4]')
    p.add_argument('--dataset_path', type=str, required=True,
                   help='Path to HDF5 dataset')
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='Path to pretrained DreaMS checkpoint')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--triplet_margin', type=float, default=0.5,
                   help='Triplet margin loss margin')
    p.add_argument('--triplet_weight', type=float, default=0.5,
                   help='Weight of triplet loss vs mask loss (0-1)')
    p.add_argument('--overlap_threshold', type=float, default=0.3,
                   help='Min Jaccard overlap for positive pairs')
    p.add_argument('--overlap_categories', type=str, default='NL,CF,ISO',
                   help='Rule categories for overlap computation (comma-separated)')
    p.add_argument('--save_dir', type=str, default='./contrastive_checkpoints')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


# ==============================================================================
# 规则重叠度计算工具
# ==============================================================================

def compute_batch_rule_vectors(
    engine: ChemicalRuleEngine,
    specs: torch.Tensor,
    padding_masks: torch.Tensor,
    categories: List[str],
) -> torch.Tensor:
    """
    为一个 batch 的谱图计算规则匹配向量。

    参数：
        engine: ChemicalRuleEngine 实例
        specs: (batch, n_peaks, 2) — [m/z, intensity]
        padding_masks: (batch, n_peaks) — True = 填充位
        categories: 用于计算重叠度的规则类别

    返回：
        match_vecs: (batch, n_rules) — 每张谱图中各规则的命中标志
    """
    batch_size = specs.shape[0]
    match_vecs_list = []

    for b in range(batch_size):
        mz = specs[b:b+1, :, 0]
        pad = padding_masks[b:b+1]
        mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(mz)
        vec = engine.get_rule_match_vectors(
            mz_diffs, mz_values=mz, precursor_mz=mz[:, 0],
            padding_mask=pad, categories=categories
        )
        match_vecs_list.append(vec)

    return torch.cat(match_vecs_list, dim=0)  # (batch, n_rules)


def build_triplets_from_overlap(
    match_vecs: torch.Tensor,
    overlap_threshold: float = 0.3,
    margin_min: float = 0.1,
) -> List[Tuple[int, int, int]]:
    """
    根据规则重叠度构建三元组 (anchor, positive, negative)。

    策略：对每个 anchor，选重叠度最高的其他样本作为 positive（需超过阈值），
          选重叠度最低的作为 negative（需低于 positive - margin_min）。

    参数：
        match_vecs: (batch, n_rules)
        overlap_threshold: 正样本对的最小 Jaccard 重叠度
        margin_min: 正/负样本对至少的重叠度差距

    返回：
        triplets: [(anchor_idx, positive_idx, negative_idx), ...]
    """
    batch = match_vecs.shape[0]
    # 计算 pairwise Jaccard 矩阵
    # intersection = A @ B^T (二进制向量的交集大小)
    intersection = match_vecs @ match_vecs.T  # (batch, batch)
    # union = |A| + |B| - intersection
    n_matches = match_vecs.sum(dim=-1, keepdim=True)  # (batch, 1)
    union = n_matches + n_matches.T - intersection
    overlap = intersection / union.clamp(min=1)  # (batch, batch)

    triplets = []
    for i in range(batch):
        # 排除自身
        scores = overlap[i].clone()
        scores[i] = -1.0

        # 正样本：最高重叠度且超过阈值
        best_pos_score, best_pos_idx = scores.max(dim=0)
        if best_pos_score.item() < overlap_threshold:
            continue

        # 负样本：最低重叠度且足够低于正样本
        worst_neg_score, worst_neg_idx = scores.min(dim=0)
        if worst_neg_score.item() > best_pos_score.item() - margin_min:
            continue

        triplets.append((i, best_pos_idx.item(), worst_neg_idx.item()))

    return triplets


# ==============================================================================
# 对比微调循环
# ==============================================================================

def train_contrastive(
    model: DreaMS,
    engine: ChemicalRuleEngine,
    dataloader,
    epochs: int = 10,
    lr: float = 1e-5,
    triplet_margin: float = 0.5,
    triplet_weight: float = 0.5,
    overlap_threshold: float = 0.3,
    overlap_categories: List[str] = None,
    save_dir: Path = Path('./contrastive_checkpoints'),
    device: torch.device = torch.device('cpu'),
):
    """
    规则驱动的对比学习微调。

    每步：
      1. 前向传播 → 编码所有谱图
      2. 计算规则匹配向量 → 构建三元组
      3. 计算 L = (1-α)*L_mask + α*L_triplet + β*L_preserve
      4. 反向传播 → 更新全部参数（低学习率）
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()
    engine = engine.to(device)

    # 保存原版嵌入用于 preservation loss（防止坍塌）
    # 先用 base model 对第一个 batch 计算初始嵌入，之后不再更新
    model_orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 全模型微调 + 极低学习率（对比学习不应破坏预训练结构）
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    print(f'   Trainable: {trainable}/{total} params (full model, low LR)')

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    triplet_loss_fn = nn.TripletMarginLoss(margin=triplet_margin, p=2)

    if overlap_categories is None:
        overlap_categories = ['NL', 'CF', 'ISO']

    history = {
        'epoch': [], 'step': [], 'loss_mask': [], 'loss_triplet': [],
        'n_triplets': [], 'avg_overlap_pos': [], 'avg_overlap_neg': [],
    }

    global_step = 0

    for epoch in range(epochs):
        epoch_losses_mask = []
        epoch_losses_trip = []
        epoch_n_trip = []

        for batch_idx, batch in enumerate(dataloader):
            # ---- 数据准备 ----
            if isinstance(batch, dict):
                spec = batch['spectrum']
                spec_real = batch.get('spec_real', None)
                mask = batch.get('mask', None)
                charge = batch.get('charge', None)
            elif isinstance(batch, (tuple, list)):
                spec = batch[0]
                spec_real = mask = charge = None
            else:
                spec = batch
                spec_real = mask = charge = None

            spec = spec.to(device) if isinstance(spec, torch.Tensor) else torch.as_tensor(spec, device=device)
            batch_size = spec.shape[0]
            padding_mask = spec[:, :, 0] == 0

            # ---- 规则匹配向量（用于构建三元组） ----
            with torch.no_grad():
                match_vecs = compute_batch_rule_vectors(
                    engine, spec, padding_mask, overlap_categories
                )
                triplets = build_triplets_from_overlap(
                    match_vecs, overlap_threshold=overlap_threshold
                )

            # ---- Mask prediction loss（保留预训练能力） ----
            if spec_real is not None and mask is not None:
                spec_real = spec_real.to(device)
                mask = mask.to(device)
                loss_mask, embs, _, _ = model.spec_ssl_step(spec, spec_real, mask, charge)
                loss_mask = loss_mask.sum() / loss_mask.numel()
            else:
                # 无预置 mask：动态生成
                embs = model(spec, charge)
                n_peaks = spec.shape[1]
                n_mask = max(1, int(n_peaks * 0.15))
                mask_bool = torch.zeros(batch_size, n_peaks, dtype=torch.bool, device=device)
                spec_mask = spec.clone()
                for b in range(batch_size):
                    idx = torch.randperm(n_peaks - 1, device=device)[:n_mask] + 1
                    mask_bool[b, idx] = True
                    spec_mask[b, idx, :] = 0.0
                loss_mask, _, _, _ = model.spec_ssl_step(spec_mask, spec, mask_bool, charge)
                loss_mask = loss_mask.sum() / loss_mask.numel()

            # ---- Contrastive loss（软版本：cos_sim 逼近规则重叠度） ----
            # 不硬推/硬拉，而是让嵌入相似度去追踪规则重叠度
            # 这比硬 triplet margin 更鲁棒，不易坍塌
            loss_contrastive = torch.tensor(0.0, device=device)
            n_pairs_used = 0
            avg_overlap = 0.0

            if len(triplets) > 0:
                anchor_idx = torch.tensor([t[0] for t in triplets], device=device)
                pos_idx = torch.tensor([t[1] for t in triplets], device=device)

                # 只做 anchor-positive 对（不推远负样本，避免错误地把相关分子推远）
                anchor_emb = embs[anchor_idx, 0, :]  # (n_pairs, d_model)
                pos_emb = embs[pos_idx, 0, :]

                # 计算余弦相似度
                cos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=-1)  # (n_pairs,)

                # 目标：cos_sim 应逼近规则重叠度
                target_overlaps = []
                for a_idx, p_idx, _ in triplets:
                    target_overlaps.append(
                        ChemicalRuleEngine.compute_rule_overlap(
                            match_vecs[a_idx], match_vecs[p_idx]
                        ).item()
                    )
                target = torch.tensor(target_overlaps, device=device)

                # MSE: (cos_sim - rule_overlap)^2
                loss_contrastive = F.mse_loss(cos_sim, target)
                n_pairs_used = len(triplets)
                avg_overlap = target.mean().item()

            # ---- Preservation loss：防止嵌入偏离原版太远 ----
            # 对 s_0 嵌入施加 L2 约束，避免坍塌
            loss_preserve = torch.tensor(0.0, device=device)
            preservation_weight = 0.01
            if len(triplets) > 0:
                anchor_idx = torch.tensor([t[0] for t in triplets], device=device)
                # 用小的正则化防止嵌入爆炸/收缩
                loss_preserve = (embs[anchor_idx, 0, :] ** 2).mean()

            # ---- 总损失：L_mask 主导，对比损失做引导 ----
            loss_total = (1.0 - triplet_weight) * loss_mask \
                       + triplet_weight * loss_contrastive \
                       + preservation_weight * loss_preserve

            # ---- 反向传播 ----
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # ---- 记录 ----
            epoch_losses_mask.append(loss_mask.item())
            epoch_losses_trip.append(loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else loss_contrastive)
            epoch_n_trip.append(n_pairs_used)

            history['step'].append(global_step)
            history['loss_mask'].append(loss_mask.item())
            history['loss_triplet'].append(loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else loss_contrastive)
            history['n_triplets'].append(n_pairs_used)
            history['avg_overlap_pos'].append(avg_overlap)
            history['avg_overlap_neg'].append(0.0)
            global_step += 1

            if batch_idx % 10 == 0:
                print(f'   Epoch {epoch+1}/{epochs} | Step {batch_idx} | '
                      f'mask_loss={loss_mask.item():.4f} | '
                      f'contra_loss={loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else loss_contrastive:.4f} | '
                      f'preserve={loss_preserve.item():.4f} | '
                      f'pairs={n_pairs_used}/{batch_size} | '
                      f'avg_overlap={avg_overlap:.3f}')

            # 每 2000 步保存 checkpoint
            if batch_idx > 0 and batch_idx % 2000 == 0:
                ckpt_path = save_dir / f'contrastive_step{global_step}.pt'
                torch.save({
                    'epoch': epoch, 'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                }, ckpt_path)
                print(f'   [Checkpoint] Step {global_step} saved')

        # ---- Epoch 总结 ----
        avg_mask = np.mean(epoch_losses_mask) if epoch_losses_mask else 0
        avg_contra = np.mean(epoch_losses_trip) if epoch_losses_trip else 0
        avg_npairs = np.mean(epoch_n_trip) if epoch_n_trip else 0
        history['epoch'].append(epoch)

        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch+1}/{epochs} Summary')
        print(f'  Avg mask loss:      {avg_mask:.4f}')
        print(f'  Avg contrastive loss: {avg_contra:.4f}')
        print(f'  Avg pairs/batch:    {avg_npairs:.1f}')
        print(f'{"=" * 60}\n')

        ckpt_path = save_dir / f'contrastive_epoch{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, ckpt_path)
        print(f'   Checkpoint saved: {ckpt_path}\n')

    return history


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Rule-driven Contrastive Fine-tuning [v4]')
    print(f'  Overlap categories: {args.overlap_categories}')

    overlap_categories = [c.strip() for c in args.overlap_categories.split(',')]

    # ---- 加载预训练 DreaMS ----
    ckpt_path = args.ckpt_path or str(PRETRAINED / 'ssl_model_server.pt')
    print(f'Loading pretrained model: {ckpt_path}')

    pkg = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'args' in pkg and 'state_dict' in pkg:
        state_dict = pkg['state_dict']
        from argparse import Namespace
        recon_args = Namespace(**pkg['args'])
        from dreams.utils.dformats import DataFormatA
        recon_args.dformat = DataFormatA()
        for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
                   'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
                   'min_intensity_ampl', 'max_ms_level']:
            if da in pkg['args']:
                setattr(recon_args.dformat, da, pkg['args'][da])
        recon_args.d_graphormer_params = 0
        
        from dreams.utils.data import SpectrumPreprocessor
        spec_preproc = SpectrumPreprocessor(dformat=recon_args.dformat,
                                            n_highest_peaks=recon_args.max_peaks_n)
        model = DreaMS(recon_args, spec_preproc)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f'   Loaded {len(state_dict)} params')
    else:
        raise ValueError(f'Unknown checkpoint format')

    # ---- 构建规则引擎 ----
    engine = ChemicalRuleEngine(tolerance=0.02, enable_categories=None)
    print(f'   Rule engine: {len(engine.rules)} rules ({engine.get_enabled_rules_summary()})')

    # ---- 准备数据 ----
    if args.dry_run:
        print('\n*** DRY RUN ***')
        from torch.utils.data import DataLoader, TensorDataset
        dummy_specs = []
        for _ in range(64):
            mz = torch.rand(30) * 1000.0
            mz = mz.sort().values
            intens = torch.rand(30)
            intens = intens / intens.max()
            dummy_specs.append(torch.stack([mz, intens], dim=-1))
        max_len = max(s.shape[0] for s in dummy_specs)
        specs_padded = torch.zeros(64, max_len, 2)
        for i, s in enumerate(dummy_specs):
            specs_padded[i, :s.shape[0]] = s
        dataloader = DataLoader(TensorDataset(specs_padded), batch_size=args.batch_size, shuffle=True)
        print(f'   Synthetic: {len(dummy_specs)} spectra')
    else:
        msdata = du.MSData.load(args.dataset_path)
        from dreams.utils.data import SpectrumPreprocessor
        spec_preproc = model.spec_preproc
        dataset = msdata.to_torch_dataset(spec_preproc)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f'   Dataset: {len(dataset)} spectra')

    # ---- 训练 ----
    print(f'\nStarting contrastive fine-tuning ({args.epochs} epochs)...')
    print(f'  triplet_weight={args.triplet_weight}, margin={args.triplet_margin}')
    print(f'  overlap_threshold={args.overlap_threshold}')
    print('=' * 60)

    history = train_contrastive(
        model=model,
        engine=engine,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        overlap_threshold=args.overlap_threshold,
        overlap_categories=overlap_categories,
        save_dir=Path(args.save_dir),
        device=device,
    )

    print('=' * 60)
    print('Contrastive fine-tuning complete!')
    print(f'   Final mask loss:    {history["loss_mask"][-1]:.4f}')
    print(f'   Final triplet loss: {history["loss_triplet"][-1]:.4f}')
    print(f'   Checkpoints: {args.save_dir}')


if __name__ == '__main__':
    main()
