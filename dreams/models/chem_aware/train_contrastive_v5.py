"""
规则驱动的对比学习微调脚本 [v5 — 修复版]

v5 核心改进（针对 v4 四重失败原因的精准修复）：

  失败原因                   v4 表现                          v5 修复
  ─────────────────────────────────────────────────────────────────
  硬 triplet 二值化          0~1 连续信号→0/1 分类             MSE 回归（保留粒度）
  权重过高(0.5)              triplet_loss 压倒 mask_loss       α=0.05
  无正则化                   嵌入被系统性摧毁                  β=0.01 preservation
  随机采样                   简单样本无梯度，困难样本未暴露     Hard-pair 采样 + 按质量分桶
  无实时监控                 跑完 10 epoch 才发现坍缩          每 500 步 held-out AUC

三步验证计划（逐项验证，避免 v4 "多个改动同时堆"的覆辙）：
  Step 1: MSE 回归 + 随机配对(等量) + α=0.5 + β=0 → 确认 MSE 本身不导致坍缩
  Step 2: MSE 回归 + hard-pair 采样 + α=0.5 + β=0 → 确认 hard-pair 有增量价值
  Step 3: MSE 回归 + hard-pair 采样 + α=0.05 + β=0.01 → 完整 v5

关键实现细节（与初版的区别）：
  - 训练/监控严格按 fold 列切分，杜绝数据泄漏
  - Preservation loss = MSE(emb_current, emb_frozen_original)，不是 L2 模长
  - Hard-pair: 按 precursor mass 排序分桶 → 桶内最近邻配对，确保每个 batch 都有 hard pairs
  - Step 1 随机配对数量与 Step 2 hard-pair 数量对齐（而非全排列 N²）

用法：
  # 完整 v5（默认）
  python -m dreams.models.chem_aware.train_contrastive_v5 \
      --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
      --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
      --epochs 5 --batch_size 64

  # Step 1 消融
  python -m dreams.models.chem_aware.train_contrastive_v5 \
      --no_hard_mining --alpha 0.5 --beta 0.0 ...

  # Step 2 消融
  python -m dreams.models.chem_aware.train_contrastive_v5 \
      --alpha 0.5 --beta 0.0 ...

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import sys
import time
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import dreams.utils.data as du
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.definitions import PRETRAINED, SPECTRUM, PRECURSOR_MZ


# ==============================================================================
# 参数解析
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Rule-driven contrastive fine-tuning [v5]')

    # 数据
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='Pretrained DreaMS checkpoint (default: ssl_model_server.pt)')
    p.add_argument('--train_fold', type=str, default='train',
                   help='fold 列中用于训练的值 (默认 train)')
    p.add_argument('--val_fold', type=str, default='test',
                   help='fold 列中用于 held-out AUC 监控的值 (默认 test)')

    # 训练
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-6,
                   help='Learning rate (full model fine-tuning, keep low)')

    # v5 核心超参数
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Contrastive loss weight (v5 default: 0.05, Step 1/2: 0.5)')
    p.add_argument('--beta', type=float, default=0.01,
                   help='Preservation loss weight (v5 default: 0.01, Step 1/2: 0.0)')
    p.add_argument('--overlap_categories', type=str, default='NL,CF,ISO',
                   help='Rule categories for overlap (comma-separated)')

    # Loss 模式：MSE 回归 or Triplet
    p.add_argument('--loss_mode', type=str, default='mse', choices=['mse', 'triplet'],
                   help='mse = MSE(cos_sim, rule_overlap); triplet = rule-guided triplet margin')
    p.add_argument('--triplet_margin', type=float, default=0.2,
                   help='Triplet margin (only for loss_mode=triplet, default 0.2)')
    p.add_argument('--overlap_high', type=float, default=0.3,
                   help='Min rule overlap for positive candidates (triplet mode)')
    p.add_argument('--overlap_low', type=float, default=0.1,
                   help='Max rule overlap for negative candidates (triplet mode)')
    p.add_argument('--k_pairs', type=int, default=2,
                   help='Number of positive/negative candidates per anchor (triplet mode)')

    # Hard-pair 采样（MSE 模式）
    p.add_argument('--no_hard_mining', action='store_true',
                   help='Use random pairs instead of hard-pair (for Step 1 ablation)')
    p.add_argument('--sorted_batch_fraction', type=float, default=0.5,
                   help='Fraction of batches using mass-sorted sampling (0-1). '
                        '0 = all random shuffle; 1 = all mass-sorted. Default 0.5.')
    p.add_argument('--easy_pair_weight', type=float, default=0.2,
                   help='Relative weight of easy pairs vs hard pairs')

    # AUC 监控
    p.add_argument('--auc_monitor_steps', type=int, default=500,
                   help='Evaluate AUC every N steps (0 = disable)')
    p.add_argument('--auc_n_spectra', type=int, default=500,
                   help='Number of held-out spectra for AUC monitoring')
    p.add_argument('--auc_n_pairs', type=int, default=2000,
                   help='Number of pairs for AUC computation')

    # 保存
    p.add_argument('--save_dir', type=str, default='./contrastive_checkpoints_v5')
    p.add_argument('--dry_run', action='store_true')

    return p.parse_args()


# ==============================================================================
# ① 数据切分：按 fold 列严格隔离训练集和监控集
# ==============================================================================

def split_by_fold(msdata, train_fold: str = 'train', val_fold: str = 'test'):
    """
    按 fold 列分离训练和验证索引。

    返回：
        train_indices: list[int] — 训练集全局索引
        val_indices: list[int] — 验证/测试集全局索引
        fold_values: list[str] — 每条数据的 fold 值（用于日志）
    """
    n_total = len(msdata)
    train_idx, val_idx = [], []
    fold_counts = defaultdict(int)

    for i in range(n_total):
        try:
            f = msdata.get_values('fold', i)
            if isinstance(f, bytes):
                f = f.decode('utf-8')
            f = str(f).strip().lower()
        except Exception:
            f = 'unknown'

        fold_counts[f] += 1
        if f == train_fold.lower():
            train_idx.append(i)
        elif f == val_fold.lower():
            val_idx.append(i)
        # 其他 fold 值（如 'val'）忽略，既不用来训练也不用来监控

    # 如果按 fold 切分失败（列名不对或值不匹配），回退到随机切分
    fallback_used = False
    if len(train_idx) == 0 or len(val_idx) == 0:
        print(f'   ⚠ fold 列切分失败 (train={len(train_idx)}, val={len(val_idx)}), '
              f'fold 实际分布: {dict(fold_counts)}')
        print(f'   → 回退到随机 90/10 切分 (不保证分子级隔离，AUC 可能偏乐观)')
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_total)
        split = int(n_total * 0.9)
        train_idx = perm[:split].tolist()
        val_idx = perm[split:].tolist()
        fallback_used = True

    # 交集检查：train 和 val 绝不能有重叠
    train_set = set(train_idx)
    val_set = set(val_idx)
    overlap = train_set & val_set
    if overlap:
        raise RuntimeError(
            f'数据泄漏！训练集和监控集有 {len(overlap)} 条重叠！'
            f'这会导致 AUC 监控完全不可信。'
        )
    print(f'   Fold 实际分布: {dict(fold_counts)}')
    print(f'   训练集: {len(train_idx)} spectra'
          f'{"" if not fallback_used else " (随机 90%)"}')
    print(f'   监控集: {len(val_idx)} spectra'
          f'{"" if not fallback_used else " (随机 10% — AUC 可能偏乐观)"}')
    print(f'   交集检查: train ∩ val = {len(overlap)} ✓')
    return train_idx, val_idx


# ==============================================================================
# ③ 按 precursor mass 排序分桶的 BatchSampler
#    确保每个 batch 内的谱图有相近的 precursor mass，从而抓到 hard pairs
# ==============================================================================

class PrecursorMassBatchSampler(torch.utils.data.Sampler):
    """
    按 precursor mass 排序后分桶的 batch sampler。

    策略：
      1. 将训练集谱图按 precursor mass 排序
      2. 在一个宽度为 batch_size * 10 的滑动窗口内 shuffle
      3. 窗口内按 batch_size 切分为 batch
      4. 每个 epoch shuffle batch 的顺序

    效果：每个 batch 内的谱图 precursor mass 相近 → hard-pair 采样能稳定抓取
          窗口内 shuffle 提供一定随机性 → 不完全过拟合固定配对
    """

    def __init__(self, prec_mzs: np.ndarray, indices: List[int], batch_size: int):
        """
        参数：
            prec_mzs: (n_total,) — 所有谱图的 precursor mass
            indices: list[int] — 训练集在全局中的索引
            batch_size: int
        """
        self.batch_size = batch_size

        # 按 precursor mass 排序训练集索引
        train_precs = [(idx, prec_mzs[idx]) for idx in indices]
        train_precs.sort(key=lambda x: x[1])
        sorted_indices = [x[0] for x in train_precs]

        # 在窗口内 shuffle 后切分 batch
        # 窗口大小 = batch_size * 10，确保有一定随机性但质量仍然相近
        window = batch_size * 10
        rng = np.random.RandomState(42)
        self.batches = []

        i = 0
        while i < len(sorted_indices):
            window_indices = sorted_indices[i:i + window]
            rng.shuffle(window_indices)
            for j in range(0, len(window_indices), batch_size):
                batch = window_indices[j:j + batch_size]
                # 保留 ≥ half batch_size 的小 batch
                if len(batch) >= max(4, batch_size // 2):
                    self.batches.append(batch)
            i += window

        self.epoch_rng = np.random.RandomState()

    def __iter__(self):
        # 每个 epoch shuffle batch 顺序
        batch_order = list(range(len(self.batches)))
        self.epoch_rng.shuffle(batch_order)
        for idx in batch_order:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)


# ==============================================================================
# 规则匹配向量计算（批量）
# ==============================================================================

def compute_batch_rule_vectors(
    engine: ChemicalRuleEngine,
    specs: torch.Tensor,
    padding_masks: torch.Tensor,
    precursor_mzs: torch.Tensor,
    categories: List[str],
) -> torch.Tensor:
    """为一个 batch 的谱图计算规则匹配向量 (batch, n_rules)"""
    batch_size = specs.shape[0]
    match_vecs_list = []

    for b in range(batch_size):
        mz = specs[b:b+1, :, 0] 
        pad = padding_masks[b:b+1]
        prec = precursor_mzs[b:b+1]
        mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(mz)
        vec = engine.get_rule_match_vectors(
            mz_diffs, mz_values=mz, precursor_mz=prec,
            padding_mask=pad, categories=categories
        )
        match_vecs_list.append(vec)

    return torch.cat(match_vecs_list, dim=0)  # (batch, n_rules)


# ==============================================================================
# v5 核心：Pair 采样（hard-pair / 随机配对）
# ==============================================================================

def sample_pairs_from_batch(
    precursor_mzs: torch.Tensor,
    use_hard_mining: bool = True,
    n_random_pairs: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    从 batch 中采样谱图对。

    Hard-pair 模式 (use_hard_mining=True):
      - Batch 中的谱图已按 precursor mass 大致排序（来自 PrecursorMassBatchSampler）
      - 按 precursor mass 排序后，相邻元素配对 = 质量最接近的 hard pairs
      - 再从排序后的两端取 easy pairs（质量差最大）
      - 确保每 batch 都有稳定的 pair 数量

    随机模式 (use_hard_mining=False, Step 1 消融):
      - 随机采样 n_random_pairs 对（数量与 hard-pair 模式对齐）

    参数：
        precursor_mzs: (batch,) — 各谱图的 precursor m/z
        use_hard_mining: bool
        n_random_pairs: int | None — 随机配对数（None = batch_size）

    返回：
        hard_pairs: [(anchor_idx, partner_idx), ...]
        easy_pairs: [(anchor_idx, partner_idx), ...]
    """
    batch_size = precursor_mzs.shape[0]
    if batch_size < 2:
        return [], []

    if use_hard_mining:
        # ---- Hard-pair: 按 precursor mass 排序后相邻配对 ----
        sorted_idx = torch.argsort(precursor_mzs)

        hard_pairs = []
        for k in range(len(sorted_idx) - 1):
            i = sorted_idx[k].item()
            j = sorted_idx[k + 1].item()
            hard_pairs.append((i, j))

        # ---- Easy pair: 排序后从两端取（质量差最大的对） ----
        easy_pairs = []
        n_easy = max(1, len(hard_pairs) // 4)
        for k in range(min(n_easy, batch_size // 2)):
            i = sorted_idx[k].item()               # 从小端取
            j = sorted_idx[-(k + 1)].item()         # 从大端取
            easy_pairs.append((i, j))

        return hard_pairs, easy_pairs

    else:
        # ---- Step 1 消融：随机配对（数量与 hard-pair 对齐） ----
        n_pairs = n_random_pairs if n_random_pairs else max(1, batch_size - 1)
        hard_pairs = []
        for _ in range(n_pairs):
            i, j = torch.randint(0, batch_size, (2,))
            while i == j:
                j = torch.randint(0, batch_size, (1,))[0]
            hard_pairs.append((int(i.item()), int(j.item())))
        return hard_pairs, []


# ==============================================================================
# P0: 规则重叠度引导的 Triplet 采样
# ==============================================================================

def sample_triplets_by_overlap(
    match_vecs: torch.Tensor,
    k: int = 2,
    overlap_high: float = 0.3,
    overlap_low: float = 0.1,
) -> List[Tuple[int, int, int]]:
    """
    用规则重叠度排序选取 (anchor, positive, negative) 三元组。

    对 batch 中每个 anchor i：
      - 正样本：重叠度最高的 k 个中，筛选 overlap > overlap_high
      - 负样本：重叠度最低的 k 个中，筛选 overlap < overlap_low
      - 取第一个符合条件的正/负样本构成三元组

    参数：
        match_vecs: (batch, n_rules) — 规则匹配向量
        k: 候选数
        overlap_high: 正样本最小重叠度
        overlap_low: 负样本最大重叠度

    返回：
        triplets: [(anchor_idx, pos_idx, neg_idx), ...]
    """
    batch = match_vecs.shape[0]
    if batch < 3:
        return []

    # 计算 pairwise Jaccard
    intersection = match_vecs @ match_vecs.T  # (batch, batch)
    n_matches = match_vecs.sum(dim=-1, keepdim=True)  # (batch, 1)
    union = n_matches + n_matches.T - intersection
    overlap = intersection / union.clamp(min=1)  # (batch, batch)

    triplets = []
    for i in range(batch):
        scores = overlap[i].clone()
        scores[i] = -1.0  # 排除自身

        # 正样本候选：重叠度最高的 k 个
        pos_scores, pos_indices = torch.topk(scores, k=min(k + 1, batch - 1))
        valid_pos = pos_indices[pos_scores >= overlap_high]

        # 负样本候选：重叠度最低的 k 个
        neg_scores, neg_indices = torch.topk(scores, k=min(k + 1, batch - 1), largest=False)
        valid_neg = neg_indices[neg_scores <= overlap_low]

        if len(valid_pos) == 0 or len(valid_neg) == 0:
            continue

        triplets.append((i, valid_pos[0].item(), valid_neg[0].item()))

    return triplets


# ==============================================================================
# AUC 监控（held-out fold — 严格与训练集隔离）
# ==============================================================================

class AUCMonitor:
    """
    轻量级 AUC 监控器。

    仅从验证 fold（val_fold / test）中抽取谱图，
    定期提取嵌入并计算检索 AUC，用于检测嵌入坍缩。
    训练集和监控集绝不重叠。
    """

    def __init__(
        self,
        msdata,
        val_indices: List[int],
        spec_preproc,
        device: torch.device,
        n_spectra: int = 500,
        n_pairs: int = 2000,
        rng_seed: int = 12345,
    ):
        self.device = device
        self.spec_preproc = spec_preproc
        self.n_pairs = n_pairs
        self.rng = np.random.RandomState(rng_seed)

        # ---- 从验证 fold 中抽取 held-out 谱图 ----
        # 最多取 n_spectra 张
        sample_n = min(n_spectra, len(val_indices))
        chosen = self.rng.choice(val_indices, sample_n, replace=False)

        self.records = []
        for idx in chosen:
            try:
                smiles = msdata.get_values('smiles', int(idx))
                if isinstance(smiles, bytes):
                    smiles = smiles.decode('utf-8')
                smiles = str(smiles).strip()
                if len(smiles) < 2:
                    continue

                spec = torch.as_tensor(msdata.get_spectra(int(idx)), dtype=torch.float32)
                spec_pp = spec_preproc(spec.numpy(), high_form=False)
                spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)

                self.records.append({
                    'smiles': smiles,
                    'spec_t': spec_t,
                })
            except Exception:
                continue

            if len(self.records) >= n_spectra:
                break

        # ---- 预构建评估对（固定，跨轮次可比） ----
        self._build_eval_pairs()
        print(f'   AUC Monitor: {len(self.records)} spectra '
              f'(from {len(set(r["smiles"] for r in self.records))} unique molecules), '
              f'{len(self.pair_i)} eval pairs')

    def _build_eval_pairs(self):
        """构建固定的评估谱图对"""
        mol_to_indices = defaultdict(list)
        for i, r in enumerate(self.records):
            mol_to_indices[r['smiles']].append(i)

        multi = {k: v for k, v in mol_to_indices.items() if len(v) >= 2}
        all_mols = list(mol_to_indices.keys())

        n_pos = 0
        n_neg = 0
        target_pos = self.n_pairs // 2
        target_neg = self.n_pairs - target_pos
        pair_i, pair_j, pair_labels = [], [], []

        if multi:
            mol_list = list(multi.keys())
            while n_pos < target_pos:
                mol = mol_list[self.rng.randint(0, len(mol_list))]
                idxs = multi[mol]
                a, b = self.rng.choice(idxs, 2, replace=False)
                pair_i.append(a); pair_j.append(b); pair_labels.append(1)
                n_pos += 1

        while n_neg < target_neg:
            m1, m2 = self.rng.choice(all_mols, 2, replace=False)
            if m1 == m2:
                continue
            a = self.rng.choice(mol_to_indices[m1])
            b = self.rng.choice(mol_to_indices[m2])
            pair_i.append(a); pair_j.append(b); pair_labels.append(0)
            n_neg += 1

        self.pair_i = np.array(pair_i)
        self.pair_j = np.array(pair_j)
        self.pair_labels = np.array(pair_labels)

    @torch.no_grad()
    def evaluate(self, model: DreaMS) -> Dict[str, float]:
        """提取嵌入并计算检索 AUC"""
        model.eval()

        embs = []
        for r in self.records:
            spec_t = r['spec_t'].unsqueeze(0).to(self.device)
            emb = model(spec_t, None)
            embs.append(emb[:, 0, :].cpu())

        embeddings = torch.cat(embs, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        emb_i = embeddings[self.pair_i]
        emb_j = embeddings[self.pair_j]
        cos_sims = F.cosine_similarity(emb_i, emb_j, dim=-1).numpy()

        try:
            from sklearn import metrics
            fpr, tpr, _ = metrics.roc_curve(self.pair_labels, cos_sims)
            auc = float(metrics.auc(fpr, tpr))
        except Exception:
            auc = 0.5

        top1_correct = 0
        top5_correct = 0
        total_queries = 0
        for qi, qj, is_pos in zip(self.pair_i, self.pair_j, self.pair_labels):
            if not is_pos:
                continue
            total_queries += 1
            q_emb = embeddings[qi:qi+1]
            all_sims = F.cosine_similarity(q_emb, embeddings, dim=-1)
            all_sims[qi] = -1.0
            sorted_idx = all_sims.argsort(descending=True)
            if (sorted_idx[:1] == qj).any():
                top1_correct += 1
            if (sorted_idx[:5] == qj).any():
                top5_correct += 1

        top1 = top1_correct / max(total_queries, 1)
        top5 = top5_correct / max(total_queries, 1)

        pos_cos = cos_sims[self.pair_labels == 1].mean() if (self.pair_labels == 1).any() else 0.0
        neg_cos = cos_sims[self.pair_labels == 0].mean() if (self.pair_labels == 0).any() else 0.0

        model.train()
        return {
            'auc': auc,
            'top1': top1,
            'top5': top5,
            'pos_cos_mean': float(pos_cos),
            'neg_cos_mean': float(neg_cos),
            'separation': float(pos_cos - neg_cos),
        }


# ==============================================================================
# v5 训练循环
# ==============================================================================

def train_contrastive_v5(
    model: DreaMS,
    model_frozen: DreaMS,  # ② 冻结的原始模型副本，用于 preservation loss
    engine: ChemicalRuleEngine,
    dataloader,               # 纯随机 dataloader（Step 1 兼容 / fallback）
    epochs: int = 5,
    lr: float = 5e-6,
    alpha: float = 0.05,
    beta: float = 0.01,
    overlap_categories: List[str] = None,
    use_hard_mining: bool = True,
    loss_mode: str = 'mse',           # 'mse' or 'triplet'
    triplet_margin: float = 0.2,      # triplet margin
    overlap_high: float = 0.3,        # min overlap for positives (triplet)
    overlap_low: float = 0.1,         # max overlap for negatives (triplet)
    k_pairs: int = 2,                 # candidates per anchor (triplet)
    dataloader_sorted = None,  # 质量分桶 dataloader（v5 混合策略）
    sorted_batch_fraction: float = 0.5,  # 分桶 batch 的比例
    easy_pair_weight: float = 0.2,
    auc_monitor: Optional[AUCMonitor] = None,
    auc_monitor_steps: int = 500,
    save_dir: Path = Path('./contrastive_checkpoints_v5'),
    device: torch.device = torch.device('cpu'),
):
    """
    v5 对比学习微调。

    损失函数：
      L = L_mask + α·MSE(cos_sim(emb_A, emb_D), rule_overlap(A,D)) + β·MSE(emb, emb_frozen)

    每步流程：
      1. 前向传播 → 编码所有谱图
      2. 采样谱图对（hard-pair 或随机，取决于 use_hard_mining）
      3. 计算规则匹配向量 → 规则重叠度
      4. 计算三项损失 → 反向传播
      5. 每 N 步评估 held-out AUC
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model_frozen = model_frozen.to(device)
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad = False

    model.train()
    engine = engine.to(device)

    if overlap_categories is None:
        overlap_categories = ['NL', 'CF', 'ISO']

    # ---- 全模型微调 ----
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for _ in model.parameters())
    print(f'   Trainable: {trainable}/{total} params')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- 训练状态 ----
    history = {
        'step': [], 'loss_mask': [], 'loss_contrastive': [], 'loss_preserve': [],
        'loss_total': [], 'n_hard_pairs': [], 'n_easy_pairs': [],
        'avg_overlap_hard': [], 'avg_overlap_easy': [],
        'auc_heldout': [], 'auc_step': [],
        'lr': lr, 'alpha': alpha, 'beta': beta,
        'use_hard_mining': use_hard_mining,
    }

    global_step = 0
    best_auc = 0.0
    best_ckpt_path = None

    print(f'\n{"=" * 60}')
    print(f'v5 Training Config:')
    print(f'  Loss mode: {loss_mode}, α={alpha}, β={beta}, lr={lr}')
    if loss_mode == 'triplet':
        print(f'  Triplet: margin={triplet_margin}, overlap_high={overlap_high}, '
              f'overlap_low={overlap_low}, k={k_pairs}')
    else:
        print(f'  Pair sampling: {"hard-pair (mass-sorted)" if use_hard_mining else "random (ablation)"}')
    print(f'  Categories: {overlap_categories}')
    print(f'  AUC monitor: every {auc_monitor_steps} steps')
    print(f'{"=" * 60}\n')

    # ---- 混合 batch 迭代器 ----
    # 质量分桶 dataloader → hard-pair mining
    # 随机 shuffle dataloader → random pairs（保持全局多样性）
    use_mixed = (use_hard_mining and dataloader_sorted is not None)
    if use_mixed:
        print(f'  Mixed batches: sorted={sorted_batch_fraction:.0%}, '
              f'random={1-sorted_batch_fraction:.0%}')

    for epoch in range(epochs):
        epoch_losses = {'mask': [], 'contra': [], 'preserve': [], 'total': []}
        epoch_n_hard = []
        epoch_n_easy = []
        t0 = time.time()

        # 每个 epoch 重新创建迭代器
        if use_mixed:
            sorted_iter = iter(dataloader_sorted)
            random_iter = iter(dataloader)
            # epoch 长度 = 质量分桶 batch 数 / sorted_fraction（确保 cover 全部数据）
            steps_this_epoch = max(1, int(len(dataloader_sorted) / max(sorted_batch_fraction, 0.01)))
        else:
            data_iter = iter(dataloader)
            steps_this_epoch = len(dataloader)

        for batch_idx in range(steps_this_epoch):
            # ---- 决定从哪个 dataloader 取 batch ----
            if use_mixed and torch.rand(1).item() < sorted_batch_fraction:
                try:
                    batch = next(sorted_iter)
                except StopIteration:
                    sorted_iter = iter(dataloader_sorted)
                    batch = next(sorted_iter)
                batch_use_hard = True   # 质量分桶 → hard-pair mining
            elif use_mixed:
                try:
                    batch = next(random_iter)
                except StopIteration:
                    random_iter = iter(dataloader)
                    batch = next(random_iter)
                batch_use_hard = False  # 随机 shuffle → random pairs
            else:
                # 纯随机 dataloader（Step 1 兼容 / 消融模式）
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                batch_use_hard = use_hard_mining
            # ================================================================
            # 数据准备
            # ================================================================
            if isinstance(batch, dict):
                spec = batch[SPECTRUM]
                prec_mz = batch.get(PRECURSOR_MZ, None)
            elif isinstance(batch, (tuple, list)):
                spec = batch[0]
                prec_mz = batch[1] if len(batch) > 1 else None
            else:
                spec = batch
                prec_mz = None

            spec = spec.to(device) if isinstance(spec, torch.Tensor) else torch.as_tensor(spec, device=device)
            batch_size = spec.shape[0]
            padding_mask = spec[:, :, 0] == 0

            # Precursor m/z
            if prec_mz is not None:
                prec_mz = prec_mz.to(device) if isinstance(prec_mz, torch.Tensor) else torch.as_tensor(prec_mz, device=device)
                if prec_mz.dim() == 0:
                    prec_mz = prec_mz.unsqueeze(0)
            else:
                non_pad = spec[:, :, 0].clone()
                non_pad[padding_mask] = -1.0
                prec_mz = non_pad.max(dim=1).values

            # ================================================================
            # 规则匹配向量（用于计算重叠度标签，no_grad）
            # ================================================================
            with torch.no_grad():
                match_vecs = compute_batch_rule_vectors(
                    engine, spec, padding_mask, prec_mz, overlap_categories
                )

            # ================================================================
            # Pair/Triplet 采样
            # ================================================================
            if loss_mode == 'triplet':
                # P0: 规则重叠度排序 → (anchor, pos, neg) 三元组
                triplets = sample_triplets_by_overlap(
                    match_vecs, k=k_pairs,
                    overlap_high=overlap_high, overlap_low=overlap_low
                )
                hard_pairs = [(t[0], t[1]) for t in triplets]  # 兼容 preservation loss
                easy_pairs = []
            elif batch_use_hard:
                hard_pairs, easy_pairs = sample_pairs_from_batch(
                    prec_mz, use_hard_mining=True
                )
                triplets = []
            else:
                hard_pairs, easy_pairs = sample_pairs_from_batch(
                    prec_mz, use_hard_mining=False,
                    n_random_pairs=max(1, batch_size - 1)
                )
                triplets = []

            # ================================================================
            # 前向传播（获取当前嵌入）
            # ================================================================
            embs = model(spec, None)  # (batch, n_peaks, d_model)

            # ================================================================
            # ② Preservation loss: MSE(emb_current, emb_frozen_original)
            # ================================================================
            n_pairs_for_pres = len(triplets) if loss_mode == 'triplet' else len(hard_pairs)
            if beta > 0 and n_pairs_for_pres > 0:
                with torch.no_grad():
                    embs_frozen = model_frozen(spec, None)
                if loss_mode == 'triplet':
                    anchor_idx_for_pres = torch.tensor(
                        [t[0] for t in triplets], device=device
                    )
                else:
                    anchor_idx_for_pres = torch.tensor(
                        [p[0] for p in hard_pairs], device=device
                    )
                loss_preserve = F.mse_loss(
                    embs[anchor_idx_for_pres, 0, :],
                    embs_frozen[anchor_idx_for_pres, 0, :]
                )
            else:
                loss_preserve = torch.tensor(0.0, device=device)

            # ================================================================
            # Mask prediction loss
            # ================================================================
            n_peaks = spec.shape[1]
            mask_bool = torch.zeros(batch_size, n_peaks, dtype=torch.bool, device=device)
            spec_mask = spec.clone()
            for b in range(batch_size):
                valid_count = (~padding_mask[b]).sum().item()
                n_mask_b = max(1, int(valid_count * 0.15))
                valid_idx = torch.where(~padding_mask[b])[0]
                if len(valid_idx) > 1:
                    perm = torch.randperm(len(valid_idx) - 1, device=device)[:n_mask_b] + 1
                    idx = valid_idx[perm]
                else:
                    idx = valid_idx[:1]
                mask_bool[b, idx] = True
                spec_mask[b, idx, :] = 0.0
            loss_mask, _, _, _ = model.spec_ssl_step(spec_mask, spec, mask_bool, None)
            loss_mask = loss_mask.sum() / loss_mask.numel()

            # ================================================================
            # Contrastive loss
            # ================================================================
            loss_contrastive = torch.tensor(0.0, device=device)
            avg_overlap_hard = 0.0
            avg_overlap_easy = 0.0
            sep_pos_neg = 0.0  # Sep = mean(cos_pos) - mean(cos_neg)
            n_triplets_used = 0

            if loss_mode == 'triplet' and len(triplets) > 0:
                # ---- Triplet Margin Loss ----
                anchor_idx = torch.tensor([t[0] for t in triplets], device=device)
                pos_idx = torch.tensor([t[1] for t in triplets], device=device)
                neg_idx = torch.tensor([t[2] for t in triplets], device=device)

                anchor_emb = embs[anchor_idx, 0, :]
                pos_emb = embs[pos_idx, 0, :]
                neg_emb = embs[neg_idx, 0, :]

                loss_contrastive = F.triplet_margin_loss(
                    anchor_emb, pos_emb, neg_emb,
                    margin=triplet_margin, p=2
                )

                # Sep 指标
                cos_pos = F.cosine_similarity(anchor_emb, pos_emb, dim=-1)
                cos_neg = F.cosine_similarity(anchor_emb, neg_emb, dim=-1)
                sep_pos_neg = (cos_pos.mean() - cos_neg.mean()).item()

                # 规则重叠度（仅用于日志）
                overlaps_pos = torch.tensor([
                    ChemicalRuleEngine.compute_rule_overlap(
                        match_vecs[t[0]], match_vecs[t[1]]
                    ).item() for t in triplets
                ], device=device)
                avg_overlap_hard = overlaps_pos.mean().item()
                n_triplets_used = len(triplets)

            elif loss_mode == 'mse':
                # ---- MSE 回归（原 v5 逻辑） ----
                avg_overlap_easy = 0.0
                if len(hard_pairs) > 0:
                    anchor_idx = torch.tensor([p[0] for p in hard_pairs], device=device)
                    partner_idx = torch.tensor([p[1] for p in hard_pairs], device=device)
                    anchor_emb = embs[anchor_idx, 0, :]
                    partner_emb = embs[partner_idx, 0, :]
                    cos_sim = F.cosine_similarity(anchor_emb, partner_emb, dim=-1)
                    target = torch.tensor([
                        ChemicalRuleEngine.compute_rule_overlap(
                            match_vecs[a], match_vecs[p]
                        ).item() for a, p in hard_pairs
                    ], device=device)
                    loss_contrastive = F.mse_loss(cos_sim, target)
                    avg_overlap_hard = target.mean().item()

                if len(easy_pairs) > 0:
                    easy_anchor_idx = torch.tensor([p[0] for p in easy_pairs], device=device)
                    easy_partner_idx = torch.tensor([p[1] for p in easy_pairs], device=device)
                    easy_anchor_emb = embs[easy_anchor_idx, 0, :]
                    easy_partner_emb = embs[easy_partner_idx, 0, :]
                    easy_cos_sim = F.cosine_similarity(easy_anchor_emb, easy_partner_emb, dim=-1)
                    easy_target = torch.tensor([
                        ChemicalRuleEngine.compute_rule_overlap(
                            match_vecs[a], match_vecs[p]
                        ).item() for a, p in easy_pairs
                    ], device=device)
                    loss_contrastive_easy = F.mse_loss(easy_cos_sim, easy_target)
                    loss_contrastive = loss_contrastive + easy_pair_weight * loss_contrastive_easy
                    avg_overlap_easy = easy_target.mean().item()

            # ================================================================
            # 总损失: L = L_mask + α·L_contrastive + β·L_preservation
            # ================================================================
            loss_total = loss_mask + alpha * loss_contrastive + beta * loss_preserve

            # ================================================================
            # 反向传播
            # ================================================================
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ================================================================
            # 记录
            # ================================================================
            loss_mask_val = loss_mask.item()
            loss_contra_val = loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else loss_contrastive
            loss_pres_val = loss_preserve.item() if isinstance(loss_preserve, torch.Tensor) else loss_preserve
            loss_total_val = loss_total.item()

            epoch_losses['mask'].append(loss_mask_val)
            epoch_losses['contra'].append(loss_contra_val)
            epoch_losses['preserve'].append(loss_pres_val)
            epoch_losses['total'].append(loss_total_val)
            epoch_n_hard.append(len(hard_pairs))
            epoch_n_easy.append(len(easy_pairs))

            history['step'].append(global_step)
            history['loss_mask'].append(loss_mask_val)
            history['loss_contrastive'].append(loss_contra_val)
            history['loss_preserve'].append(loss_pres_val)
            history['loss_total'].append(loss_total_val)
            history['n_hard_pairs'].append(len(triplets) if loss_mode == 'triplet' else len(hard_pairs))
            history['n_easy_pairs'].append(len(easy_pairs))
            history['avg_overlap_hard'].append(avg_overlap_hard)
            history['avg_overlap_easy'].append(avg_overlap_easy)
            history.setdefault('sep_pos_neg', []).append(sep_pos_neg)
            history.setdefault('n_triplets', []).append(n_triplets_used if loss_mode == 'triplet' else len(hard_pairs))
            global_step += 1

            # ================================================================
            # 日志
            # ================================================================
            if batch_idx % 10 == 0:
                elapsed = time.time() - t0
                steps_done = batch_idx + 1
                print(f'   E{epoch+1} | Step {batch_idx:4d} | '
                      f'mask={loss_mask_val:.4f} '
                      f'contra={loss_contra_val:.4f} '
                      f'pres={loss_pres_val:.4f} | '
                      f'n_trip={n_triplets_used if loss_mode=="triplet" else len(hard_pairs)} '
                      f'Sep={sep_pos_neg:+.3f} | '
                      f'olap_h={avg_overlap_hard:.3f} '
                      f'{elapsed/steps_done:.1f}s/step')

                # n_trip 稀疏报警
                if loss_mode == 'triplet' and n_triplets_used < 10:
                    print(f'   [WARNING] Step {global_step}: n_trip={n_triplets_used} '
                          f'(< 10), gradient signal may be too sparse')

            # ================================================================
            # ① AUC 监控（仅用 held-out fold，与训练集无重叠）
            # ================================================================
            if auc_monitor is not None and auc_monitor_steps > 0 and global_step % auc_monitor_steps == 0:
                print(f'\n   --- AUC Check @ Step {global_step} (held-out fold) ---')
                metrics = auc_monitor.evaluate(model)
                history['auc_heldout'].append(metrics['auc'])
                history['auc_step'].append(global_step)

                print(f'   AUC={metrics["auc"]:.4f}  Top-1={metrics["top1"]:.4f}  '
                      f'Top-5={metrics["top5"]:.4f}')
                print(f'   Pos cos={metrics["pos_cos_mean"]:.4f}  '
                      f'Neg cos={metrics["neg_cos_mean"]:.4f}  '
                      f'Sep={metrics["separation"]:.4f}')

                # 保存最佳 checkpoint
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_ckpt_path = save_dir / f'best_auc_step{global_step}.pt'
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history,
                        'auc_metrics': metrics,
                    }, best_ckpt_path)
                    print(f'   >>> Best AUC checkpoint: {best_ckpt_path} (AUC={best_auc:.4f})')

                if metrics['auc'] < 0.65:
                    print(f'   ⚠ WARNING: AUC={metrics["auc"]:.4f} < 0.65! '
                          f'Possible embedding collapse.')
                print()

        # ================================================================
        # Epoch 总结
        # ================================================================
        avg_mask = np.mean(epoch_losses['mask'])
        avg_contra = np.mean(epoch_losses['contra'])
        avg_pres = np.mean(epoch_losses['preserve'])
        avg_total = np.mean(epoch_losses['total'])
        avg_hard = np.mean(epoch_n_hard) if epoch_n_hard else 0
        avg_easy = np.mean(epoch_n_easy) if epoch_n_easy else 0

        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch+1}/{epochs} Summary')
        print(f'  L_mask:         {avg_mask:.4f}')
        print(f'  L_contrastive:  {avg_contra:.4f}')
        print(f'  L_preserve:     {avg_pres:.4f}')
        print(f'  L_total:        {avg_total:.4f}')
        print(f'  Hard pairs:     {avg_hard:.1f}/batch')
        print(f'  Easy pairs:     {avg_easy:.1f}/batch')
        print(f'{"=" * 60}\n')

        # Epoch checkpoint
        ckpt_path = save_dir / f'contrastive_v5_epoch{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, ckpt_path)
        print(f'   Checkpoint saved: {ckpt_path}\n')

    return history, best_auc, best_ckpt_path


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Rule-driven Contrastive Fine-tuning [v5]')
    print(f'  Pair mode: {"random (Step 1 ablation)" if args.no_hard_mining else "hard-pair mining"}')

    overlap_categories = [c.strip() for c in args.overlap_categories.split(',')]

    # =========================================================================
    # 加载预训练 DreaMS
    # =========================================================================
    ckpt_path = args.ckpt_path or str(PRETRAINED / 'ssl_model_server.pt')
    print(f'\nLoading pretrained model: {ckpt_path}')

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

        spec_preproc = du.SpectrumPreprocessor(dformat=recon_args.dformat,
                                               n_highest_peaks=recon_args.max_peaks_n)
        model = DreaMS(recon_args, spec_preproc)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f'   Loaded {len(state_dict)} params')
    else:
        raise ValueError(f'Unknown checkpoint format')

    # =========================================================================
    # ② 冻结原始模型副本（用于 preservation loss）
    # =========================================================================
    model_frozen = copy.deepcopy(model)
    model_frozen.eval()
    for p in model_frozen.parameters():
        p.requires_grad = False
    print(f'   Frozen reference model created for preservation loss')

    # =========================================================================
    # 构建规则引擎
    # =========================================================================
    engine = ChemicalRuleEngine(tolerance=0.02, enable_categories=None)
    print(f'   Rule engine: {len(engine.rules)} rules ({engine.get_enabled_rules_summary()})')

    # =========================================================================
    # 准备数据
    # =========================================================================
    if args.dry_run:
        print('\n*** DRY RUN — 验证所有组件接口 ***')
        from torch.utils.data import DataLoader, TensorDataset

        # 模拟真实数据格式
        dummy_specs = []
        dummy_prec = []
        for _ in range(64):
            mz = torch.rand(30) * 1000.0
            mz = mz.sort().values
            intens = torch.rand(30)
            intens = intens / intens.max()
            dummy_specs.append(torch.stack([mz, intens], dim=-1))
            dummy_prec.append(mz.max().item())
        max_len = max(s.shape[0] for s in dummy_specs)
        specs_padded = torch.zeros(64, max_len, 2)
        for i, s in enumerate(dummy_specs):
            specs_padded[i, :s.shape[0]] = s
        prec_tensor = torch.tensor(dummy_prec, dtype=torch.float32)
        dummy_dataset = TensorDataset(specs_padded, prec_tensor)
        dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)
        auc_monitor = None
        print(f'   Synthetic: {len(dummy_specs)} spectra')
        print(f'   All imports OK, model forward OK, engine OK')
    else:
        # =====================================================================
        # ① 加载数据 + 按 fold 切分
        # =====================================================================
        print(f'\n[Data] Loading: {args.dataset_path}')
        msdata = du.MSData.load(args.dataset_path)
        print(f'   Total spectra: {len(msdata)}')
        print(f'   Columns: {msdata.columns()}')

        spec_preproc = model.spec_preproc

        train_indices, val_indices = split_by_fold(
            msdata, train_fold=args.train_fold, val_fold=args.val_fold
        )

        # 获取所有 precursor mass（用于排序分桶 batch sampler）
        all_prec_mzs = msdata.get_prec_mzs()

        # 创建完整 dataset（所有谱图），然后用 Subset 取训练部分
        full_dataset = msdata.to_torch_dataset(spec_preproc)
        from torch.utils.data import Subset, DataLoader
        train_dataset = Subset(full_dataset, train_indices)

        # ③ DataLoader：质量分桶 + 随机 shuffle 混合
        #    纯质量分桶 → 模型只看到相似质量样本 → 丢失全局检索能力
        #    纯随机 shuffle → hard pairs 几乎不存在 → 规则信号浪费
        #    混合策略：sorted_batch_fraction 控制分桶比例
        if not args.no_hard_mining:
            sorted_sampler = PrecursorMassBatchSampler(
                all_prec_mzs, train_indices, args.batch_size
            )
            dataloader_sorted = DataLoader(full_dataset, batch_sampler=sorted_sampler)
            dataloader_random = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            dataloader = dataloader_random  # 传给训练循环的随机 dataloader
            print(f'   Batch sampler: mixed (sorted={args.sorted_batch_fraction:.0%}, '
                  f'random={1-args.sorted_batch_fraction:.0%})')
            print(f'     Sorted: {len(sorted_sampler)} batches of ~{args.batch_size}')
            print(f'     Random: ~{len(train_indices)//args.batch_size} batches of ~{args.batch_size}')
        else:
            # Step 1 消融：纯随机
            dataloader_sorted = None
            dataloader_random = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            dataloader = dataloader_random  # 兼容旧路径
            print(f'   Batch sampler: random shuffle (ablation)')

        # =====================================================================
        # ① AUC 监控器（仅用验证 fold — 与训练集严格隔离）
        # =====================================================================
        if args.auc_monitor_steps > 0:
            print(f'\n[Setup] AUC Monitor (held-out fold={args.val_fold})')
            auc_monitor = AUCMonitor(
                msdata=msdata,
                val_indices=val_indices,
                spec_preproc=spec_preproc,
                device=device,
                n_spectra=args.auc_n_spectra,
                n_pairs=args.auc_n_pairs,
            )
        else:
            auc_monitor = None

    # =========================================================================
    # 训练
    # =========================================================================
    print(f'\nStarting v5 contrastive fine-tuning ({args.epochs} epochs)...')
    print(f'  α={args.alpha}, β={args.beta}, lr={args.lr}')
    print(f'  Pair sampling: {"random (ablation)" if args.no_hard_mining else "hard-pair (mass-sorted)"}')
    print(f'  AUC monitor: every {args.auc_monitor_steps} steps')
    print('=' * 60)

    history, best_auc, best_ckpt_path = train_contrastive_v5(
        model=model,
        model_frozen=model_frozen,
        engine=engine,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        overlap_categories=overlap_categories,
        use_hard_mining=not args.no_hard_mining,
        loss_mode=args.loss_mode,
        triplet_margin=args.triplet_margin,
        overlap_high=args.overlap_high,
        overlap_low=args.overlap_low,
        k_pairs=args.k_pairs,
        dataloader_sorted=dataloader_sorted,
        sorted_batch_fraction=args.sorted_batch_fraction,
        easy_pair_weight=args.easy_pair_weight,
        auc_monitor=auc_monitor,
        auc_monitor_steps=args.auc_monitor_steps,
        save_dir=Path(args.save_dir),
        device=device,
    )

    # =========================================================================
    # 最终报告
    # =========================================================================
    print('=' * 60)
    print('v5 Contrastive Fine-tuning Complete!')
    print(f'   Best held-out AUC (fold={args.val_fold}): {best_auc:.4f}')
    print(f'   Best checkpoint:   {best_ckpt_path}')
    print(f'   All checkpoints:   {args.save_dir}')
    print('=' * 60)

    # 保存训练历史
    import json
    history_path = Path(args.save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        serializable = {}
        for k, v in history.items():
            if isinstance(v, list):
                serializable[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                serializable[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        json.dump(serializable, f, indent=2)
    print(f'   Training history: {history_path}')


if __name__ == '__main__':
    main()
