"""
train_mil.py — Attention MIL 两阶段训练 + 验证 [v1]

阶段 A: 分类 + 诊断对齐loss → 训练模型主体
阶段 B: Temperature scaling → 校准置信度（仅用 val 集）
阶段 C: 测试集一次性评估（AUC + Mann-Whitney + Reliability Diagram）

用法：
  # full run with 5-fold CV
  python -m dreams.models.mil_interpretable.train_mil \
      --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
      --n_pairs 10000 --n_folds 5 --epochs 200

  # dry run with synthetic data
  python -m dreams.models.mil_interpretable.train_mil --dry_run
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
import time

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL


# ==============================================================================
# 损失：MSE 回归
# ==============================================================================


# ==============================================================================
# 训练工具
# ==============================================================================

def collate_bags(batch):
    """
    DataLoader collate：将变长 bag 列表拼接为单 batch。

    返回：
        instances_list: List[Tensor] — 各 bag 的 instance tensor
        labels: (batch,) — label tensor
        levels_list: List[Tensor] — 各 bag 的 level tensor
    """
    bags, labels = zip(*batch)
    return list(bags), torch.tensor(labels, dtype=torch.float32)


class BagDataset(torch.utils.data.Dataset):
    """简单的 bag dataset wrapper"""
    def __init__(self, bags, labels, levels_list):
        self.bags = bags
        self.labels = labels
        self.levels_list = levels_list

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx], self.levels_list[idx]


def extract_levels(instances, engine):
    """从 instance 特征提取 level 信息。"""
    if instances.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    # level 在 instance 特征第 0 维 (level/2.0)，恢复为 0/1/2
    levels = (instances[:, 0] * 2.0).round().long().clamp(0, 2)
    return levels


# ==============================================================================
# 阶段 A：主训练
# ==============================================================================

def train_epoch(model, dataloader, optimizer):
    """训练一个 epoch（MSE 回归）。"""
    model.train()
    total_loss = 0.0
    n_bags = 0
    for bags, labels, levels_list in dataloader:
        batch_loss = 0.0
        batch_n = 0
        for bag, label, levels in zip(bags, labels, levels_list):
            if bag.shape[0] == 0:
                continue
            pred, attn = model(bag)
            loss = F.mse_loss(pred, label.float().unsqueeze(0))
            loss.backward()
            batch_loss += loss.item()
            batch_n += 1
        if batch_n > 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss
            n_bags += batch_n
    return {'loss': total_loss / max(n_bags, 1)}


@torch.no_grad()
def evaluate_pearson_r(model, dataloader):
    """计算验证集 Pearson r（预测 vs 真实 Tanimoto）。"""
    model.eval()
    preds, labels = [], []
    for bags, lbls, _ in dataloader:
        for bag, label in zip(bags, lbls):
            if bag.shape[0] == 0:
                preds.append(0.0)
            else:
                pred, _ = model(bag)
                preds.append(pred.item())
            labels.append(label.item())
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(preds, labels)
        return max(r, 0.0)  # 负相关视为 0
    except Exception:
        return 0.0


# ==============================================================================
# （阶段 B/C 已移除 — 回归模式不需要温度校准和分类评估）
# ==============================================================================


# ==============================================================================
# 完整训练流程（单折）
# ==============================================================================

def run_single_fold(
    model,
    train_bags, train_labels, train_levels,
    val_bags, val_labels, val_levels,
    test_bags, test_labels, test_levels,
    epochs=200,
    lr=1e-3,
    patience=20,
    batch_size=1,
):
    """单折训练（MSE 回归）。"""
    train_dataset = BagDataset(train_bags, train_labels, train_levels)
    val_dataset = BagDataset(val_bags, val_labels, val_levels)
    test_dataset = BagDataset(test_bags, test_labels, test_levels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_r = 0.0
    best_state = None
    patience_counter = 0

    print(f'   Training (max {epochs} epochs, MSE regression)...')
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer)
        val_r = evaluate_pearson_r(model, val_loader)

        if epoch % 20 == 0:
            print(f'     Epoch {epoch:3d}: val_r={val_r:.4f}  loss={train_metrics["loss"]:.4f}')

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'     Early stop at epoch {epoch}')
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    test_r = evaluate_pearson_r(model, test_loader)
    print(f'   Test Pearson r: {test_r:.4f}  (best val r: {best_val_r:.4f})')

    return {'test_r': test_r, 'best_val_r': best_val_r}


# ==============================================================================
# 5 折交叉验证
# ==============================================================================

def run_cross_validation(
    bags, labels, levels_list, folds,
    instance_dim=12, hidden_dim=32,
    epochs=200, lr=1e-3, patience=20,
):
    """5 折交叉验证（MSE 回归），返回每折 test Pearson r。"""
    all_results = []
    for fold_idx, split in enumerate(folds):
        print(f'\nFold {fold_idx + 1}/{len(folds)}')

        train_idx = split['train']; val_idx = split['val']
        train_bags = [bags[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_levels = [levels_list[i] for i in train_idx]
        val_bags = [bags[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_levels = [levels_list[i] for i in val_idx]

        model = RuleAttentionMIL(instance_dim=instance_dim, hidden_dim=hidden_dim)
        result = run_single_fold(
            model,
            train_bags, train_labels, train_levels,
            val_bags, val_labels, val_levels,
            val_bags, val_labels, val_levels,
            epochs=epochs, lr=lr, patience=patience,
        )
        all_results.append(result)

    rs = [r['test_r'] for r in all_results]
    print(f'\nCV Summary: Pearson r = {np.mean(rs):.4f} +/- {np.std(rs):.4f}')
    return all_results


# ==============================================================================
# 主入口
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='MIL Training')
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--data_dir', type=str, default='./mil_data')
    p.add_argument('--n_pairs', type=int, default=10000)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print('=' * 60)
        print('MIL DRY RUN — synthetic 10-sample test')
        print('=' * 60)

        model = RuleAttentionMIL(instance_dim=12, hidden_dim=32)
        # 合成 bag
        n_rules = 335
        dummy_bags, dummy_labels, dummy_levels = [], [], []
        for _ in range(10):
            n_instances = np.random.randint(5, 30)
            bag = torch.randn(n_instances, 12)
            # level in feature[0]: clamp to [0, 2]
            bag[:, 0] = torch.randint(0, 3, (n_instances,)).float() / 2.0
            levels = (bag[:, 0] * 2.0).round().long().clamp(0, 2)
            dummy_bags.append(bag)
            dummy_labels.append(torch.tensor(float(np.random.rand() > 0.5)))
            dummy_levels.append(levels)

        # 假切分
        folds = [{'train': list(range(6)), 'val': list(range(6, 10))} for _ in range(2)]

        results = run_cross_validation(
            dummy_bags, dummy_labels, dummy_levels, folds,
            epochs=5, lr=1e-3, patience=5,
        )
        rs = [r['test_r'] for r in results]
        print(f'\nDry run passed! Pearson r: [{rs}]')
        print('All components: model forward, MSE regression, evaluation — OK')
        return

    # ---- 真实数据 ----
    print('=' * 60)
    print('MIL Full Training Pipeline')
    print('=' * 60)

    # 加载数据
    import dreams.utils.data as du
    from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
    from dreams.models.mil_interpretable.build_data import (
        build_spectrum_pairs, molecule_level_split, build_kfold_splits,
        match_vec_to_bag, build_instance_features,
    )

    print('\n[1] Loading data and engine...')
    msdata = du.MSData.load(args.dataset_path)
    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'   {len(msdata)} spectra, {len(engine.rules)} rules')

    # 构造谱图对
    print(f'\n[2] Building pairs (n_max={args.n_pairs})...')
    pairs, mass_diffs, stats = build_spectrum_pairs(msdata, n_max=args.n_pairs)
    print(f'   {stats["n_pairs"]} pairs')

    # 读取 InChIKeys
    print('\n[3] Reading InChIKeys and computing match vectors...')
    inchikey_list = []
    match_vecs_cache = {}
    n_use = min(5000, len(msdata))  # small for local CPU testing

    import dreams.utils.data as du
    import dreams.utils.dformats as dformats
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    for i in tqdm(range(n_use), desc='   Spectra'):
        try:
            ik = msdata.get_values('INCHIKEY', i)
            if isinstance(ik, bytes):
                ik = ik.decode('utf-8')
            inchikey_list.append(str(ik).strip())

            spec = torch.as_tensor(msdata.get_spectra(i), dtype=torch.float32)
            spec_pp = spec_preproc(spec.numpy(), high_form=False)
            spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)
            mz = spec_t[:, 0].unsqueeze(0)
            n_peaks = mz.shape[1]
            pad = mz[:, 0] == 0
            mz_diffs = torch.abs(mz.unsqueeze(-1) - mz.unsqueeze(-2))
            vec = engine.get_rule_match_vectors(
                mz_diffs, mz_values=mz,
                precursor_mz=mz[:, 0].unsqueeze(0) if mz.shape[1] > 0 else None,
                padding_mask=pad, categories=['NL', 'CF', 'ISO', 'HR']
            )
            match_vecs_cache[i] = vec.squeeze(0)
        except Exception:
            inchikey_list.append('error')
            match_vecs_cache[i] = torch.zeros(len(engine.rules))

    # 构造 bags
    print('\n[4] Building bags...')
    bags, labels, levels_list = [], [], []
    for pi, (i, j) in enumerate(tqdm(pairs, desc='   Bags')):
        if i not in match_vecs_cache or j not in match_vecs_cache:
            continue
        vec_a = match_vecs_cache[i]
        vec_b = match_vecs_cache[j]
        common = (vec_a * vec_b) > 0

        instances = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category == 'HR':
                    level = 2
                elif rule.category in ('NR', 'EE'):
                    level = 0
                elif rule.category == 'ISO':
                    level = 2
                instances.append(build_instance_features({
                    'level': level, 'category': rule.category,
                    'match_type': rule.match_type,
                    'mass_diff_precision': 0.5,
                }))

        if len(instances) == 0:
            bag = torch.zeros(0, 12)
            lvls = torch.zeros(0, dtype=torch.long)
        else:
            bag = torch.tensor(np.stack(instances), dtype=torch.float32)
            lvls = (bag[:, 0] * 2.0).round().long().clamp(0, 2)

        # Label = Tanimoto (real structural similarity from SMILES, NOT rule overlap!)
        try:
            smi_a = msdata.get_values('smiles', int(i))
            smi_b = msdata.get_values('smiles', int(j))
            if isinstance(smi_a, bytes): smi_a = smi_a.decode('utf-8')
            if isinstance(smi_b, bytes): smi_b = smi_b.decode('utf-8')
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            mol_a = Chem.MolFromSmiles(str(smi_a).strip())
            mol_b = Chem.MolFromSmiles(str(smi_b).strip())
            if mol_a is not None and mol_b is not None:
                fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
                fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
                tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_b)
            else:
                tanimoto = 0.0
        except Exception:
            tanimoto = 0.0

        bags.append(bag)
        labels.append(tanimoto)
        levels_list.append(lvls)

    # 统计 bag size
    bag_sizes = [b.shape[0] for b in bags]
    empty_frac = (np.array(bag_sizes) == 0).mean()
    print(f'   Bags: {len(bags)}, empty={empty_frac:.1%}, '
          f'size (min/mean/max)={min(bag_sizes)}/{np.mean(bag_sizes):.1f}/{max(bag_sizes)}')

    # 分子级 k 折切分
    print(f'\n[5] Molecule-level {args.n_folds}-fold split...')
    folds = build_kfold_splits(pairs, inchikey_list, n_folds=args.n_folds)

    # 训练
    print(f'\n[6] Training ({args.n_folds}-fold CV)...')
    results = run_cross_validation(
        bags, labels, levels_list, folds,
        instance_dim=12, hidden_dim=args.hidden_dim,
        epochs=args.epochs, lr=args.lr,
        patience=args.patience,
    )

    print('\nDone!')


if __name__ == '__main__':
    main()
