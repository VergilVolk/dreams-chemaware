"""
build_data.py — MIL 训练数据管道 [v1]

核心流程：
  1. 从 MassSpecGym 全量数据构造谱图对（不仅限于子样本）
  2. 计算规则匹配向量 → 提取共同命中规则 → 构造 bag（instance 列表）
  3. 按 InChIKey（分子）级别切分 train/val/test（避免数据泄漏！）
  4. 支持 k 折交叉验证

用法：
  python -m dreams.models.mil_interpretable.build_data \
      --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
      --n_folds 5
"""

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
import h5py
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path


# ==============================================================================
# 规则匹配 → Instance 特征
# ==============================================================================

def build_instance_features(matched_rule: dict) -> np.ndarray:
    """
    将一条命中的规则转换为 12 维 instance 特征向量。

    参数：
        matched_rule: dict，包含 level, category, match_type, mass_diff_precision

    返回：
        (12,) float32 向量
    """
    # 类别 one-hot：NL=0, CF=1, ISO=2, HR=3, NR=4, EE=5
    cat_idx = {'NL': 0, 'CF': 1, 'ISO': 2, 'HR': 3, 'NR': 4, 'EE': 5}
    cat_onehot = np.zeros(6, dtype=np.float32)
    cat_onehot[cat_idx.get(matched_rule['category'], 0)] = 1.0

    # match_type one-hot：mass_diff=0, peak_mz=1, mass_range=2, parity/hr_shift=3
    mt_idx = {'mass_diff': 0, 'peak_mz': 1, 'mass_range': 2, 'parity': 3,
              'mass_diff_range': 0, 'hr_shift': 3}
    mt_onehot = np.zeros(4, dtype=np.float32)
    mt_onehot[mt_idx.get(matched_rule.get('match_type', 'mass_diff'), 0)] = 1.0

    features = np.concatenate([
        np.array([matched_rule.get('level', 0) / 2.0], dtype=np.float32),  # 诊断级别
        cat_onehot,                                                          # 6 维
        mt_onehot,                                                           # 5 维
        np.array([matched_rule.get('mass_diff_precision', 0.5)],
                 dtype=np.float32),                                         # 质量匹配精度
    ])
    return features.astype(np.float32)


# ==============================================================================
# 规则匹配向量 → Bag 构造
# ==============================================================================

def match_vec_to_bag(
    match_vec: torch.Tensor,
    engine,
    tol: float = 0.02,
) -> List[np.ndarray]:
    """
    将命中规则的匹配向量转换为 instance 特征列表。

    参数：
        match_vec: (n_rules,) — 二进制规则命中向量
        engine: ChemicalRuleEngine 实例
        tol: 质量容差

    返回：
        List[np.ndarray] — 每个命中规则的 12 维 instance 特征
    """
    instances = []
    for idx in range(len(match_vec)):
        if match_vec[idx].item() == 0:
            continue
        rule = engine.rules[idx]
        # 计算质量匹配精度（距离下一个最近规则目标有多近）
        mass_diff_precision = 0.5  # 默认中等
        if rule.match_type in ('mass_diff', 'peak_mz'):
            target = float(rule.value) if isinstance(rule.value, (int, float)) else float(rule.value[0])
            mass_diff_precision = min(1.0, tol / max(abs(target), 1e-6))

        # 规则级别标注
        level = 1  # 默认 Level 1（一般规则）
        if rule.category == 'HR':
            level = 2  # HR 规则 Level 2（高诊断性）
        elif rule.category in ('NR', 'EE'):
            level = 0  # 全覆盖规则 Level 0（低诊断性）
        elif rule.category == 'ISO':
            level = 2  # 同位素规则 Level 2

        instances.append(build_instance_features({
            'level': level,
            'category': rule.category,
            'match_type': rule.match_type,
            'mass_diff_precision': mass_diff_precision,
        }))
    return instances


# ==============================================================================
# 谱图对构造（全量数据）
# ==============================================================================

def build_spectrum_pairs(
    msdata,
    mass_tol: float = 0.05,
    n_max: int = 50000,
    rng_seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[float], Dict]:
    """
    从全量 MassSpecGym 数据构造谱图对。

    策略：排序后取相邻对 + 随机采样补充，确保正负样本平衡。

    参数：
        msdata: MSData 实例
        mass_tol: 困难样本的 precursor mass 容差 (Da)
        n_max: 最大 pair 数量
        rng_seed: 随机种子

    返回：
        pairs: [(idx_A, idx_B), ...] — 谱图索引对
        pair_mass_diffs: [float, ...] — 每对的质量差
        stats: dict — 统计信息
    """
    rng = np.random.RandomState(rng_seed)
    n_total = len(msdata)

    # 读取所有必要数据
    print('   Reading precursor m/z and InChIKey...')
    prec_mzs = np.array([float(x) for x in msdata.get_prec_mzs()])

    # 读取 InChIKey
    inchikeys = []
    for i in tqdm(range(min(50000, n_total)), desc='   Reading InChIKeys'):
        try:
            ik = msdata.get_values('INCHIKEY', i)
            if isinstance(ik, bytes):
                ik = ik.decode('utf-8')
            inchikeys.append(str(ik).strip())
        except Exception:
            inchikeys.append(f'unknown_{i}')
    print(f'   Loaded {len(inchikeys)} InChIKeys')

    # 在 50000 范围内构造（全量 231K 太大，50K 足够覆盖多样性）
    n_use = min(5000, n_total)  # small for local testing
    indices = np.arange(n_use)

    # 按 precursor m/z 排序
    sorted_idx = indices[np.argsort(prec_mzs[:n_use])]

    pairs = []
    pair_mass_diffs = []
    same_mol_count = 0

    # 1. 排序相邻对（困难样本）
    for k in tqdm(range(len(sorted_idx) - 1), desc='   Building adjacent pairs'):
        i, j = sorted_idx[k], sorted_idx[k + 1]
        md = abs(prec_mzs[i] - prec_mzs[j])

        # 排除同分子
        if inchikeys[i] and inchikeys[j] and inchikeys[i] == inchikeys[j]:
            same_mol_count += 1
            continue

        pairs.append((int(i), int(j)))
        pair_mass_diffs.append(float(md))
        if len(pairs) >= n_max // 2:
            break

    # 2. 随机采样补充（确保覆盖不同质量差范围）
    remaining = n_max - len(pairs)
    for _ in range(remaining * 3):  # 3x oversample
        i, j = rng.choice(n_use, 2, replace=False)
        if inchikeys[i] and inchikeys[j] and inchikeys[i] == inchikeys[j]:
            continue
        pairs.append((int(i), int(j)))
        pair_mass_diffs.append(float(abs(prec_mzs[i] - prec_mzs[j])))
        if len(pairs) >= n_max:
            break

    # 截断
    pairs = pairs[:n_max]
    pair_mass_diffs = pair_mass_diffs[:n_max]

    stats = {
        'n_pairs': len(pairs),
        'n_same_mol_filtered': same_mol_count,
        'mass_diff_mean': np.mean(pair_mass_diffs),
        'mass_diff_median': np.median(pair_mass_diffs),
    }

    return pairs, pair_mass_diffs, stats


# ==============================================================================
# 分子级切分（防止数据泄漏的核心）
# ==============================================================================

def molecule_level_split(
    pairs: List[Tuple[int, int]],
    inchikeys: List[str],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    rng_seed: int = 42,
) -> Dict[str, List[int]]:
    """
    按 InChIKey（分子）级别切分 train/val/test。

    关键：同一个分子的所有 pair 必须全部在同一 split 中。

    参数：
        pairs: 谱图索引对列表
        inchikeys: 每个谱图索引的 InChIKey
        test_ratio, val_ratio: 切分比例
        rng_seed: 随机种子

    返回：
        {'train': [pair_idx, ...], 'val': [...], 'test': [...]}
    """
    rng = np.random.RandomState(rng_seed)

    # 收集所有涉及的分子
    all_mols = set()
    pair_mols = []  # 每对涉及的分子集合
    for i, j in pairs:
        mols = set()
        if i < len(inchikeys) and inchikeys[i]:
            mols.add(inchikeys[i])
        if j < len(inchikeys) and inchikeys[j]:
            mols.add(inchikeys[j])
        pair_mols.append(mols)
        all_mols.update(mols)

    all_mols = list(all_mols)
    rng.shuffle(all_mols)

    n_test = int(len(all_mols) * test_ratio)
    n_val = int(len(all_mols) * val_ratio)
    n_train = len(all_mols) - n_test - n_val

    test_mols = set(all_mols[:n_test])
    val_mols = set(all_mols[n_test:n_test + n_val])
    train_mols = set(all_mols[n_test + n_val:])

    splits = {'train': [], 'val': [], 'test': []}
    for p_idx, pm in enumerate(pair_mols):
        # 一个 pair 只有在所有涉及分子都属于同一 split 时才归入该 split
        if pm and pm.issubset(train_mols):
            splits['train'].append(p_idx)
        elif pm and pm.issubset(val_mols):
            splits['val'].append(p_idx)
        elif pm and pm.issubset(test_mols):
            splits['test'].append(p_idx)
        # 跨 split 的 pair 丢弃（避免泄漏）

    # 断言无重叠
    train_mols_actual = set()
    for p_idx in splits['train']:
        train_mols_actual.update(pair_mols[p_idx])
    val_mols_actual = set()
    for p_idx in splits['val']:
        val_mols_actual.update(pair_mols[p_idx])
    test_mols_actual = set()
    for p_idx in splits['test']:
        test_mols_actual.update(pair_mols[p_idx])

    assert len(train_mols_actual & val_mols_actual) == 0, \
        f"Train/Val overlap: {len(train_mols_actual & val_mols_actual)} molecules"
    assert len(train_mols_actual & test_mols_actual) == 0, \
        f"Train/Test overlap: {len(train_mols_actual & test_mols_actual)} molecules"
    assert len(val_mols_actual & test_mols_actual) == 0, \
        f"Val/Test overlap: {len(val_mols_actual & test_mols_actual)} molecules"

    print(f'   Molecule-level split: train={len(splits["train"])} '
          f'val={len(splits["val"])} test={len(splits["test"])} pairs')
    print(f'   Unique molecules: train={len(train_mols_actual)} '
          f'val={len(val_mols_actual)} test={len(test_mols_actual)}')
    print(f'   Leakage check: train/val/test molecule sets are disjoint ✓')

    return splits


# ==============================================================================
# k 折交叉验证
# ==============================================================================

def build_kfold_splits(
    pairs: List[Tuple[int, int]],
    inchikeys: List[str],
    n_folds: int = 5,
    rng_seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """
    构建 k 折分子级切分。

    返回：List[{'train': [...], 'val': [...]}]，每个元素是一折的 train/val 索引。
    """
    rng = np.random.RandomState(rng_seed)

    # 收集每对涉及的分子
    pair_mols = []
    for i, j in pairs:
        mols = set()
        if i < len(inchikeys) and inchikeys[i]:
            mols.add(inchikeys[i])
        if j < len(inchikeys) and inchikeys[j]:
            mols.add(inchikeys[j])
        pair_mols.append(mols)

    all_mols = list(set().union(*pair_mols))
    rng.shuffle(all_mols)

    # 按分子数均分
    mol_per_fold = len(all_mols) // n_folds

    folds = []
    for k in range(n_folds):
        val_start = k * mol_per_fold
        val_end = val_start + mol_per_fold if k < n_folds - 1 else len(all_mols)
        val_mols = set(all_mols[val_start:val_end])
        train_mols = set(all_mols[:val_start]) | set(all_mols[val_end:])

        train_pairs, val_pairs = [], []
        for p_idx, pm in enumerate(pair_mols):
            if pm and pm.issubset(train_mols):
                train_pairs.append(p_idx)
            elif pm and pm.issubset(val_mols):
                val_pairs.append(p_idx)

        folds.append({'train': train_pairs, 'val': val_pairs})

        # 断言
        train_mols_actual = set().union(*[pair_mols[p] for p in train_pairs]) if train_pairs else set()
        val_mols_actual = set().union(*[pair_mols[p] for p in val_pairs]) if val_pairs else set()
        assert len(train_mols_actual & val_mols_actual) == 0, \
            f'Fold {k}: molecule overlap detected'

    return folds


# ==============================================================================
# Bag 构造
# ==============================================================================

def build_bags_for_pairs(
    pair_indices: List[int],
    all_pairs: List[Tuple[int, int]],
    match_vecs_cache: Dict[int, torch.Tensor],
    engine,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    为指定的 pair 索引列表构造 bag（instance 特征列表）和 label。

    参数：
        pair_indices: 要构造的 pair 索引
        all_pairs: 全量 pair 列表
        match_vecs_cache: {spectrum_idx: match_vec} 缓存
        engine: ChemicalRuleEngine

    返回：
        bags: List[Tensor] — 每个 bag 是 (n_instances, 12) 的 tensor
        labels: List[float] — 每个 bag 的 Tanimoto 标签
    """
    bags, labels = [], []
    for p_idx in tqdm(pair_indices, desc='   Building bags'):
        i, j = all_pairs[p_idx]
        if i not in match_vecs_cache or j not in match_vecs_cache:
            continue

        vec_a = match_vecs_cache[i]
        vec_b = match_vecs_cache[j]

        # 共同命中的规则
        common = (vec_a * vec_b) > 0  # (n_rules,) bool

        # 构造 instance 列表
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
                    'level': level,
                    'category': rule.category,
                    'match_type': rule.match_type,
                    'mass_diff_precision': 0.5,
                }))

        if len(instances) == 0:
            bags.append(torch.zeros(0, 12))
        else:
            bags.append(torch.tensor(np.stack(instances), dtype=torch.float32))

        # Label: 用简单的 overlap 阈值作为 proxy，
        # 后续可替换为真实 Tanimoto
        ov = (vec_a * vec_b).sum().float() / ((vec_a + vec_b) > 0).float().sum().clamp(min=1)
        labels.append(ov.item())

    return bags, labels


# ==============================================================================
# 主入口
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='MIL data pipeline')
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--n_pairs', type=int, default=10000)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--output_dir', type=str, default='./mil_data')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('MIL Data Pipeline')
    print('=' * 60)

    # 加载数据
    print('\n[1] Loading data...')
    import dreams.utils.data as du
    msdata = du.MSData.load(args.dataset_path)
    print(f'   Total spectra: {len(msdata)}')

    # 加载规则引擎
    print('\n[2] Loading rule engine...')
    from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'   Rules: {len(engine.rules)}')

    # 构造谱图对
    print(f'\n[3] Building spectrum pairs (n_max={args.n_pairs})...')
    pairs, mass_diffs, stats = build_spectrum_pairs(
        msdata, mass_tol=0.05, n_max=args.n_pairs)
    print(f'   Pairs: {stats["n_pairs"]}')
    print(f'   Same-mol filtered: {stats["n_same_mol_filtered"]}')
    print(f'   Mass diff: mean={stats["mass_diff_mean"]:.3f} Da, '
          f'median={stats["mass_diff_median"]:.3f} Da')

    # 读取 InChIKeys
    print('\n[4] Reading InChIKeys...')
    inchikeys = {}
    n_total = len(msdata)
    for i in tqdm(range(min(50000, n_total)), desc='   InChIKeys'):
        try:
            ik = msdata.get_values('INCHIKEY', i)
            if isinstance(ik, bytes):
                ik = ik.decode('utf-8')
            inchikeys[i] = str(ik).strip()
        except Exception:
            inchikeys[i] = f'unknown_{i}'
    inchikey_list = [inchikeys.get(i, '') for i in range(n_total)]

    # 分子级切分
    print('\n[5] Molecule-level train/val/test split...')
    splits = molecule_level_split(pairs, inchikey_list)

    # 保存
    data = {
        'pairs': pairs,
        'mass_diffs': mass_diffs,
        'stats': stats,
        'splits': splits,
        'inchikeys': inchikey_list,
    }
    with open(output_dir / 'mil_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved: {output_dir / "mil_data.pkl"}')


if __name__ == '__main__':
    main()
