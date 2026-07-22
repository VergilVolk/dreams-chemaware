"""
compare_models.py — 三个模型在平衡数据上的统一对比

加载 build_balanced_data.py 输出的 pickle 文件，
运行 5 折分子级 CV，报告三个模型的 Pearson r。

用法：
  python -m dreams.models.mil_interpretable.compare_models \
      --data_path ./mil_data/mil_balanced_data.pkl \
      --n_folds 5 --epochs 100
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='./mil_data/mil_balanced_data.pkl')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--hidden_dim', type=int, default=32)
    return p.parse_args()


# ---- Data loading ----
def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    pairs = data['pairs']
    labels = np.array(data['labels'], dtype=np.float32)
    match_vecs_cache = data['match_vecs_cache']
    inchikeys = data['inchikeys']
    engine = None  # will be loaded separately
    return pairs, labels, match_vecs_cache, inchikeys


# ---- Molecule-level k-fold split ----
def build_kfold_splits(pairs, inchikeys, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    pair_mols = []
    for a, b in pairs:
        mols = set()
        if a < len(inchikeys) and inchikeys[a]:
            mols.add(inchikeys[a])
        if b < len(inchikeys) and inchikeys[b]:
            mols.add(inchikeys[b])
        pair_mols.append(mols)
    all_mols = list(set().union(*pair_mols))
    rng.shuffle(all_mols)
    mol_per_fold = len(all_mols) // n_folds

    folds = []
    for k in range(n_folds):
        vs = k * mol_per_fold
        ve = vs + mol_per_fold if k < n_folds - 1 else len(all_mols)
        val_mols = set(all_mols[vs:ve])
        train_mols = set(all_mols[:vs]) | set(all_mols[ve:])
        train_pairs, val_pairs = [], []
        for pi, pm in enumerate(pair_mols):
            if pm and pm.issubset(train_mols):
                train_pairs.append(pi)
            elif pm and pm.issubset(val_mols):
                val_pairs.append(pi)
        folds.append({'train': train_pairs, 'val': val_pairs})
    return folds


# ---- Feature extraction ----
def extract_features(pairs, match_vecs_cache, engine):
    """提取聚合特征、平均池化特征、实例特征。"""
    feat_agg, feat_meanpool, instances_list, levels_list = [], [], [], []

    for a, b in pairs:
        if a not in match_vecs_cache or b not in match_vecs_cache:
            feat_agg.append(np.zeros(5, dtype=np.float32))
            feat_meanpool.append(np.zeros(12, dtype=np.float32))
            instances_list.append(torch.zeros(0, 12))
            levels_list.append(torch.zeros(0, dtype=torch.long))
            continue

        va = match_vecs_cache[a]
        vb = match_vecs_cache[b]
        inter = (va * vb).sum().float()
        union = ((va + vb) > 0).float().sum()
        ov = (inter / union.clamp(min=1)).item()
        common = (va * vb) > 0
        nc = common.sum().item()

        # Aggregated features (5-dim)
        nl = common[:214].sum().item()
        cf = common[214:214+102].sum().item()
        iso_hr = (common[214+102:214+102+8].sum() + common[-9:].sum()).item()
        feat_agg.append([
            ov, float(nc),
            iso_hr / max(nc, 1),
            nl / max(nc, 1),
            cf / max(nc, 1),
        ])

        # Instance features
        inst_feats = []
        lvls = []
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
                inst_feats.append(build_instance_features({
                    'level': level, 'category': rule.category,
                    'match_type': rule.match_type,
                    'mass_diff_precision': 0.5,
                }))
                lvls.append(level)

        # Mean-pool
        if inst_feats:
            feat_meanpool.append(np.stack(inst_feats).mean(axis=0))
        else:
            feat_meanpool.append(np.zeros(12, dtype=np.float32))

        # Instance tensor
        if inst_feats:
            instances_list.append(torch.tensor(np.stack(inst_feats), dtype=torch.float32))
        else:
            instances_list.append(torch.zeros(0, 12))

        levels_list.append(torch.tensor(lvls, dtype=torch.long) if lvls else torch.zeros(0, dtype=torch.long))

    return (np.array(feat_agg, dtype=np.float32),
            np.array(feat_meanpool, dtype=np.float32),
            instances_list, levels_list)


# ---- LR baselines ----
def eval_baseline(X, y, folds, valid_pairs):
    rs = []
    for split in folds:
        tr = [pi for pi in split['train'] if pi in valid_pairs]
        va = [pi for pi in split['val'] if pi in valid_pairs]
        if len(tr) < 10 or len(va) < 10:
            continue
        lr = Ridge(alpha=1.0)
        lr.fit(X[tr], y[tr])
        yp = lr.predict(X[va])
        r, _ = pearsonr(y[va], yp)
        rs.append(max(r, 0.0))
    return np.mean(rs), np.std(rs)


# ---- MIL training ----
def train_mil_fold(model, train_bags, train_labels, val_bags, val_labels,
                   epochs=100, lr=1e-3, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_r, best_state, counter = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for bag, label in zip(train_bags, train_labels):
            if bag.shape[0] == 0:
                continue
            pred, _ = model(bag)
            loss = F.mse_loss(pred, torch.tensor(label, dtype=torch.float32).unsqueeze(0))
            loss.backward()
            total_loss += loss.item()
            n += 1
        if n > 0:
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bag, label in zip(val_bags, val_labels):
                if bag.shape[0] == 0:
                    preds.append(0.0)
                else:
                    p, _ = model(bag)
                    preds.append(p.item())
                trues.append(label.item())
        r, _ = pearsonr(preds, trues)
        r = max(r, 0.0)

        if r > best_r:
            best_r = r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    # Final test r on val set
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bag, label in zip(val_bags, val_labels):
            if bag.shape[0] == 0:
                preds.append(0.0)
            else:
                p, _ = model(bag)
                preds.append(p.item())
            trues.append(label.item())
    r, _ = pearsonr(preds, trues)
    return max(r, 0.0)


def eval_mil(instances_list, labels, folds, valid_pairs, epochs=100, lr=1e-3, patience=20):
    rs = []
    for fold_idx, split in enumerate(folds):
        tr = [pi for pi in split['train'] if pi in valid_pairs]
        va = [pi for pi in split['val'] if pi in valid_pairs]
        if len(tr) < 10 or len(va) < 10:
            continue
        train_bags = [instances_list[i] for i in tr]
        train_labels = [labels[i] for i in tr]
        val_bags = [instances_list[i] for i in va]
        val_labels = [labels[i] for i in va]

        model = RuleAttentionMIL(instance_dim=12, hidden_dim=32)
        r = train_mil_fold(model, train_bags, train_labels, val_bags, val_labels,
                           epochs=epochs, lr=lr, patience=patience)
        rs.append(r)
        print(f'   Fold {fold_idx+1}: MIL r={r:.4f}')
    return np.mean(rs), np.std(rs)


# ---- Main ----
def main():
    args = parse_args()

    print('=' * 60)
    print('THREE-MODEL COMPARISON (Regression, Pearson r)')
    print('=' * 60)

    # Load data
    print(f'\n[1] Loading: {args.data_path}')
    pairs, labels, match_vecs_cache, inchikeys = load_data(args.data_path)
    print(f'   {len(pairs)} pairs, Tanimoto mean={labels.mean():.4f} std={labels.std():.4f}')

    # Load engine (for instance feature extraction)
    from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
    engine = ChemicalRuleEngine(tolerance=0.02)

    # Extract features
    print(f'\n[2] Extracting features...')
    X_agg, X_mp, instances_list, levels_list = extract_features(pairs, match_vecs_cache, engine)
    valid_pairs = list(range(len(pairs)))
    print(f'   Agg: {X_agg.shape}, MeanPool: {X_mp.shape}, Instances: {len(instances_list)}')

    # Build folds
    print(f'\n[3] Building {args.n_folds}-fold molecule-level splits...')
    folds = build_kfold_splits(pairs, inchikeys, n_folds=args.n_folds)
    for k, s in enumerate(folds):
        print(f'   Fold {k+1}: train={len(s["train"])} val={len(s["val"])}')

    # ---- Run all three models ----
    print(f'\n[4] Running models...')

    print(f'\n   --- Baseline 1: LR-agg (5-dim aggregated features) ---')
    r1_mean, r1_std = eval_baseline(X_agg, labels, folds, valid_pairs)
    print(f'   LR-agg: r = {r1_mean:.4f} +/- {r1_std:.4f}')

    print(f'\n   --- Baseline 2: LR-meanpool (12-dim mean-pooled) ---')
    r2_mean, r2_std = eval_baseline(X_mp, labels, folds, valid_pairs)
    print(f'   LR-meanpool: r = {r2_mean:.4f} +/- {r2_std:.4f}')

    print(f'\n   --- Model 3: Attention MIL ---')
    r3_mean, r3_std = eval_mil(instances_list, labels, folds, valid_pairs,
                                epochs=args.epochs, lr=args.lr, patience=args.patience)
    print(f'   MIL: r = {r3_mean:.4f} +/- {r3_std:.4f}')

    # ---- Final table ----
    print(f'\n{"="*60}')
    print(f'FINAL RESULTS')
    print(f'{"="*60}')
    print(f'  {"Model":20s}  {"Pearson r":15s}')
    print(f'  {"-"*20}  {"-"*15}')
    print(f'  {"LR-agg":20s}  {r1_mean:.4f} +/- {r1_std:.4f}')
    print(f'  {"LR-meanpool":20s}  {r2_mean:.4f} +/- {r2_std:.4f}')
    print(f'  {"Attention MIL":20s}  {r3_mean:.4f} +/- {r3_std:.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
