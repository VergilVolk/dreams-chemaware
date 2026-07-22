"""
baseline.py — Linear Regression baselines for MIL comparison (REGRESSION)

两个基线：
  基线1: 聚合特征 (5维) → LinearRegression/Ridge
  基线2: 实例特征平均池化 (12维) → LinearRegression/Ridge

与 MIL 模型使用完全相同的分子级 k 折切分，报告 Pearson r。

用法：
  python -m dreams.models.mil_interpretable.baseline \
      --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
      --n_pairs 2000 --n_folds 5
"""

import torch
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse
from pathlib import Path
from collections import defaultdict

from dreams.models.mil_interpretable.build_data import build_instance_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--n_pairs', type=int, default=2000)
    p.add_argument('--n_folds', type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    import dreams.utils.data as du
    import dreams.utils.dformats as dformats
    from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
    from dreams.models.mil_interpretable.build_data import (
        build_spectrum_pairs, build_kfold_splits,
    )

    print('=' * 60)
    print('BASELINE: Linear Regression on Rule Features (REGRESSION)')
    print('=' * 60)

    msdata = du.MSData.load(args.dataset_path)
    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Data: {len(msdata)} spectra, Engine: {len(engine.rules)} rules')

    # Build pairs
    print(f'\n[1] Building pairs...')
    pairs, mass_diffs, stats = build_spectrum_pairs(msdata, n_max=args.n_pairs)
    print(f'   {stats["n_pairs"]} pairs')

    # InChIKeys
    print(f'\n[2] Reading InChIKeys...')
    inchikey_list = []
    for i in tqdm(range(min(5000, len(msdata))), desc='   InChIKeys'):
        try:
            ik = msdata.get_values('INCHIKEY', i)
            if isinstance(ik, bytes): ik = ik.decode('utf-8')
            inchikey_list.append(str(ik).strip())
        except Exception:
            inchikey_list.append(f'error_{i}')
    inchikey_list.extend([''] * (len(msdata) - len(inchikey_list)))

    # Compute match vectors + Tanimoto labels
    print(f'\n[3] Computing match vectors and Tanimoto labels...')
    spec_preproc = du.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(), n_highest_peaks=128)

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    match_vecs_cache = {}
    n_use = 5000
    for i in range(n_use):
        try:
            spec = torch.as_tensor(msdata.get_spectra(i), dtype=torch.float32)
            spec_pp = spec_preproc(spec.numpy(), high_form=False)
            spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)
            mz = spec_t[:, 0].unsqueeze(0)
            pad = mz[:, 0] == 0
            mz_diffs = torch.abs(mz.unsqueeze(-1) - mz.unsqueeze(-2))
            vec = engine.get_rule_match_vectors(
                mz_diffs, mz_values=mz,
                precursor_mz=mz[:, 0].unsqueeze(0),
                padding_mask=pad, categories=['NL', 'CF', 'ISO', 'HR'])
            match_vecs_cache[i] = vec.squeeze(0)
        except Exception:
            match_vecs_cache[i] = torch.zeros(len(engine.rules))

    X_agg, X_meanpool, y = [], [], []
    valid_pairs = []

    for pi, (i, j) in enumerate(tqdm(pairs, desc='   Features')):
        if i not in match_vecs_cache or j not in match_vecs_cache:
            continue
        va, vb = match_vecs_cache[i], match_vecs_cache[j]
        inter = (va * vb).sum().float()
        union = ((va + vb) > 0).float().sum()
        ov = (inter / union.clamp(min=1)).item()

        # 共同命中规则
        common = (va * vb) > 0
        nc = common.sum().item()

        # ---- 聚合特征 (5维) ----
        nl = common[:214].sum().item()
        cf = common[214:214+102].sum().item()
        iso_hr = (common[214+102:214+102+8].sum() + common[-9:].sum()).item()
        l2_frac = iso_hr / max(nc, 1)
        nl_frac = nl / max(nc, 1)
        cf_frac = cf / max(nc, 1)
        X_agg.append([ov, float(nc), l2_frac, nl_frac, cf_frac])

        # ---- 实例特征平均池化 (12维) ----
        instance_feats = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category == 'HR': level = 2
                elif rule.category in ('NR', 'EE'): level = 0
                elif rule.category == 'ISO': level = 2
                instance_feats.append(build_instance_features({
                    'level': level, 'category': rule.category,
                    'match_type': rule.match_type,
                    'mass_diff_precision': 0.5,
                }))
        if len(instance_feats) > 0:
            meanpool = np.stack(instance_feats).mean(axis=0)
        else:
            meanpool = np.zeros(12, dtype=np.float32)
        X_meanpool.append(meanpool)

        # ---- Tanimoto label ----
        try:
            smi_a = msdata.get_values('smiles', int(i))
            smi_b = msdata.get_values('smiles', int(j))
            if isinstance(smi_a, bytes): smi_a = smi_a.decode('utf-8')
            if isinstance(smi_b, bytes): smi_b = smi_b.decode('utf-8')
            ma = Chem.MolFromSmiles(str(smi_a).strip())
            mb = Chem.MolFromSmiles(str(smi_b).strip())
            if ma and mb:
                fpa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
                fpb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
                tan = DataStructs.TanimotoSimilarity(fpa, fpb)
            else:
                tan = 0.0
        except Exception:
            tan = 0.0
        y.append(tan)
        valid_pairs.append(pi)

    X_agg = np.array(X_agg, dtype=np.float32)
    X_meanpool = np.array(X_meanpool, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f'   Valid pairs: {len(y)}')
    print(f'   Tanimoto: mean={y.mean():.4f}, std={y.std():.4f}, '
          f'min={y.min():.4f}, max={y.max():.4f}')

    # k-fold CV
    print(f'\n[4] {args.n_folds}-fold CV (molecule-level split)...')
    folds = build_kfold_splits(pairs, inchikey_list, n_folds=args.n_folds)

    results = defaultdict(list)
    for fold_idx, split in enumerate(folds):
        train_idx = [pi for pi in split['train'] if pi in valid_pairs]
        val_idx = [pi for pi in split['val'] if pi in valid_pairs]
        if len(train_idx) < 10 or len(val_idx) < 10:
            continue

        Xt_a, Xv_a = X_agg[train_idx], X_agg[val_idx]
        Xt_m, Xv_m = X_meanpool[train_idx], X_meanpool[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        for name, Xt, Xv in [('LR-agg', Xt_a, Xv_a), ('LR-meanpool', Xt_m, Xv_m)]:
            lr = Ridge(alpha=1.0)
            lr.fit(Xt, yt)
            yp = lr.predict(Xv)
            r, _ = pearsonr(yv, yp)
            mae = mean_absolute_error(yv, yp)
            results[name + '_r'].append(r)
            results[name + '_mae'].append(mae)
        print(f'   Fold {fold_idx+1}: LR-agg r={results["LR-agg_r"][-1]:.4f} '
              f'LR-meanpool r={results["LR-meanpool_r"][-1]:.4f}')

    print(f'\n{"="*60}')
    print(f'BASELINE RESULTS (Regression, Pearson r)')
    print(f'{"="*60}')
    for name in ['LR-agg', 'LR-meanpool']:
        rs = results[name + '_r']
        maes = results[name + '_mae']
        print(f'  {name:15s}: r = {np.mean(rs):.4f} +/- {np.std(rs):.4f}  '
              f'MAE = {np.mean(maes):.4f} +/- {np.std(maes):.4f}')
    print(f'\n  MIL must beat these numbers to demonstrate value.')


if __name__ == '__main__':
    main()
