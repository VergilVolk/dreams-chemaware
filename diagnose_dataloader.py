"""
P0 诊断脚本 v2 — 绕过 DreaMS 完整 import 链，仅依赖 h5py + torch + rdkit
用法：python diagnose_dataloader.py
"""
import h5py
import torch
import numpy as np
from collections import defaultdict
from scipy import stats
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

DATASET = 'data/MassSpecGym_MurckoHist_split.hdf5'


def compute_tanimoto(smiles_a, smiles_b):
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return -1.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return -1.0


def main():
    print("=" * 60)
    print("P0 诊断 v2: DataLoader 随机性 + 困难样本 Pearson r")
    print("=" * 60)

    # ---- 直接读 HDF5 ----
    print("\n[1] Loading HDF5 data...")
    f = h5py.File(DATASET, 'r')
    columns = list(f.keys())
    print(f"   Columns: {columns}")
    n_total = len(f[columns[0]])
    print(f"   Total rows: {n_total}")

    # 找关键列
    prec_col = None
    smiles_col = None
    spectrum_col = None
    fold_col = None
    for col in columns:
        col_lower = col.lower()
        if 'precursor' in col_lower and 'mz' in col_lower:
            prec_col = col
        if 'smiles' in col_lower:
            smiles_col = col
        if 'spectrum' in col_lower or 'spec' in col_lower:
            spectrum_col = col
        if 'fold' in col_lower or 'split' in col_lower:
            fold_col = col
    print(f"   prec_mz: {prec_col}, smiles: {smiles_col}, "
          f"spec: {spectrum_col}, fold: {fold_col}")

    # ---- 读 precursor m/z ----
    print("\n[2] Reading precursor m/z values...")
    if prec_col:
        prec_data = f[prec_col][:]
        prec_array = np.array([float(x) for x in prec_data])
    else:
        print("   ERROR: no precursor_mz column found!")
        f.close()
        return

    print(f"   Range: {prec_array.min():.1f} - {prec_array.max():.1f} Da")
    print(f"   Mean: {prec_array.mean():.1f}, Std: {prec_array.std():.1f}")

    # ---- 读 fold 分布 ----
    if fold_col:
        fold_data = f[fold_col][:]
        fold_counts = defaultdict(int)
        for x in fold_data[:]:
            if isinstance(x, bytes):
                fold_counts[x.decode('utf-8').strip().lower()] += 1
            else:
                fold_counts[str(x).strip().lower()] += 1
        print(f"   Fold distribution: {dict(fold_counts)}")

    # ---- 读 SMILES ----
    smiles_data = None
    if smiles_col:
        smiles_data = f[smiles_col][:]

    # =================================================================
    # 任务 1: 模拟随机 shuffle DataLoader 的 batch 质量分布
    # =================================================================
    print("\n" + "=" * 60)
    print("[TASK 1] 模拟 random-shuffle DataLoader batch 质量分布")
    print("=" * 60)

    rng = np.random.RandomState(42)
    batch_size = 64
    shuffled_idx = rng.permutation(len(prec_array))

    batch_stats = []
    for b in range(30):
        start = b * batch_size
        end = start + batch_size
        batch_prec = prec_array[shuffled_idx[start:end]]
        pmin = batch_prec.min()
        pmax = batch_prec.max()
        pstd = batch_prec.std()
        batch_stats.append((pmin, pmax, pstd))
        print(f"   Batch {b:3d}: min={pmin:8.1f}  max={pmax:8.1f}  "
              f"range={pmax-pmin:7.1f}  std={pstd:6.1f}")

    stds = [s[2] for s in batch_stats]
    ranges = [s[1]-s[0] for s in batch_stats]
    print(f"\n   Summary across 30 random batches:")
    print(f"   std:  min={min(stds):.1f}  mean={np.mean(stds):.1f}  max={max(stds):.1f}")
    print(f"   range: min={min(ranges):.1f}  mean={np.mean(ranges):.1f}  max={max(ranges):.1f}")

    if np.mean(stds) < 15:
        verdict1 = "FAIL — batches are mass-concentrated (DataLoader NOT random)"
    elif np.mean(stds) < 50:
        verdict1 = "WARNING — borderline"
    else:
        verdict1 = "PASS — DataLoader appears properly random"
    print(f"   >>> {verdict1}")

    # 额外检查：如果数据本身按 mass 排序了
    print(f"\n   Additional check: is raw data sorted by precursor mass?")
    diffs = np.diff(prec_array[:5000])
    same_sign = (diffs >= 0).mean()
    print(f"   Fraction of adjacent pairs with non-decreasing mass: {same_sign:.3f}")
    if same_sign > 0.95:
        print(f"   → WARNING: data appears sorted by mass in storage order")
    else:
        print(f"   → Data order appears random")

    # =================================================================
    # 任务 2: 困难样本子集 Pearson r
    # =================================================================
    print("\n" + "=" * 60)
    print("[TASK 2] Hard-pair subset: rule overlap vs Tanimoto Pearson r")
    print("=" * 60)

    if smiles_data is None or spectrum_col is None:
        print("   Cannot run: SMILES or spectrum data missing from HDF5")
        f.close()
        print_final_report(verdict1, None)
        return

    # 导入规则引擎 — 绕过 __init__.py 的完整 import 链
    print("   Importing ChemicalRuleEngine (bypassing DreaMS)...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "chem_rules", "dreams/models/chem_aware/chem_rules.py")
    chem_rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chem_rules)
    ChemicalRuleEngine = chem_rules.ChemicalRuleEngine
    engine = ChemicalRuleEngine(tolerance=0.02)

    # 采样 2000 张谱图
    n_sample = min(2000, n_total)
    sample_idx = rng.choice(n_total, n_sample, replace=False)

    records = []
    print(f"   Loading {n_sample} spectra + computing rule vectors...")
    for idx in tqdm(sample_idx):
        try:
            # SMILES
            smi = smiles_data[idx]
            if isinstance(smi, bytes): smi = smi.decode('utf-8')
            smi = str(smi).strip()
            if len(smi) < 2: continue

            # Spectrum
            raw_spec = f[spectrum_col][idx]  # shape (2, 128) or (128, 2)?
            spec_arr = np.array(raw_spec, dtype=np.float32)
            # HDF5 stores as (2, 128): row0=mz, row1=intensity → transpose to (n, 2)
            if spec_arr.ndim == 2 and spec_arr.shape[0] == 2:
                spec_arr = spec_arr.T  # (128, 2)
            elif spec_arr.ndim == 2 and spec_arr.shape[1] == 2:
                pass

            valid = spec_arr[:, 0] > 0.1
            spec_filt = spec_arr[valid]
            if len(spec_filt) < 3: continue

            # Rule vector
            mz = torch.as_tensor(spec_filt[:, 0], dtype=torch.float32).unsqueeze(0)
            n = mz.shape[1]
            pad = torch.zeros(1, n, dtype=torch.bool)
            mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(mz)
            prec_val = float(prec_array[idx])
            vec = engine.get_rule_match_vectors(
                mz_diffs, mz_values=mz, precursor_mz=torch.tensor([prec_val]),
                padding_mask=pad, categories=['NL', 'CF', 'ISO']
            )

            records.append({
                'smiles': smi,
                'prec_mz': prec_val,
                'match_vec': vec.squeeze(0),
            })
        except Exception as e:
            continue

    print(f"   Loaded {len(records)} valid spectra with rule vectors")

    if len(records) < 100:
        print("   ERROR: Not enough records. Check spectrum data format.")
        f.close()
        print_final_report(verdict1, None)
        return

    # 采样谱图对，按 mass diff 分层
    valid_idx = list(range(len(records)))
    pairs = {'hard': [], 'medium': [], 'easy': []}

    for _ in range(20000):
        i, j = rng.choice(valid_idx, 2, replace=False)
        mass_diff = abs(records[i]['prec_mz'] - records[j]['prec_mz'])
        if mass_diff <= 0.05:
            pairs['hard'].append((i, j))
        elif mass_diff <= 1.0:
            pairs['medium'].append((i, j))
        else:
            pairs['easy'].append((i, j))
        if all(len(v) >= 300 for v in pairs.values()):
            break

    for k, v in pairs.items():
        print(f"   {k}: {len(v)} pairs")

    # 计算 Pearson r
    print("\n   Computing correlations...")
    results = {}
    for category, pair_list in pairs.items():
        if len(pair_list) < 30: continue
        overlaps, tanimotos, mass_diffs = [], [], []
        for i, j in tqdm(pair_list, desc=f"   {category}"):
            ov = ChemicalRuleEngine.compute_rule_overlap(
                records[i]['match_vec'], records[j]['match_vec']
            ).item()
            tan = compute_tanimoto(records[i]['smiles'], records[j]['smiles'])
            if tan < 0: continue
            overlaps.append(ov)
            tanimotos.append(tan)
            mass_diffs.append(abs(records[i]['prec_mz'] - records[j]['prec_mz']))

        if len(overlaps) < 30: continue
        r, p = stats.pearsonr(overlaps, tanimotos)
        sr, sp = stats.spearmanr(overlaps, tanimotos)
        results[category] = {
            'n': len(overlaps), 'pearson_r': r, 'pearson_p': p,
            'spearman_r': sr, 'mean_mass_diff': np.mean(mass_diffs),
            'mean_overlap': np.mean(overlaps), 'mean_tanimoto': np.mean(tanimotos),
        }
        print(f"   {category}: n={len(overlaps)}, r={r:.4f} (p={p:.2e}), "
              f"spearman={sr:.4f}, mass_diff_mean={np.mean(mass_diffs):.3f} Da")

    f.close()
    print_final_report(verdict1, results)


def print_final_report(verdict1, results):
    print("\n" + "=" * 60)
    print("FINAL DIAGNOSIS")
    print("=" * 60)
    print(f"\n  Task 1 (DataLoader): {verdict1}")

    if results:
        print(f"\n  Task 2 (Hard-pair Pearson r):")
        for cat, res in results.items():
            print(f"    {cat:8s} (Δmass {res['mean_mass_diff']:.3f} Da): "
                  f"r={res['pearson_r']:.4f} (n={res['n']}), "
                  f"mean_overlap={res['mean_overlap']:.3f}, "
                  f"mean_tan={res['mean_tanimoto']:.3f}")

    # 核心判断
    if results:
        hard_r = results.get('hard', {}).get('pearson_r', None)
        medium_r = results.get('medium', {}).get('pearson_r', None)
        easy_r = results.get('easy', {}).get('pearson_r', None)

        if hard_r is not None:
            if hard_r > 0.3:
                v2 = f"hard r={hard_r:.3f} > 0.3 -> signal still valid in hard regime"
            elif hard_r > 0.15:
                v2 = f"hard r={hard_r:.3f}: 0.15-0.3 -> weak correlation"
            else:
                v2 = f"hard r={hard_r:.3f} < 0.15 -> signal ~ noise in hard regime"
        else:
            v2 = f"hard: insufficient pairs ({pairs['hard']} valid) -- hard pairs are naturally extremely rare"
        print(f"\n  >>> {v2}")

    # 决策
    print("\n  Decision:")
    dl_ok = 'PASS' in verdict1
    hard_r_val = results.get('hard', {}).get('pearson_r', 0) if results else 0

    if not dl_ok:
        print("  → DataLoader 有问题。修复后重跑所有实验。")
    elif hard_r_val > 0.3:
        print("  → 信号有效。可以尝试 Step 3 (低 α=0.05 + β=0.01)。")
    elif hard_r_val > 0.15:
        print("  → 弱相关。建议降 α 到 0.01，强 β=0.05，做最后一次尝试。")
    else:
        print("  → 困难场景信号为噪声。放弃对比学习，转向后验验证。")
    print("=" * 60)


if __name__ == '__main__':
    main()
