"""
验证核心假设：规则重叠度是否与结构相似度（Tanimoto）相关？

如果相关性 < 0.2 → 规则作为结构代理信号站不住，需重构
如果相关性 0.2-0.4 → 勉强可用但噪声大，需降 triplet_weight
如果相关性 > 0.4 → 方向正确，继续调参优化

用法：
  python validate_rule_overlap.py --n_pairs 3000

作者：module1-chem-attn 开发分支
"""

import torch
import numpy as np
import argparse
import traceback
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--n_pairs', type=int, default=3000)
    p.add_argument('--output_dir', type=str, default='./validation')
    return p.parse_args()


def compute_tanimoto(smiles_a: str, smiles_b: str) -> float:
    """计算两个 SMILES 的 Morgan 指纹 Tanimoto 相似度"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return -1.0

        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except ImportError:
        return -1.0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('VALIDATION: Rule Overlap vs Structural Similarity')
    print('=' * 60)

    # ---- 加载数据 ----
    print(f'\n[1] Loading dataset...')
    msdata = du.MSData.load(args.dataset_path)

    # 找 SMILES 列
    smiles_col = None
    for col in ['smiles', 'SMILES']:
        if col in msdata.columns():
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError('No SMILES column')

    # ---- 预提取 SMILES ----
    print(f'[2] Extracting SMILES + spectra...')
    n_total = len(msdata)
    rng = np.random.RandomState(42)

    # 先收集所有 (SMILES, spectrum_index) 对
    records = []
    first_err = None
    for idx in tqdm(range(min(10000, n_total)), desc='Scanning'):
        try:
            smiles = msdata.get_values('smiles', int(idx))
            if isinstance(smiles, bytes):
                smiles = smiles.decode('utf-8')
            smiles = str(smiles).strip()
            if len(smiles) < 2:
                continue

            # spectrum 列
            spec_raw = msdata.get_values('spectrum', int(idx))
            if spec_raw is None:
                if first_err is None:
                    first_err = f'idx={idx}: spectrum is None'
                continue

            # 诊断前3条数据的格式
            if idx < 3:
                print(f'   [DEBUG idx={idx}] spec_raw type={type(spec_raw)}, '
                      f'len={len(spec_raw) if hasattr(spec_raw, "__len__") else "N/A"}')
                if hasattr(spec_raw, '__len__') and len(spec_raw) > 0:
                    first_elem = spec_raw[0]
                    print(f'   [DEBUG idx={idx}] first elem type={type(first_elem)}, '
                          f'value={str(first_elem)[:100]}')

            spec_arr = np.array(spec_raw, dtype=np.float32)
            # HDF5 存的是 (2, 128): 第0行=m/z, 第1行=intensity
            # 转置成 (n_peaks, 2) = [[mz, int], ...]
            if spec_arr.ndim == 2 and spec_arr.shape[0] == 2:
                spec_arr = spec_arr.T  # (128, 2)
            elif spec_arr.ndim == 2 and spec_arr.shape[1] == 2:
                pass  # 已经是 (n, 2)
            else:
                if first_err is None:
                    first_err = f'idx={idx}: unexpected shape {spec_arr.shape}'
                continue

            if spec_arr.shape[0] < 3:
                continue

            records.append({'idx': idx, 'smiles': smiles, 'spec': spec_arr})
        except Exception as e:
            if first_err is None:
                first_err = f'idx={idx}: {type(e).__name__}: {e}'
            continue

    print(f'   First error (if any): {first_err}')

    print(f'   Valid records: {len(records)}')
    unique_smiles = len(set(r['smiles'] for r in records))
    print(f'   Unique molecules: {unique_smiles}')

    # ---- 构建引擎 ----
    engine = ChemicalRuleEngine(tolerance=0.02)
    spec_preproc = du.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(), n_highest_peaks=128)

    # ---- 随机采样谱图对 ----
    print(f'\n[3] Sampling {args.n_pairs} spectrum pairs...')
    n_pairs = min(args.n_pairs, len(records) * (len(records) - 1) // 2)

    # 分层采样：确保覆盖同分子对（高 Tanimoto）和不同分子对
    # 按 SMILES 分组
    mol_groups = defaultdict(list)
    for i, r in enumerate(records):
        mol_groups[r['smiles']].append(i)

    multi_mols = {k: v for k, v in mol_groups.items() if len(v) >= 2}
    single_indices = [i for i, r in enumerate(records)
                      if len(mol_groups[r['smiles']]) == 1]

    n_same = min(n_pairs // 3, 500)  # 1/3 同分子对
    n_diff = n_pairs - n_same

    pair_indices = []

    # 同分子对
    if multi_mols:
        mol_list = list(multi_mols.keys())
        for _ in range(n_same):
            mol = mol_list[rng.randint(0, len(mol_list))]
            i, j = rng.choice(multi_mols[mol], 2, replace=False)
            pair_indices.append((i, j, 1))  # 1 = same molecule

    # 不同分子对
    all_indices = list(range(len(records)))
    for _ in range(n_diff):
        i = rng.choice(all_indices)
        j = rng.choice(all_indices)
        if i == j:
            continue
        if records[i]['smiles'] == records[j]['smiles']:
            continue
        pair_indices.append((i, j, 0))  # 0 = different molecule

    rng.shuffle(pair_indices)
    print(f'   Same mol pairs: {sum(1 for _,_,t in pair_indices if t==1)}')
    print(f'   Diff mol pairs: {sum(1 for _,_,t in pair_indices if t==0)}')

    # ---- 计算规则重叠度 + Tanimoto ----
    print(f'\n[4] Computing rule overlap & Tanimoto for {len(pair_indices)} pairs...')
    rule_overlaps = []
    tanimoto_sims = []
    same_mol_flags = []
    pair_details = []

    # 预处理所有谱图（直接取非零 m/z 值，不用 spec_preproc）
    print('   Computing rule match vectors...')
    match_vecs_cache = {}
    fail_count = 0
    for i, r in enumerate(tqdm(records, desc='Match vectors')):
        try:
            spec_arr = r['spec']  # (n_peaks, 2) [mz, intensity]
            # 去零填充
            valid = spec_arr[:, 0] > 0.1  # m/z > 0.1 才是真实峰
            spec_filt = spec_arr[valid]
            if len(spec_filt) < 3:
                fail_count += 1
                match_vecs_cache[i] = None
                continue

            mz = torch.as_tensor(spec_filt[:, 0], dtype=torch.float32).unsqueeze(0)
            n = mz.shape[1]
            pad = torch.zeros(1, n, dtype=torch.bool)
            mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(mz)
            vec = engine.get_rule_match_vectors(
                mz_diffs, mz_values=mz, precursor_mz=mz[:, 0],
                padding_mask=pad, categories=['NL', 'CF', 'ISO']
            )
            match_vecs_cache[i] = vec.squeeze(0)
        except Exception:
            match_vecs_cache[i] = None
            fail_count += 1

    if fail_count > 0:
        print(f'   Failed: {fail_count}/{len(records)} spectra')

    # 计算 pairwise
    valid_pairs = 0
    for i, j, is_same in tqdm(pair_indices, desc='Computing overlaps'):
        if match_vecs_cache[i] is None or match_vecs_cache[j] is None:
            continue

        # 规则重叠度
        overlap = ChemicalRuleEngine.compute_rule_overlap(
            match_vecs_cache[i], match_vecs_cache[j]
        ).item()

        # Tanimoto
        tanimoto = compute_tanimoto(records[i]['smiles'], records[j]['smiles'])
        if tanimoto < 0:
            continue  # RDKit 不可用或 SMILES 无效

        rule_overlaps.append(overlap)
        tanimoto_sims.append(tanimoto)
        same_mol_flags.append(is_same)
        pair_details.append({
            'smiles_a': records[i]['smiles'],
            'smiles_b': records[j]['smiles'],
            'overlap': overlap,
            'tanimoto': tanimoto,
            'same_mol': is_same,
        })
        valid_pairs += 1

    print(f'\n   Valid pairs: {valid_pairs} (rule overlap + Tanimoto both computed)')

    if valid_pairs < 30:
        print('   ERROR: Not enough valid pairs. Install rdkit or check data.')
        return

    rule_overlaps = np.array(rule_overlaps)
    tanimoto_sims = np.array(tanimoto_sims)
    same_mol_flags = np.array(same_mol_flags)

    # ---- 统计分析 ----
    print(f'\n[5] Results')
    print('=' * 60)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(rule_overlaps, tanimoto_sims)
    spearman_r, spearman_p = stats.spearmanr(rule_overlaps, tanimoto_sims)

    print(f'  Pearson r:  {pearson_r:.4f}  (p={pearson_p:.2e})')
    print(f'  Spearman r: {spearman_r:.4f}  (p={spearman_p:.2e})')

    # 分组统计
    # 按规则重叠度分组，看各组的平均 Tanimoto
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f'\n  Overlap bin → avg Tanimoto (sample count):')
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rule_overlaps >= lo) & (rule_overlaps < hi)
        if mask.sum() > 0:
            avg = tanimoto_sims[mask].mean()
            print(f'    [{lo:.1f}, {hi:.1f}): avg Tanimoto = {avg:.4f}  (n={mask.sum()})')

    # 同分子 vs 不同分子的分离度
    same_mask = same_mol_flags == 1
    diff_mask = same_mol_flags == 0
    print(f'\n  Same mol pairs:   overlap={rule_overlaps[same_mask].mean():.4f}, '
          f'tanimoto={tanimoto_sims[same_mask].mean():.4f} (n={same_mask.sum()})')
    print(f'  Diff mol pairs:   overlap={rule_overlaps[diff_mask].mean():.4f}, '
          f'tanimoto={tanimoto_sims[diff_mask].mean():.4f} (n={diff_mask.sum()})')

    # ---- 判定 ----
    print(f'\n[6] Verdict')
    if pearson_r > 0.4:
        verdict = 'PASS: Rule overlap correlates with structure. Direction is valid.'
    elif pearson_r > 0.2:
        verdict = 'MARGINAL: Weak correlation. Reduce triplet_weight to 0.05.'
    else:
        verdict = 'FAIL: No meaningful correlation. Core hypothesis rejected.'

    print(f'  {verdict}')

    # ---- 保存 ----
    with open(output_dir / 'validation_results.txt', 'w') as f:
        f.write('RULE OVERLAP VS STRUCTURAL SIMILARITY VALIDATION\n')
        f.write('=' * 50 + '\n')
        f.write(f'Valid pairs: {valid_pairs}\n')
        f.write(f'Pearson r:  {pearson_r:.4f}  p={pearson_p:.2e}\n')
        f.write(f'Spearman r: {spearman_r:.4f}  p={spearman_p:.2e}\n')
        f.write(f'\nVerdict: {verdict}\n')
        f.write(f'\nTop 10 high-overlap pairs:\n')
        sorted_by_ol = sorted(pair_details, key=lambda x: x['overlap'], reverse=True)
        for d in sorted_by_ol[:10]:
            f.write(f'  overlap={d["overlap"]:.3f} tanimoto={d["tanimoto"]:.3f} '
                    f'{d["smiles_a"][:40]} ↔ {d["smiles_b"][:40]}\n')
        f.write(f'\nTop 10 high-tanimoto pairs:\n')
        sorted_by_tan = sorted(pair_details, key=lambda x: x['tanimoto'], reverse=True)
        for d in sorted_by_tan[:10]:
            f.write(f'  tanimoto={d["tanimoto"]:.3f} overlap={d["overlap"]:.3f} '
                    f'{d["smiles_a"][:40]} ↔ {d["smiles_b"][:40]}\n')

    print(f'\nSaved: {output_dir / "validation_results.txt"}')

    # ---- 散点图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Rule Overlap vs Structural Similarity (Tanimoto)',
                 fontsize=14, fontweight='bold')

    # Scatter
    ax = axes[0]
    ax.scatter(rule_overlaps[diff_mask], tanimoto_sims[diff_mask],
               c='#3498db', alpha=0.3, s=10, label='Diff mol')
    ax.scatter(rule_overlaps[same_mask], tanimoto_sims[same_mask],
               c='#e74c3c', alpha=0.5, s=15, label='Same mol')
    # 趋势线
    z = np.polyfit(rule_overlaps, tanimoto_sims, 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5,
            label=f'r={pearson_r:.3f}')
    ax.set_xlabel('Rule Overlap (Jaccard)')
    ax.set_ylabel('Tanimoto Similarity')
    ax.set_title('Scatter Plot')
    ax.legend(fontsize=8)

    # Binned bar chart
    ax = axes[1]
    bin_centers = [(lo+hi)/2 for lo, hi in zip(bins[:-1], bins[1:])]
    bin_avgs = []
    bin_stds = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (rule_overlaps >= lo) & (rule_overlaps < hi)
        if mask.sum() > 0:
            bin_avgs.append(tanimoto_sims[mask].mean())
            bin_stds.append(tanimoto_sims[mask].std())
        else:
            bin_avgs.append(0)
            bin_stds.append(0)
    ax.bar(bin_centers, bin_avgs, width=0.15, color='#3498db', alpha=0.7,
           yerr=bin_stds, capsize=3)
    ax.set_xlabel('Rule Overlap Bin')
    ax.set_ylabel('Mean Tanimoto Similarity')
    ax.set_title('Binned: Overlap → Tanimoto')
    ax.axhline(y=tanimoto_sims.mean(), color='gray', linestyle='--', alpha=0.5,
               label=f'Mean Tanimoto={tanimoto_sims.mean():.3f}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_vs_tanimoto.png', dpi=150)
    print(f'Saved: {output_dir / "overlap_vs_tanimoto.png"}')


if __name__ == '__main__':
    main()
