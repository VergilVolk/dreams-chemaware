"""
DreaMS 基线复现 #1：零样本相似度（Zero-shot Similarity）
对应论文 Figure 4a

不训练，直接加载预训练权重，计算谱图嵌入的余弦相似度与分子 Tanimoto 相似度的 Pearson R。
论文报告值：DreaMS（零样本）≈ 0.70，MS2DeepScore ≈ 0.65。

输出：
  - Pearson R（散点图）
  - 论文图 4a 风格的可视化

用法：
  python baseline_zeroshot.py --n_spectra 2000 --n_pairs 3000

作者：module1-chem-attn 开发分支
"""

import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--ckpt_path', type=str,
                   default='dreams/models/pretrained/ssl_model_server.pt')
    p.add_argument('--n_spectra', type=int, default=2000)
    p.add_argument('--n_pairs', type=int, default=4000)
    p.add_argument('--output_dir', type=str, default='./baseline_results')
    return p.parse_args()


def compute_tanimoto(smiles_a: str, smiles_b: str) -> float:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('DreaMS BASELINE: Zero-shot Similarity (Figure 4a)')
    print('=' * 60)
    print(f'Device: {device}')

    # ---- 1. 加载模型 ----
    print('\n[1] Loading pretrained DreaMS...')
    pkg = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

    from argparse import Namespace
    recon_args = Namespace(**pkg['args'])
    recon_args.dformat = dformats.DataFormatA()
    for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
               'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
               'min_intensity_ampl', 'max_ms_level']:
        if da in pkg['args']:
            setattr(recon_args.dformat, da, pkg['args'][da])

    sp = du.SpectrumPreprocessor(dformat=recon_args.dformat,
                                 n_highest_peaks=recon_args.max_peaks_n)
    model = DreaMS(recon_args, sp)
    state = model.state_dict()
    for k in state:
        if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
            state[k] = pkg['state_dict'][k].clone()
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f'   Loaded {len(state)} params')

    # ---- 2. 加载数据 ----
    print(f'\n[2] Loading MassSpecGym...')
    msdata = du.MSData.load(args.dataset_path)
    n_total = len(msdata)
    rng = np.random.RandomState(42)

    # 收集 (smiles, spectrum_tensor) — 完全复刻 evaluate_contrastive.py 的读取方式
    records = []
    sample_indices = rng.choice(min(20000, n_total), min(20000, n_total), replace=False)
    for idx in tqdm(sample_indices, desc='Scanning data'):
        try:
            smiles = msdata.get_values('smiles', int(idx))
            if isinstance(smiles, bytes):
                smiles = smiles.decode('utf-8')
            smiles = str(smiles).strip()
            if len(smiles) < 2:
                continue

            # 复刻 evaluate_contrastive.py: get_spectra → torch tensor → numpy → preproc
            spec = torch.as_tensor(msdata.get_spectra(int(idx)), dtype=torch.float32)
            spec_np = spec.numpy()
            spec_pp = sp(spec_np, high_form=False)
            spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)

            records.append({'smiles': smiles, 'spec_t': spec_t})
        except Exception:
            continue

    # 取子集
    n_spectra = min(args.n_spectra, len(records))
    selected = rng.choice(len(records), n_spectra, replace=False)
    records = [records[i] for i in selected]
    print(f'   Selected: {len(records)} spectra, {len(set(r["smiles"] for r in records))} unique molecules')

    # ---- 3. 提取 DreaMS 嵌入 ----
    print(f'\n[3] Extracting DreaMS embeddings...')
    embeddings = []
    fail_count = 0
    for i, r in enumerate(tqdm(records, desc='Embedding')):
        try:
            spec_t = r['spec_t'].unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = model(spec_t, None)
            embeddings.append(emb[:, 0, :].cpu())  # s_0: precursor embedding
        except Exception as e:
            embeddings.append(None)
            fail_count += 1
            if fail_count <= 3:
                print(f'   [FAIL idx={i}] {type(e).__name__}: {e}')

    print(f'   Failed: {fail_count}/{len(records)}')

    valid_mask = [e is not None for e in embeddings]
    records = [r for r, v in zip(records, valid_mask) if v]
    embeddings = torch.cat([e for e in embeddings if e is not None], dim=0)
    print(f'   Valid embeddings: {len(embeddings)}')
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    # ---- 4. 采样谱图对 + 计算余弦相似度 + Tanimoto ----
    print(f'\n[4] Computing pairwise similarities for {args.n_pairs} pairs...')

    # 分层采样：同分子对 + 不同分子对
    mol_groups = defaultdict(list)
    for i, r in enumerate(records):
        mol_groups[r['smiles']].append(i)
    multi = {k: v for k, v in mol_groups.items() if len(v) >= 2}
    all_mols = list(mol_groups.keys())

    n_same = min(args.n_pairs // 3, 800)
    n_diff = args.n_pairs - n_same

    pair_i, pair_j = [], []

    if multi:
        mol_list = list(multi.keys())
        while len(pair_i) < n_same:
            mol = mol_list[rng.randint(0, len(mol_list))]
            a, b = rng.choice(multi[mol], 2, replace=False)
            pair_i.append(a); pair_j.append(b)

    while len(pair_i) < n_same + n_diff:
        m1, m2 = rng.choice(all_mols, 2, replace=False)
        if m1 == m2:
            continue
        a = rng.choice(mol_groups[m1])
        b = rng.choice(mol_groups[m2])
        pair_i.append(a); pair_j.append(b)

    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)

    # 批量计算余弦相似度
    cos_sims = torch.nn.functional.cosine_similarity(
        embeddings[pair_i], embeddings[pair_j], dim=-1
    ).numpy()

    # 批量计算 Tanimoto
    tanimoto_sims = []
    valid_pair_mask = []
    for pi, pj in tqdm(zip(pair_i, pair_j), desc='Tanimoto', total=len(pair_i)):
        t = compute_tanimoto(records[pi]['smiles'], records[pj]['smiles'])
        if t >= 0:
            tanimoto_sims.append(t)
            valid_pair_mask.append(True)
        else:
            tanimoto_sims.append(0.0)
            valid_pair_mask.append(False)

    valid_pair_mask = np.array(valid_pair_mask)
    cos_sims = cos_sims[valid_pair_mask]
    tanimoto_sims = np.array(tanimoto_sims)[valid_pair_mask]

    print(f'   Valid pairs (with Tanimoto): {len(cos_sims)}')

    # ---- 5. 统计 ----
    print(f'\n[5] Results')
    print('=' * 60)
    pearson_r, pearson_p = stats.pearsonr(cos_sims, tanimoto_sims)
    spearman_r, _ = stats.spearmanr(cos_sims, tanimoto_sims)
    print(f'  Pearson R:  {pearson_r:.4f}  (p={pearson_p:.2e})')
    print(f'  Spearman R: {spearman_r:.4f}')
    print(f'  Pairs evaluated: {len(cos_sims)}')
    print(f'  DreaMS (paper): ~0.70')
    print(f'  MS2DeepScore (paper): ~0.65')
    print(f'  Your result vs paper: {pearson_r/0.70*100:.1f}% of reported value')

    # ---- 6. 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('DreaMS Zero-shot Similarity (Reproducing Figure 4a)',
                 fontsize=14, fontweight='bold')

    # 散点图
    ax = axes[0]
    ax.scatter(cos_sims, tanimoto_sims, alpha=0.15, s=5, c='#3498db', edgecolors='none')
    z = np.polyfit(cos_sims, tanimoto_sims, 1)
    x_line = np.linspace(cos_sims.min(), cos_sims.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r-', lw=2, label=f'Pearson r={pearson_r:.3f}')
    ax.set_xlabel('DreaMS Cosine Similarity')
    ax.set_ylabel('Morgan Fingerprint Tanimoto')
    ax.set_title(f'Zero-shot Embedding Similarity\nvs Structural Similarity')
    ax.legend(fontsize=10)
    ax.text(0.05, 0.95, f'n={len(cos_sims)} pairs\nr={pearson_r:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 分箱图
    ax = axes[1]
    bins = np.linspace(cos_sims.min(), cos_sims.max(), 12)
    bin_centers = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    bin_avgs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (cos_sims >= lo) & (cos_sims < hi)
        bin_avgs.append(tanimoto_sims[mask].mean() if mask.sum() > 0 else 0)
    ax.bar(bin_centers, bin_avgs, width=(bins[1]-bins[0])*0.8,
           color='#3498db', alpha=0.7, edgecolor='#2980b9')
    ax.set_xlabel('DreaMS Cosine Similarity Bin')
    ax.set_ylabel('Mean Tanimoto Similarity')
    ax.set_title('Binned: DreaMS similarity → Molecular similarity')
    ax.axhline(y=tanimoto_sims.mean(), color='gray', linestyle='--',
               label=f'Mean Tanimoto={tanimoto_sims.mean():.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_zeroshot.png', dpi=150)
    print(f'\nSaved: {output_dir / "baseline_zeroshot.png"}')

    # 文本
    with open(output_dir / 'baseline_results.txt', 'w') as f:
        f.write('DreaMS Baseline Reproduction\n')
        f.write('=' * 40 + '\n')
        f.write(f'Zero-shot Pearson R:  {pearson_r:.4f}\n')
        f.write(f'Zero-shot Spearman R: {spearman_r:.4f}\n')
        f.write(f'Paper reported:       ~0.70\n')
        f.write(f'Pairs: {len(cos_sims)}\n')
        f.write(f'Spectra: {len(embeddings)}\n')
    print(f'Saved: {output_dir / "baseline_results.txt"}')


if __name__ == '__main__':
    main()
