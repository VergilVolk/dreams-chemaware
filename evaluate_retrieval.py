"""
谱图检索评测 — 严格遵循 DreaMS 原论文 SpecRetrievalValidation 协议

协议（来自 DreaMS dreams.py 和 data.py）：
  1. 构建谱图对 (i, j)
  2. 计算 cosine_sim(emb[i], emb[j])
  3. label = 1 如果同分子（相同 SMILES），否则 0
  4. 计算 ROC AUC — 衡量嵌入相似度区分同/不同分子的能力

优势（相比 Top-K 准确率）：
  - 不需要选 K 值，AUC 是全局排序指标
  - 与 DreaMS 原论文的验证方式完全一致，可直接对比
  - 利用了 MassSpecGym 的 SMILES 标注

用法：
  cd D:\DreaMS
  python evaluate_retrieval.py

输出：
  retrieval_results.txt  — AUC 对比表 + 谱图对统计
  retrieval_curves.png   — ROC 曲线并排对比

作者：module1-chem-attn 开发分支
"""

import torch, numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_aware_dreams import ChemAwareDreaMS
from dreams.definitions import PRETRAINED


# ==============================================================================
# 1. 加载模型
# ==============================================================================
def load_models(device):
    """都从 ssl_model_server.pt 构建，避免 PyTorch 2.6 weights_only 问题"""

    # 加载共享/‘权重和参数
    pkg = torch.load(PRETRAINED / 'ssl_model_server.pt', map_location='cpu',
                     weights_only=False)
    from argparse import Namespace
    recon_args = Namespace(**pkg['args'])
    recon_args.dformat = dformats.DataFormatA()
    for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n',
               'min_charge','max_charge','max_prec_mz','high_intensity_thld',
               'min_intensity_ampl','max_ms_level']:
        if da in pkg['args']:
            setattr(recon_args.dformat, da, pkg['args'][da])

    # 原版 DreaMS — chem_attn=False，纯 backbone
    print('Loading original DreaMS (chem_attn=False)...')
    recon_args.chem_attn = False
    sp = du.SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
    model_orig = DreaMS(recon_args, sp)
    # 只加载 backbone 的参数
    orig_state = model_orig.state_dict()
    for k in orig_state:
        if k in pkg['state_dict'] and orig_state[k].shape == pkg['state_dict'][k].shape:
            orig_state[k] = pkg['state_dict'][k].clone()
    model_orig.load_state_dict(orig_state, strict=False)
    model_orig.eval().to(device)

    # ChemAwareDreaMS — chem_attn=True
    print('Loading ChemAwareDreaMS (chem_attn=True)...')
    recon_args.chem_attn = True
    recon_args.chem_attn_attenuation = -0.12  # v2 trained: lambda converged from 2.0 to 0.12
    recon_args.chem_attn_tolerance = 0.02
    recon_args.chem_attn_entropy_w = 0.0
    sp2 = du.SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
    model_chem = ChemAwareDreaMS(recon_args, sp2)
    cs = model_chem.state_dict()
    for k in cs:
        if k in pkg['state_dict'] and cs[k].shape == pkg['state_dict'][k].shape:
            cs[k] = pkg['state_dict'][k].clone()
    model_chem.load_state_dict(cs, strict=False)
    model_chem.eval().to(device)

    if model_chem.chem_rule_engine is not None:
        lam = model_chem.chem_rule_engine._effective_attenuation()
        lam_val = abs(lam.item()) if hasattr(lam, 'item') else abs(lam)
        print(f'   Lambda: {lam_val:.4f} (v2 converged value)')

    return model_orig, model_chem


# ==============================================================================
# 2. 提取嵌入 + 构建标签（同 DreaMS 协议）
# ==============================================================================
def extract_and_build_pairs(model_orig, model_chem, device, n_spectra=2000, n_pairs=50000):
    """
    从 MassSpecGym 提取嵌入，然后构建谱图对用于 AUC 评测。
    """
    msdata = du.MSData.load('data/MassSpecGym_MurckoHist_split.hdf5')

    # 检查可用的标签列
    smiles_col = None
    for col in ['smiles', 'SMILES', 'inchikey', 'INCHIKEY']:
        if col in msdata.columns():
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError('No SMILES/InChI column found in MassSpecGym')

    # 取子集
    n_total = min(n_spectra, len(msdata))
    indices = np.random.RandomState(42).choice(len(msdata), n_total, replace=False)

    print(f'Extracting embeddings for {n_total} spectra...')
    spec_preproc = du.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(), n_highest_peaks=128)

    emb_orig_list, emb_chem_list = [], []
    labels = []

    for idx in tqdm(indices, desc='Embedding'):
        try:
            smiles = msdata.get_values(smiles_col, int(idx))
            if isinstance(smiles, bytes):
                smiles = smiles.decode('utf-8')
            smiles = str(smiles).strip()
            if len(smiles) < 2:
                continue
        except:
            continue

        spec = torch.as_tensor(msdata.get_spectra(int(idx)), dtype=torch.float32)
        if spec.dim() != 2:
            continue

        spec_np = spec.numpy()
        try:
            spec_pp_np = spec_preproc(spec_np, high_form=False)
        except:
            continue
        spec_pp = torch.as_tensor(spec_pp_np, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            eo = model_orig(spec_pp, None)
            ec = model_chem(spec_pp, None)

        emb_orig_list.append(eo[:, 0, :].cpu())
        emb_chem_list.append(ec[:, 0, :].cpu())
        labels.append(smiles)

    emb_orig = torch.cat(emb_orig_list, dim=0)
    emb_chem = torch.cat(emb_chem_list, dim=0)
    labels = np.array(labels)

    print(f'Extracted {len(emb_orig)} embeddings')
    print(f'Unique molecules: {len(set(labels))}')

    # ---- 构建谱图对 ----
    # 正样本：同分子的不同谱图
    # 负样本：不同分子的谱图
    # 构建平衡的谱图对集，数量 = n_pairs
    print(f'Building {n_pairs} spectrum pairs...')

    # 按 SMILES 分组
    mol_to_indices = defaultdict(list)
    for i, smi in enumerate(labels):
        mol_to_indices[smi].append(i)

    multi_spectra_mols = {k: v for k, v in mol_to_indices.items() if len(v) >= 2}
    print(f'Molecules with >=2 spectra: {len(multi_spectra_mols)}')

    all_mols = list(mol_to_indices.keys())
    pair_i, pair_j, pair_labels = [], [], []

    n_pos = 0
    n_neg = 0
    target_pos = n_pairs // 2
    target_neg = n_pairs - target_pos

    rng = np.random.RandomState(42)

    # 正样本对
    if multi_spectra_mols:
        mol_list = list(multi_spectra_mols.keys())
        while n_pos < target_pos:
            mol = mol_list[rng.randint(0, len(mol_list))]
            idxs = multi_spectra_mols[mol]
            i, j = rng.choice(idxs, 2, replace=False)
            pair_i.append(i); pair_j.append(j); pair_labels.append(1)
            n_pos += 1

    # 负样本对
    while n_neg < target_neg:
        m1, m2 = rng.choice(all_mols, 2, replace=False)
        if m1 == m2:
            continue
        i = rng.choice(mol_to_indices[m1])
        j = rng.choice(mol_to_indices[m2])
        pair_i.append(i); pair_j.append(j); pair_labels.append(0)
        n_neg += 1

    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    pair_labels = np.array(pair_labels)
    print(f'Pairs: {n_pos} positive + {n_neg} negative = {n_pos+n_neg} total')

    return emb_orig, emb_chem, pair_i, pair_j, pair_labels


# ==============================================================================
# 3. AUC 评测（与 DreaMS 原论文完全一致）
# ==============================================================================
def evaluate_auc(embeddings, pair_i, pair_j, pair_labels):
    """
    对给定嵌入矩阵，计算谱图对余弦相似度，输出 ROC AUC。
    """
    # 取出成对嵌入
    emb_i = embeddings[pair_i]  # (N_pairs, d_model)
    emb_j = embeddings[pair_j]

    # 余弦相似度
    cos_sims = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=-1).numpy()

    # ROC AUC
    fpr, tpr, thresholds = metrics.roc_curve(pair_labels, cos_sims)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr, cos_sims


# ==============================================================================
# 4. 主流程
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('=' * 60)
    print('SPECTRUM RETRIEVAL AUC — ChemAware vs Original DreaMS')
    print('Protocol: DreaMS SpecRetrievalValidation')
    print('=' * 60)

    # 加载
    model_orig, model_chem = load_models(device)

    # 提取 + 构建对
    emb_orig, emb_chem, pair_i, pair_j, pair_labels = extract_and_build_pairs(
        model_orig, model_chem, device, n_spectra=2000, n_pairs=20000)

    # 评测
    print('\nEvaluating...')
    auc_orig, fpr_orig, tpr_orig, sims_orig = evaluate_auc(emb_orig, pair_i, pair_j, pair_labels)
    auc_chem, fpr_chem, tpr_chem, sims_chem = evaluate_auc(emb_chem, pair_i, pair_j, pair_labels)

    # ---- 输出 ----
    delta = auc_chem - auc_orig
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  Original DreaMS  AUC: {auc_orig:.4f}')
    print(f'  ChemAware DreaMS  AUC: {auc_chem:.4f}')
    print(f'  Delta:                {delta:+.4f}  ({delta/auc_orig*100:+.2f}%)')
    print(f'  Pairs evaluated:      {len(pair_labels):,}')
    print(f'  Pos:Neg ratio:        {pair_labels.sum():.0f}:{(1-pair_labels).sum():.0f}')
    print('=' * 60)

    # 保存文本
    with open('retrieval_results.txt', 'w') as f:
        f.write('SPECTRUM RETRIEVAL AUC RESULTS\n')
        f.write('Protocol: DreaMS SpecRetrievalValidation\n')
        f.write('=' * 50 + '\n')
        f.write(f'Original DreaMS  AUC: {auc_orig:.4f}\n')
        f.write(f'ChemAware DreaMS  AUC: {auc_chem:.4f}\n')
        f.write(f'Delta:                {delta:+.4f}  ({delta/auc_orig*100:+.2f}%)\n')
        f.write(f'Pairs evaluated:      {len(pair_labels):,}\n')
    print('Saved: retrieval_results.txt')

    # ---- 画图：ROC 曲线并排 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Spectrum Retrieval AUC — ChemAware vs Original DreaMS\n'
                 '(Protocol: DreaMS SpecRetrievalValidation)',
                 fontsize=14, fontweight='bold')

    # (a) ROC 曲线
    ax = axes[0]
    ax.plot(fpr_orig, tpr_orig, color='#3498db', linewidth=2,
            label=f'Original DreaMS (AUC={auc_orig:.4f})')
    ax.plot(fpr_chem, tpr_chem, color='#e74c3c', linewidth=2,
            label=f'ChemAware DreaMS (AUC={auc_chem:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('(a) ROC Curves')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (b) 相似度分布
    ax = axes[1]
    sims_pos_orig = sims_orig[pair_labels == 1]
    sims_neg_orig = sims_orig[pair_labels == 0]
    sims_pos_chem = sims_chem[pair_labels == 1]
    sims_neg_chem = sims_chem[pair_labels == 0]

    bins = np.linspace(-1, 1, 50)
    ax.hist(sims_pos_orig, bins=bins, alpha=0.3, color='#3498db', label='Orig: Same mol')
    ax.hist(sims_neg_orig, bins=bins, alpha=0.3, color='#3498db', label='Orig: Diff mol', hatch='//')
    ax.hist(sims_pos_chem, bins=bins, alpha=0.4, color='#e74c3c', label='Chem: Same mol')
    ax.hist(sims_neg_chem, bins=bins, alpha=0.4, color='#e74c3c', label='Chem: Diff mol', hatch='//')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('(b) Embedding Similarity Distribution')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('retrieval_curves.png', dpi=150)
    print('Saved: retrieval_curves.png')


if __name__ == '__main__':
    main()
