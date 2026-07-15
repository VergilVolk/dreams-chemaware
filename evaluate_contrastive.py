"""
对比学习 v4 检索评估 — DreaMS SpecRetrievalValidation 协议

比较三个模型在 MassSpecGym 上的谱图检索能力：
  1. 原版 DreaMS（零样本 baseline）
  2. 规则对比学习微调版（v4）
  3. （可选）纯 mask loss 微调版（对照）

协议（与 DreaMS 论文一致）：
  - 构建谱图对 (i, j)
  - 计算 cosine_sim(emb[i], emb[j])
  - label = 1 同分子（相同 SMILES/InChIKey）; 0 不同
  - 计算 ROC AUC + Top-K 准确率

用法：
  python evaluate_contrastive.py \
      --ckpt_path ./contrastive_checkpoints/contrastive_epoch10.pt \
      --n_spectra 3000 --n_pairs 30000

作者：module1-chem-attn 开发分支
"""

import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn import metrics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS
from dreams.definitions import PRETRAINED


def parse_args():
    p = argparse.ArgumentParser(description='Retrieval evaluation for contrastive v4')
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='Path to v4 contrastive checkpoint (.pt)')
    p.add_argument('--control_ckpt', type=str, default=None,
                   help='Path to mask-only control checkpoint (optional)')
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--base_ckpt', type=str,
                   default='dreams/models/pretrained/ssl_model_server.pt')
    p.add_argument('--n_spectra', type=int, default=3000,
                   help='Number of test spectra')
    p.add_argument('--n_pairs', type=int, default=30000,
                   help='Number of spectrum pairs for AUC')
    p.add_argument('--output_dir', type=str, default='./retrieval_eval')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ==============================================================================
# 模型加载
# ==============================================================================

def load_base_dreaMS(ckpt_path: str, device: torch.device) -> DreaMS:
    """从 ssl_model_server.pt 加载原版 DreaMS"""
    pkg = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
    return model


def load_finetuned_dreaMS(base_ckpt: str, ft_ckpt: str, device: torch.device) -> DreaMS:
    """加载微调后的 DreaMS（base 权重 + 微调 ckpt 覆盖）"""
    model = load_base_dreaMS(base_ckpt, device)

    if ft_ckpt and Path(ft_ckpt).exists():
        pkg = torch.load(ft_ckpt, map_location=device, weights_only=False)
        ft_state = pkg.get('model_state_dict', pkg)
        # 过滤不匹配的键
        model_state = model.state_dict()
        matched = 0
        for k in ft_state:
            if k in model_state and ft_state[k].shape == model_state[k].shape:
                model_state[k] = ft_state[k].clone()
                matched += 1
        model.load_state_dict(model_state, strict=False)
        print(f'   Loaded fine-tuned weights: {matched}/{len(model_state)} params matched')
    else:
        print(f'   WARNING: Fine-tuned checkpoint not found, using base weights')

    model.eval().to(device)
    return model


# ==============================================================================
# 嵌入提取
# ==============================================================================

def extract_embeddings(model, msdata, spec_preproc, device, indices):
    """提取指定谱图的 s_0（precursor）嵌入"""
    emb_list, labels, valid_indices = [], [], []

    for idx in tqdm(indices, desc='Extracting embeddings'):
        try:
            smiles = msdata.get_values('smiles', int(idx))
            if isinstance(smiles, bytes):
                smiles = smiles.decode('utf-8')
            smiles = str(smiles).strip()
            if len(smiles) < 2:
                continue
        except Exception:
            continue

        try:
            spec = torch.as_tensor(msdata.get_spectra(int(idx)), dtype=torch.float32)
            if spec.dim() != 2:
                continue
        except Exception:
            continue

        try:
            spec_pp = spec_preproc(spec.numpy(), high_form=False)
        except Exception:
            continue

        spec_t = torch.as_tensor(spec_pp, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            emb = model(spec_t, None)

        emb_list.append(emb[:, 0, :].cpu())  # s_0: precursor embedding
        labels.append(smiles)
        valid_indices.append(int(idx))

    return torch.cat(emb_list, dim=0), np.array(labels), np.array(valid_indices)


# ==============================================================================
# 谱图对构建（DreaMS 协议）
# ==============================================================================

def build_spectrum_pairs(labels, n_pairs, rng_seed=42):
    """
    构建平衡的谱图对：
      - 正样本：同分子的不同谱图
      - 负样本：不同分子的谱图
    """
    mol_to_indices = defaultdict(list)
    for i, smi in enumerate(labels):
        mol_to_indices[smi].append(i)

    multi = {k: v for k, v in mol_to_indices.items() if len(v) >= 2}
    all_mols = list(mol_to_indices.keys())
    rng = np.random.RandomState(rng_seed)

    n_pos = 0
    n_neg = 0
    target_pos = n_pairs // 2
    target_neg = n_pairs - target_pos
    pair_i, pair_j, pair_labels = [], [], []

    # 正样本对
    if multi:
        mol_list = list(multi.keys())
        while n_pos < target_pos:
            mol = mol_list[rng.randint(0, len(mol_list))]
            idxs = multi[mol]
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

    return (np.array(pair_i), np.array(pair_j), np.array(pair_labels))


# ==============================================================================
# 评估指标
# ==============================================================================

def compute_retrieval_metrics(embeddings, pair_i, pair_j, pair_labels):
    """计算 ROC AUC + Top-K 准确率"""
    emb_i = embeddings[pair_i]
    emb_j = embeddings[pair_j]

    cos_sims = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=-1).numpy()

    # ROC AUC
    fpr, tpr, _ = metrics.roc_curve(pair_labels, cos_sims)
    auc = metrics.auc(fpr, tpr)

    # Top-K 检索准确率
    # 对每个 query (pair_i)，在 gallery (所有不重复的嵌入) 中检索
    # 简化：取所有嵌入，对每个正样本对检查排名
    top1_correct = 0
    top5_correct = 0
    total_queries = 0

    all_embs = embeddings  # (N, d)
    for qi, qj, is_pos in zip(pair_i, pair_j, pair_labels):
        if not is_pos:
            continue
        total_queries += 1
        q_emb = embeddings[qi:qi+1]  # (1, d)
        all_sims = torch.nn.functional.cosine_similarity(q_emb, all_embs, dim=-1)
        # 排除自身
        all_sims[qi] = -1.0
        sorted_indices = all_sims.argsort(descending=True)
        # Top-1
        if (sorted_indices[:1] == qj).any():
            top1_correct += 1
        # Top-5
        if (sorted_indices[:5] == qj).any():
            top5_correct += 1

    top1_acc = top1_correct / max(total_queries, 1)
    top5_acc = top5_correct / max(total_queries, 1)

    return {'auc': auc, 'fpr': fpr, 'tpr': tpr,
            'top1': top1_acc, 'top5': top5_acc,
            'cos_sims': cos_sims, 'n_total': total_queries}


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('RETRIEVAL EVALUATION — Contrastive v4 vs Original DreaMS')
    print('Protocol: SpecRetrievalValidation (ROC AUC + Top-K)')
    print('=' * 60)
    print(f'Device: {device}')

    # ---- 加载模型 ----
    print('\n[1] Loading models...')
    model_orig = load_base_dreaMS(args.base_ckpt, device)
    print(f'   Original DreaMS loaded')

    # v4 模型：base + 微调权重
    if args.ckpt_path:
        model_v4 = load_finetuned_dreaMS(args.base_ckpt, args.ckpt_path, device)
        print(f'   v4 Contrastive loaded')
    else:
        model_v4 = None
        print(f'   Skipping v4 (no checkpoint)')

    # 对照模型（可选）
    if args.control_ckpt:
        model_ctrl = load_finetuned_dreaMS(args.base_ckpt, args.control_ckpt, device)
        print(f'   Control loaded')
    else:
        model_ctrl = None

    # ---- 加载数据 ----
    print(f'\n[2] Loading dataset: {args.dataset_path}')
    msdata = du.MSData.load(args.dataset_path)
    smiles_col = None
    for col in ['smiles', 'SMILES']:
        if col in msdata.columns():
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError('No SMILES column found')

    # 取测试子集（使用固定随机种子确保一致性）
    n_total = min(args.n_spectra, len(msdata))
    rng = np.random.RandomState(42)
    test_indices = rng.choice(len(msdata), n_total, replace=False)

    spec_preproc = du.SpectrumPreprocessor(
        dformat=model_orig.spec_preproc.dformat,
        n_highest_peaks=model_orig.spec_preproc.n_highest_peaks
    )

    # ---- 提取嵌入 ----
    print(f'\n[3] Extracting embeddings for {n_total} spectra...')
    emb_orig, labels, valid_idx = extract_embeddings(
        model_orig, msdata, spec_preproc, device, test_indices)
    n_valid = len(emb_orig)
    print(f'   Original: {n_valid} embeddings, {len(set(labels))} unique molecules')

    results = {}

    if model_v4 is not None:
        emb_v4, _, _ = extract_embeddings(
            model_v4, msdata, spec_preproc, device, valid_idx)
        results['v4'] = emb_v4
        print(f'   v4: {len(emb_v4)} embeddings')

    if model_ctrl is not None:
        emb_ctrl, _, _ = extract_embeddings(
            model_ctrl, msdata, spec_preproc, device, valid_idx)
        results['control'] = emb_ctrl
        print(f'   Control: {len(emb_ctrl)} embeddings')

    # ---- 构建谱图对 ----
    print(f'\n[4] Building {args.n_pairs} spectrum pairs...')
    pair_i, pair_j, pair_labels = build_spectrum_pairs(labels, args.n_pairs)
    n_pos = pair_labels.sum()
    n_neg = len(pair_labels) - n_pos
    print(f'   Pairs: {int(n_pos)} positive + {int(n_neg)} negative')

    # ---- 评估 ----
    print(f'\n[5] Evaluating...')
    metrics_orig = compute_retrieval_metrics(emb_orig, pair_i, pair_j, pair_labels)
    print(f'\n{"=" * 50}')
    print(f'  Original DreaMS (baseline):')
    print(f'    AUC:   {metrics_orig["auc"]:.4f}')
    print(f'    Top-1: {metrics_orig["top1"]:.4f}')
    print(f'    Top-5: {metrics_orig["top5"]:.4f}')

    metrics_v4 = None
    metrics_ctrl = None

    if 'v4' in results:
        metrics_v4 = compute_retrieval_metrics(results['v4'], pair_i, pair_j, pair_labels)
        delta_auc = metrics_v4['auc'] - metrics_orig['auc']
        delta_top1 = metrics_v4['top1'] - metrics_orig['top1']
        print(f'\n  v4 Contrastive:')
        print(f'    AUC:   {metrics_v4["auc"]:.4f}  (Δ={delta_auc:+.4f})')
        print(f'    Top-1: {metrics_v4["top1"]:.4f}  (Δ={delta_top1:+.4f})')
        print(f'    Top-5: {metrics_v4["top5"]:.4f}')

    if 'control' in results:
        metrics_ctrl = compute_retrieval_metrics(results['control'], pair_i, pair_j, pair_labels)
        print(f'\n  Control (mask-only):')
        print(f'    AUC:   {metrics_ctrl["auc"]:.4f}')
        print(f'    Top-1: {metrics_ctrl["top1"]:.4f}')
        print(f'    Top-5: {metrics_ctrl["top5"]:.4f}')
    print(f'{"=" * 50}')

    # ---- 保存结果 ----
    with open(output_dir / 'retrieval_results.txt', 'w') as f:
        f.write('RETRIEVAL EVALUATION RESULTS\n')
        f.write('Protocol: DreaMS SpecRetrievalValidation\n')
        f.write('=' * 50 + '\n')
        f.write(f'Spectra: {n_valid}, Pairs: {len(pair_labels)}\n\n')
        f.write(f'Original DreaMS:  AUC={metrics_orig["auc"]:.4f}  Top-1={metrics_orig["top1"]:.4f}  Top-5={metrics_orig["top5"]:.4f}\n')
        if metrics_v4:
            delta = metrics_v4['auc'] - metrics_orig['auc']
            f.write(f'v4 Contrastive:   AUC={metrics_v4["auc"]:.4f}  Top-1={metrics_v4["top1"]:.4f}  Top-5={metrics_v4["top5"]:.4f}  (ΔAUC={delta:+.4f})\n')
        if metrics_ctrl:
            f.write(f'Control:          AUC={metrics_ctrl["auc"]:.4f}  Top-1={metrics_ctrl["top1"]:.4f}  Top-5={metrics_ctrl["top5"]:.4f}\n')
    print(f'Saved: {output_dir / "retrieval_results.txt"}')

    # ---- ROC 曲线图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Spectrum Retrieval — Contrastive v4 vs Original DreaMS',
                 fontsize=14, fontweight='bold')

    # ROC
    ax = axes[0]
    ax.plot(metrics_orig['fpr'], metrics_orig['tpr'], color='#3498db', lw=2,
            label=f'Original DreaMS (AUC={metrics_orig["auc"]:.4f})')
    if metrics_v4:
        ax.plot(metrics_v4['fpr'], metrics_v4['tpr'], color='#e74c3c', lw=2,
                label=f'v4 Contrastive (AUC={metrics_v4["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves'); ax.legend(fontsize=10)

    # Cosine similarity distribution
    ax = axes[1]
    bins = np.linspace(-1, 1, 50)
    ax.hist(metrics_orig['cos_sims'][pair_labels == 1], bins=bins, alpha=0.3,
            color='#3498db', label='Orig: same mol')
    ax.hist(metrics_orig['cos_sims'][pair_labels == 0], bins=bins, alpha=0.3,
            color='#3498db', label='Orig: diff mol', hatch='//')
    if metrics_v4:
        ax.hist(metrics_v4['cos_sims'][pair_labels == 1], bins=bins, alpha=0.4,
                color='#e74c3c', label='v4: same mol')
        ax.hist(metrics_v4['cos_sims'][pair_labels == 0], bins=bins, alpha=0.4,
                color='#e74c3c', label='v4: diff mol', hatch='//')
    ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Frequency')
    ax.set_title('Embedding Similarity Distribution'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_curves.png', dpi=150)
    print(f'Saved: {output_dir / "retrieval_curves.png"}')


if __name__ == '__main__':
    main()
