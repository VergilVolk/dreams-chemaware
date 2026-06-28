"""
注意力可视化 — 化学感知 vs 原版 DreaMS 并排对比

将模型在每个峰上的注意力权重叠加在 MS/MS 谱图上，同时标注已知中性丢失。
直接出论文配图：原版 vs 化学感知的注意力聚焦差异。

用法：
  cd D:\DreaMS
  python visualize_attention.py

输出：
  attention_vis.png — 每张谱图 4 面板（谱图+原版注意力+化学感知注意力+中性丢失匹配）

作者：module1-chem-attn 开发分支
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List

from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import (
    ChemicalRuleEngine, NEUTRAL_LOSSES as NL_DICT
)
from dreams.definitions import PRETRAINED
import dreams.utils.data as du


# ==============================================================================
# 核心可视化
# ==============================================================================

def visualize_spectrum_with_attention(
    spec_idx: int,
    mz_values: np.ndarray,           # (n_peaks,) 有效峰的 m/z
    intensities: np.ndarray,         # (n_peaks,) 有效峰的强度
    attn_orig: np.ndarray,           # (n_heads, n_peaks, n_peaks) 原版注意力
    attn_chem: np.ndarray,           # (n_heads, n_peaks, n_peaks) 化学感知注意力
    chem_bias: np.ndarray,           # (n_peaks, n_peaks) 化学偏置矩阵
    mz_diffs: np.ndarray,            # (n_peaks, n_peaks) 质量差矩阵
    neutral_losses: Dict[str, float],
    save_path: str = 'attention_vis.png'
):
    """
    4 面板图：谱图 + 原版注意力 + 化学感知注意力 + 中性丢失匹配
    """
    n_peaks = len(mz_values)
    n_heads = attn_orig.shape[0]

    # ---- 聚合注意力 ----
    # 每个峰被关注的总量 = 所有头 + 所有行对该列的注意力之和（排除自注意）
    def peak_attention(attn):
        """计算每个峰获得的注意力总权重"""
        a = attn.copy()  # (n_heads, n, n)
        for h in range(n_heads):
            np.fill_diagonal(a[h], 0.0)  # 排除自注
        # 平均：所有头 + 所有查询位置对该峰的关注
        return a.sum(axis=(0, 1))  # (n,) — 每个峰被关注的总分
        # 归一化到 [0, 1]
        raw = a.sum(axis=(0, 1))
        return raw / (raw.max() + 1e-8)

    peak_attn_orig = peak_attention(attn_orig)
    peak_attn_chem = peak_attention(attn_chem)

    # ---- 中性丢失匹配标注 ----
    loss_labels = []
    for i in range(n_peaks):
        for j in range(i + 1, n_peaks):
            diff = mz_diffs[i, j]
            for name, mass in neutral_losses.items():
                if abs(diff - mass) < 0.02:
                    loss_labels.append((i, j, name, diff))
    # 去重，优先标注母离子参与的碎裂
    loss_labels = sorted(loss_labels, key=lambda x: (x[0] != 0, -abs(x[3])))

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'Attention Visualization — Spectrum #{spec_idx} ({n_peaks} peaks, {n_heads} heads)',
        fontsize=14, fontweight='bold'
    )

    # ---- (a) MS/MS 谱图 ----
    ax = axes[0, 0]
    ax.vlines(mz_values, 0, intensities, colors='#2980b9', linewidth=1.5)
    ax.scatter(mz_values, intensities, s=40, c='#2980b9', zorder=5)
    ax.set_title('(a) MS/MS Spectrum')
    ax.set_xlabel('m/z (Da)')
    ax.set_ylabel('Relative Intensity')
    ax.set_xlim(mz_values.min() - 20, mz_values.max() + 20)
    ax.set_ylim(0, 1.15)

    # 标注 precursor
    ax.annotate('Precursor', xy=(mz_values[0], intensities[0]),
                xytext=(mz_values[0] + 30, intensities[0] + 0.05),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, fontweight='bold')

    # ---- (b) 原版 DreaMS 注意力 ----
    ax = axes[0, 1]
    colors_orig_norm = peak_attn_orig / (peak_attn_orig.max() + 1e-8)
    ax.vlines(mz_values, 0, intensities, colors='lightgray', linewidth=0.5)
    sizes = 20 + colors_orig_norm * 180  # [20, 200]
    sc = ax.scatter(mz_values, intensities, s=sizes,
                    c=colors_orig_norm, cmap='YlOrRd',
                    edgecolors='black', linewidth=0.5, zorder=5, vmin=0, vmax=1)
    ax.set_title(f'(b) Original DreaMS Attention')
    ax.set_xlabel('m/z (Da)')
    ax.set_ylabel('Relative Intensity')
    ax.set_xlim(mz_values.min() - 20, mz_values.max() + 20)
    ax.set_ylim(0, 1.15)
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Attn weight')

    # ---- (c) 化学感知注意力 ----
    ax = axes[1, 0]
    colors_chem_norm = peak_attn_chem / (peak_attn_chem.max() + 1e-8)
    ax.vlines(mz_values, 0, intensities, colors='lightgray', linewidth=0.5)
    sizes = 20 + colors_chem_norm * 180
    sc = ax.scatter(mz_values, intensities, s=sizes,
                    c=colors_chem_norm, cmap='YlOrRd',
                    edgecolors='black', linewidth=0.5, zorder=5, vmin=0, vmax=1)
    # 标注中性丢失匹配的峰
    loss_peaks = set()
    for i, j, name, _ in loss_labels[:8]:
        loss_peaks.add(i)
        loss_peaks.add(j)
    for p in loss_peaks:
        ax.annotate(f'P{p}', xy=(mz_values[p], intensities[p]),
                    xytext=(mz_values[p] + 15, intensities[p] + 0.03),
                    fontsize=7, color='red', fontweight='bold')
    ax.set_title(f'(c) Chem-Aware Attention')
    ax.set_xlabel('m/z (Da)')
    ax.set_ylabel('Relative Intensity')
    ax.set_xlim(mz_values.min() - 20, mz_values.max() + 20)
    ax.set_ylim(0, 1.15)
    plt.colorbar(sc, ax=ax, shrink=0.8, label='Attn weight')

    # ---- (d) 中性丢失匹配表 + 偏置热图 ----
    ax = axes[1, 1]
    ax.axis('off')

    # 上半：中性丢失匹配摘要
    text = 'Matched Neutral Losses:\n'
    text += '=' * 45 + '\n'
    shown = set()
    count = 0
    for i, j, name, diff in loss_labels:
        key = f'{i}->{j}'
        if key not in shown and count < 12:
            text += f'  Peak {i}→{j}  |  delta={diff:.2f} Da  |  {name}\n'
            shown.add(key)
            count += 1
    if count == 0:
        text += '  (no matches found in spectrum)\n'

    # 下半：化学偏置统计
    n_valid_pairs = n_peaks * n_peaks - n_peaks  # 排除对角线
    n_attenuated = int((chem_bias < -1).sum())
    n_free = int((chem_bias > -0.1).sum() - n_peaks)  # 排除 precursor 行/列
    text += f'\nChemical Bias Stats:\n'
    text += f'  Valid peak pairs: {n_valid_pairs}\n'
    text += f'  Free (bias=0):    {n_free} ({n_free/n_valid_pairs*100:.1f}%)\n'
    text += f'  Attenuated (bias<0): {n_attenuated} ({n_attenuated/n_valid_pairs*100:.1f}%)\n'

    # 注意力分配变化
    top_orig = np.argsort(peak_attn_orig)[-5:][::-1]
    top_chem = np.argsort(peak_attn_chem)[-5:][::-1]
    overlap = len(set(top_orig) & set(top_chem))
    text += f'\nTop-5 attention peaks:\n'
    text += f'  Original:   {list(top_orig)}\n'
    text += f'  Chem-Aware: {list(top_chem)}\n'
    text += f'  Overlap:    {overlap}/5\n'
    text += f'\n  => Chem-aware attention {"focuses on" if overlap < 3 else "mostly aligns with"} original top peaks'

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'   Figure saved: {save_path}')
    plt.close()


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    print('=' * 60)
    print('Attention Visualization — Chem-Aware vs Original')
    print('=' * 60)

    # ---- 加载模型 ----
    print('\n[1/3] Loading model...')
    model = DreaMS.load_from_checkpoint(
        PRETRAINED / 'ssl_model.ckpt', map_location=torch.device('cpu'))
    model.eval()
    print(f'   Model: {model.n_layers} layers, {model.n_heads} heads')

    # ---- 加载数据 ----
    print('[2/3] Loading spectra...')
    msdata = du.MSData.load(Path('data/examples/example_5_spectra.mgf'))
    spec_preproc = du.SpectrumPreprocessor(
        dformat=model.dformat, n_highest_peaks=model.spec_preproc.n_highest_peaks)
    dataset = msdata.to_torch_dataset(spec_preproc)
    from torch.utils.data import Subset
    dataset = Subset(dataset, list(range(min(3, len(dataset)))))

    # ---- 化学规则引擎 ----
    engine = ChemicalRuleEngine(attenuation=-2.0, tolerance=0.02)

    print(f'[3/3] Visualizing {len(dataset)} spectra...')
    print('=' * 60)

    for spec_idx in range(len(dataset)):
        sample = dataset[spec_idx]
        spec = torch.as_tensor(sample['spectrum'], dtype=torch.float32).unsqueeze(0)
        raw_mz = spec[0, :, 0].clone()
        n_valid = (raw_mz > 0).sum().item()
        spec_valid = spec[:, :n_valid, :]
        raw_mz_v = raw_mz[:n_valid]
        raw_intens = spec[0, :n_valid, 1].clone()

        # 提取原版注意力
        all_attns = []
        def make_hook(c):
            def h(m, i, o):
                if isinstance(o, tuple) and len(o) >= 2:
                    c.append(o[1].detach().cpu())
            return h
        hooks = [att.register_forward_hook(make_hook(all_attns))
                 for att in model.transformer_encoder.atts]
        with torch.inference_mode():
            _ = model(spec_valid)
        for h in hooks:
            h.remove()

        # 取最后一层注意力
        orig_attn = all_attns[-1]
        if orig_attn.dim() == 4:
            orig_attn = orig_attn[0]  # (n_heads, n, n)

        # 计算化学偏置 + 模拟化学感知注意力
        mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(raw_mz_v.unsqueeze(0))
        chem_bias = engine(mz_diffs, mz_values=raw_mz_v.unsqueeze(0))[0]  # (1, n, n) -> (n, n)

        # 注入 chem_bias 到原版注意力
        eps = 1e-8
        logits = torch.log(orig_attn + eps)              # (n_heads, n, n)
        chem_bias_2d = chem_bias.squeeze(0)              # (1, n, n) → (n, n)
        biased_logits = logits + chem_bias_2d            # (n_heads, n, n)
        chem_attn = torch.softmax(biased_logits, dim=-1).numpy()

        # 可视化
        visualize_spectrum_with_attention(
            spec_idx=spec_idx,
            mz_values=raw_mz_v.numpy(),
            intensities=raw_intens.numpy(),
            attn_orig=orig_attn.numpy(),
            attn_chem=chem_attn,
            chem_bias=chem_bias.numpy(),
            mz_diffs=mz_diffs[0].numpy(),
            neutral_losses=NL_DICT,
            save_path=f'attention_vis_spec{spec_idx}.png'
        )

        # 摘要
        top5_orig = np.argsort(orig_attn.numpy().sum(axis=(0,1)))[-5:][::-1]
        top5_chem = np.argsort(chem_attn.sum(axis=(0,1)))[-5:][::-1]
        print(f'\n   Spectrum #{spec_idx} ({n_valid} peaks):')
        print(f'   Top-5 original:   {list(top5_orig)}')
        print(f'   Top-5 chem-aware: {list(top5_chem)}')
        print(f'   Overlap: {len(set(top5_orig) & set(top5_chem))}/5')

    print('\nDone! Saved attention_vis_spec*.png')
    print('=' * 60)


if __name__ == '__main__':
    main()
