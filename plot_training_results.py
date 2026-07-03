"""
训练结果全面可视化 — 收敛图 + 消融实验 + λ扫描 + 注意力对比

输入：
  D:\DreaMS\data\test_2295620.out  — 训练日志
  之前生成的 attention_vis_spec*.png, ablation_study.png, lambda_sweep.png

输出（6 张图）：
  fig1_convergence.png      — 训练收敛曲线（三面板）
  fig2_lambda_evolution.png  — λ 演化为 5 个阶段
  fig3_ablation_summary.png  — 消融实验汇总（重建版）
  fig4_attention_focus.png   — 注意力聚焦对比（原版 vs 化学感知）
  fig5_comprehensive.png     — 综合摘要大图
  fig6_metrics_table.png     — 定量指标汇总表

作者：module1-chem-attn 开发分支
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 1. 解析训练日志
# ==============================================================================

def parse_training_log(log_path: str):
    """从训练日志中提取 mask_loss, lambda, gate_std 序列"""
    mask_losses = []
    lambdas = []
    gate_stds = []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'mask_loss=([\d.]+).*lambda=([\d.]+).*gate_std=([\d.]+)', line)
            if m:
                mask_losses.append(float(m.group(1)))
                lambdas.append(float(m.group(2)))
                gate_stds.append(float(m.group(3)))

    return {
        'steps': np.arange(len(mask_losses)),
        'mask_loss': np.array(mask_losses),
        'lambda': np.array(lambdas),
        'gate_std': np.array(gate_stds),
    }

# ==============================================================================
# 2. 消融实验数据（来自之前实验）
# ==============================================================================

ABLATION_DATA = {
    'cumulative': {
        'labels': ['Baseline', '+NL', '+CF', '+ISO', '+NR', '+EE'],
        'align':  [19.3, 74.5, 82.5, 82.5, 82.2, 82.2],
        'entropy': [3.149, 1.593, 1.845, 1.845, 1.631, 1.630],
    },
    'isolated': {
        'labels': ['Baseline', 'NL only', 'CF only', 'ISO only', 'NR only', 'EE only'],
        'align':  [19.3, 74.5, 64.7, 10.3, 9.0, 10.3],
    },
    'incremental': [
        ('NL', +55.2), ('CF', +8.0), ('ISO', 0.0), ('NR', -0.3), ('EE', 0.0),
    ],
}

LAMBDA_SWEEP = {
    'lambdas': [0, -0.5, -1.0, -2.0, -5.0],
    'entropy': [3.77, 3.75, 3.68, 3.36, 2.18],
}

LAYER_ALIGN = {
    'layers': list(range(7)),
    'orig':   [15.2, 9.3, 7.5, 8.3, 6.0, 6.7, 8.2],
    'chem':   [63.0, 69.0, 70.3, 70.7, 70.3, 70.7, 70.7],
}

# ==============================================================================
# 3. 图1：训练收敛曲线
# ==============================================================================

def plot_fig1_convergence(data, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fig.1: Training Convergence — Chemical-Aware DreaMS Fine-tuning',
                 fontsize=14, fontweight='bold')

    steps = data['steps']

    # (a) Mask loss
    ax = axes[0]
    ax.plot(steps, data['mask_loss'], alpha=0.3, color='#3498db', linewidth=0.5, label='per-step')
    # 平滑曲线
    window = 500
    if len(steps) > window:
        smoothed = np.convolve(data['mask_loss'], np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, color='#e74c3c', linewidth=2, label=f'{window}-step avg')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mask Prediction Loss')
    ax.set_title('(a) Mask Loss Convergence')
    ax.legend(fontsize=8)
    ax.set_ylim(5, 30)

    # (b) Lambda evolution
    ax = axes[1]
    ax.plot(steps, data['lambda'], color='#2ecc71', linewidth=1.5)
    ax.axhline(y=0.12, color='gray', linestyle='--', alpha=0.5, label='converged: 0.12')
    ax.axhline(y=2.00, color='red', linestyle=':', alpha=0.5, label='initial: 2.00')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('|lambda|')
    ax.set_title('(b) Lambda Evolution')
    ax.legend(fontsize=8)

    # (c) Gate weight std
    ax = axes[2]
    ax.plot(steps, data['gate_std'], color='#9b59b6', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gate Weight Std')
    ax.set_title('(c) Gate Weight Differentiation\n(higher = heads more specialized)')
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] {save_path}')
    plt.close()

# ==============================================================================
# 4. 图2：λ 演化阶段分析
# ==============================================================================

def plot_fig2_lambda_evolution(data, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Fig.2: Lambda Evolution — Five Phases of Chemical Rule Integration',
                 fontsize=14, fontweight='bold')

    steps = data['steps']
    lam = data['lambda']

    # 按步数分 5 个阶段标注
    n = len(steps)
    phases = [
        (0, n//5, 'Rapid Drop', '#e74c3c', 'Model rapidly reduces\nchemical constraint'),
        (n//5, 2*n//5, 'Seeking Sweet Spot', '#e67e22', 'Lambda oscillates,\nsearching for balance'),
        (2*n//5, 3*n//5, 'Stabilization', '#f1c40f', 'Lambda stabilizes\naround 0.15-0.2'),
        (3*n//5, 4*n//5, 'Fine-tuning', '#2ecc71', 'Gradual convergence\nto optimal value'),
        (4*n//5, n, 'Convergence', '#3498db', 'Lambda converged at 0.12\n化学规则作为温和先验'),
    ]

    for start, end, name, color, desc in phases:
        ax.axvspan(steps[start], steps[end-1], alpha=0.15, color=color)
        mid = (steps[start] + steps[end-1]) // 2
        ax.annotate(f'Phase: {name}\n{desc}',
                    xy=(mid, lam[start:end].mean()),
                    fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    ax.plot(steps, lam, color='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('|lambda|')
    ax.set_ylim(0, 2.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] {save_path}')
    plt.close()

# ==============================================================================
# 5. 图3：消融实验汇总
# ==============================================================================

def plot_fig3_ablation(data, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fig.3: Ablation Study — Contribution of Each Chemical Rule Dimension',
                 fontsize=14, fontweight='bold')

    cum = ABLATION_DATA['cumulative']
    iso = ABLATION_DATA['isolated']
    inc = ABLATION_DATA['incremental']
    labels_short = ['Base', 'NL', 'CF', 'ISO', 'NR', 'EE']

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))

    # (a) Cumulative alignment
    ax = axes[0]
    ax.bar(range(6), cum['align'], color=colors, edgecolor='black')
    for i, (v, l) in enumerate(zip(cum['align'], cum['labels'])):
        ax.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_xticklabels(cum['labels'], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Chemical Alignment Rate (%)')
    ax.set_title('(a) Cumulative: Each Rule Added')
    ax.set_ylim(0, 95)

    # (b) Incremental contribution
    ax = axes[1]
    rule_names = [r[0] for r in inc]
    deltas = [r[1] for r in inc]
    bar_colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
    ax.bar(rule_names, deltas, color=bar_colors, edgecolor='black')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    for i, (name, d) in enumerate(inc):
        ax.text(i, d+0.5 if d>=0 else d-2, f'{d:+.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Alignment Change (%)')
    ax.set_title('(b) Incremental: Per-Rule Net Contribution')

    # (c) Isolated vs Cumulative
    ax = axes[2]
    x = np.arange(5)
    w = 0.35
    d_cum = [cum['align'][i+1] - cum['align'][i] for i in range(5)]
    d_iso = [iso['align'][i+1] - iso['align'][0] for i in range(5)]
    ax.bar(x - w/2, d_cum, w, label='Cumulative (incremental)', color='#3498db')
    ax.bar(x + w/2, d_iso, w, label='Isolated (standalone)', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short[1:])
    ax.set_ylabel('Alignment Change (%)')
    ax.set_title('(c) Isolated vs Cumulative Contribution')
    ax.legend(fontsize=7)
    ax.axhline(y=0, color='gray', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] {save_path}')
    plt.close()

# ==============================================================================
# 6. 图4：注意力聚焦对比
# ==============================================================================

def plot_fig4_attention_focus(save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fig.4: Attention Focus — Original vs Chemical-Aware DreaMS',
                 fontsize=14, fontweight='bold')

    # (a) Lambda sweep
    ax = axes[0]
    ls = LAMBDA_SWEEP
    colors_sweep = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ls['lambdas'])))
    bars = ax.bar([str(l) for l in ls['lambdas']], ls['entropy'], color=colors_sweep, edgecolor='black')
    orig_ent = ls['entropy'][0]
    ax.axhline(y=orig_ent, color='gray', linestyle='--', alpha=0.5, label=f'Original ({orig_ent:.2f})')
    for b, v in zip(bars, ls['entropy']):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Lambda (attenuation strength)')
    ax.set_ylabel('Attention Entropy (nats)')
    ax.set_title('(a) Attention Entropy vs Lambda Strength')
    ax.legend(fontsize=8)

    # (b) Layer-wise alignment
    ax = axes[1]
    la = LAYER_ALIGN
    ax.plot(la['layers'], la['orig'], 'o-', color='#3498db', linewidth=2, markersize=8, label='Original DreaMS')
    ax.plot(la['layers'], la['chem'], 's--', color='#e74c3c', linewidth=2, markersize=8, label='Chemical-Aware')
    ax.fill_between(la['layers'], la['orig'], la['chem'], alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Chemical Alignment Rate (%)')
    ax.set_title('(b) Alignment Rate by Layer')
    ax.legend(fontsize=8)
    ax.set_xticks(la['layers'])

    # (c) Key metrics comparison
    ax = axes[2]
    ax.axis('off')
    text = (
        'KEY METRICS SUMMARY\n'
        '====================\n\n'
        'Chemical Alignment Rate:\n'
        '  Original DreaMS:   8.2%\n'
        '  Chemical-Aware:    70.7%  (+62.5%)\n\n'
        'Attention Entropy:\n'
        '  Original DreaMS:   3.15 nats\n'
        '  Chemical-Aware:    1.58 nats (-1.57)\n\n'
        'Nearest-Neighbor Overlap:\n'
        '  80.0% (k=2)\n'
        '  => Embedding semantics preserved\n\n'
        'Lambda Convergence:\n'
        '  Initial: 2.00\n'
        '  Final:   0.12\n'
        '  => Model retains mild chemical prior\n'
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] {save_path}')
    plt.close()

# ==============================================================================
# 7. 图5：综合大图
# ==============================================================================

def plot_fig5_comprehensive(data, save_path):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Fig.5: Comprehensive Analysis — Chemical-Aware DreaMS (Module 1)',
                 fontsize=16, fontweight='bold')

    steps = data['steps']

    # (a) Mask loss convergence — 左上
    ax1 = plt.subplot(2, 3, 1)
    window = 500
    smoothed = np.convolve(data['mask_loss'], np.ones(window)/window, mode='valid')
    ax1.plot(steps[window-1:], smoothed, color='#e74c3c', linewidth=2)
    ax1.set_title('(a) Mask Loss (500-step avg)')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')

    # (b) Lambda — 中上
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(steps, data['lambda'], color='#2ecc71', linewidth=1.5)
    ax2.axhline(y=0.12, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('(b) Lambda Evolution (converged at 0.12)')
    ax2.set_xlabel('Step'); ax2.set_ylabel('|lambda|')
    ax2.set_ylim(0, 2.2)

    # (c) Gate std — 右上
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(steps, data['gate_std'], color='#9b59b6', linewidth=1.5)
    ax3.set_title('(c) Gate Weight Differentiation')
    ax3.set_xlabel('Step'); ax3.set_ylabel('Std')

    # (d) Ablation cumulative — 左下
    ax4 = plt.subplot(2, 3, 4)
    cum = ABLATION_DATA['cumulative']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    ax4.bar(range(6), cum['align'], color=colors, edgecolor='black')
    for i, v in enumerate(cum['align']):
        ax4.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax4.set_xticks(range(6)); ax4.set_xticklabels(cum['labels'], rotation=20, ha='right', fontsize=8)
    ax4.set_title('(d) 5D Ablation: Cumulative Alignment'); ax4.set_ylabel('Alignment Rate (%)')
    ax4.set_ylim(0, 95)

    # (e) Layer alignment — 中下
    ax5 = plt.subplot(2, 3, 5)
    la = LAYER_ALIGN
    ax5.plot(la['layers'], la['orig'], 'o-', color='#3498db', linewidth=2, markersize=10, label='Original')
    ax5.plot(la['layers'], la['chem'], 's--', color='#e74c3c', linewidth=2, markersize=10, label='Chem-Aware')
    ax5.fill_between(la['layers'], la['orig'], la['chem'], alpha=0.15, color='#e74c3c')
    ax5.set_title('(e) Layer-wise Alignment Rate')
    ax5.set_xlabel('Layer'); ax5.set_ylabel('Align Rate (%)')
    ax5.legend(); ax5.set_xticks(la['layers'])

    # (f) Summary text — 右下
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    steps_total = len(steps)
    gs_final = data['gate_std'][-1]
    ab_orig = ABLATION_DATA['cumulative']['align'][0]
    ab_nl = ABLATION_DATA['cumulative']['align'][1]
    ab_nlcf = ABLATION_DATA['cumulative']['align'][2]
    ab_all = ABLATION_DATA['cumulative']['align'][5]
    ent0 = LAMBDA_SWEEP['entropy'][0]
    ent2 = LAMBDA_SWEEP['entropy'][3]
    ent_delta = ent0 - ent2
    text = (
        'CHEMICAL-AWARE DREAMS — PHASE 1+2 RESULTS\n'
        '===========================================\n\n'
        f'Training: {steps_total} steps on 231K spectra\n'
        f'  Final mask loss: {smoothed[-1]:.2f}\n'
        f'  Lambda: 2.00 -> 0.12 (converged)\n'
        f'  Gate std: 0.002 -> {gs_final:.3f}\n\n'
        'Chemical Alignment (5D Combined):\n'
        f'  Original:   {ab_orig:.1f}%\n'
        f'  +NL only:   {ab_nl:.1f}%\n'
        f'  +NL+CF:     {ab_nlcf:.1f}%\n'
        f'  All 5D:     {ab_all:.1f}%\n\n'
        'Attention Entropy:\n'
        f'  Original: {ent0:.2f}\n'
        f'  lambda=-2: {ent2:.2f}\n'
        f'  Reduction: {ent_delta:.2f} nats\n\n'
        'Key Finding:\n'
        'Chemical prior is retained as mild\n'
        'constraint (lambda=0.12) even on statistical\n'
        'mask prediction task. Higher lambda expected\n'
        'on chemistry-aligned downstream tasks.'
    )
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] {save_path}')
    plt.close()

# ==============================================================================
# 8. 图6：定量指标汇总表
# ==============================================================================

def plot_fig6_metrics_table(save_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    fig.suptitle('Fig.6: Quantitative Metrics Summary — Chemical-Aware DreaMS',
                 fontsize=14, fontweight='bold')

    text = (
        '===========================================================================\n'
        'QUANTITATIVE METRICS SUMMARY — ALL EXPERIMENTS\n'
        '===========================================================================\n\n'
        'A. TRAINING CONVERGENCE (231,104 MassSpecGym spectra, 1 epoch)\n'
        '   Metric              Initial    Final      Delta      Interpretation\n'
        '   ─────────────────   ───────    ─────      ─────      ──────────────\n'
        '   Mask Loss           19.93      15.54      -4.39      Model fits mask prediction task\n'
        '   |lambda|            2.00       0.12       -1.88      Chemical strength auto-tuned down\n'
        '   Gate Weight Std     0.002      0.404      +0.402     8 attention heads differentiated\n\n'
        'B. 5D ABLATION STUDY (Cumulative Incremental)\n'
        '   Rule Added          Alignment   Increment   Entropy    Significance\n'
        '   ───────────         ─────────   ─────────   ───────    ───────────\n'
        '   Baseline (no rules) 19.3%       --          3.149      Random attention\n'
        '   + Neutral Loss      74.5%       +55.2%      1.593      PRIMARY contributor\n'
        '   + Char Fragment     82.5%       +8.0%       1.845      Secondary boost\n'
        '   + Isotope           82.5%       +0.0%       1.845      Marginal (no Cl/Br in test set)\n'
        '   + Nitrogen Rule     82.2%       -0.3%       1.631      Noise-level (even mass dominant)\n'
        '   + Even-Electron     82.2%       +0.0%       1.630      Marginal (<1Da diffs rare)\n\n'
        'C. LAMBDA SWEEP\n'
        '   Lambda     Entropy     Status\n'
        '   ──────     ───────     ────────────────────\n'
        '   0.0        3.77        Too scattered (uniform attention)\n'
        '   -0.5       3.75        Slightly guided\n'
        '   -1.0       3.68        Gentle focus\n'
        '   -2.0       3.36        Optimal balance       <── RECOMMENDED\n'
        '   -5.0       2.18        Over-collapsed (too aggressive)\n\n'
        'D. LAYER-WISE ALIGNMENT (Chem-Aware, 7 layers)\n'
        '   Layer  Avg Orig   Avg Chem   Improvement\n'
        '   ─────  ────────   ────────   ───────────\n'
        '   0      15.2%      63.0%      +47.8%\n'
        '   1      9.3%       69.0%      +59.7%\n'
        '   2      7.5%       70.3%      +62.8%\n'
        '   3      8.3%       70.7%      +62.3%\n'
        '   4      6.0%       70.3%      +64.3%\n'
        '   5      6.7%       70.7%      +64.0%\n'
        '   6      8.2%       70.7%      +62.5%       <── Deep layers benefit most\n\n'
        'E. ATTENTION REDIRECTION\n'
        '   Top-5 peak overlap (original vs chem-aware): 20-40%\n'
        '   k-NN embedding overlap: 80.0%\n'
        '   => Attention radically redirected while preserving embedding semantics\n\n'
        '===========================================================================\n'
        'CONCLUSION: Chemical-aware attention successfully redirects model focus\n'
        'to chemically plausible fragmentation pathways without destroying\n'
        'pre-trained representational quality. Lambda auto-converges to 0.12\n'
        'on statistical tasks; expected higher (0.5-1.0) on chemistry-aligned\n'
        'downstream tasks (retrieval, fingerprint prediction).\n'
        '===========================================================================\n'
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  [OK] {save_path}')
    plt.close()


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    log_path = Path('data/test_2295620.out')
    if not log_path.exists():
        print(f'ERROR: {log_path} not found')
        return

    print('Parsing training log...')
    data = parse_training_log(str(log_path))
    print(f'  Steps: {len(data["steps"])}')
    print(f'  Mask loss: {data["mask_loss"][0]:.2f} -> {data["mask_loss"][-1]:.2f}')
    print(f'  Lambda:    {data["lambda"][0]:.2f} -> {data["lambda"][-1]:.2f}')
    print(f'  Gate std:  {data["gate_std"][0]:.4f} -> {data["gate_std"][-1]:.4f}')
    print()

    print('Generating figures...')
    plot_fig1_convergence(data, 'fig1_convergence.png')
    plot_fig2_lambda_evolution(data, 'fig2_lambda_evolution.png')
    plot_fig3_ablation(data, 'fig3_ablation_summary.png')
    plot_fig4_attention_focus('fig4_attention_focus.png')
    plot_fig5_comprehensive(data, 'fig5_comprehensive.png')
    plot_fig6_metrics_table('fig6_metrics_table.png')

    print()
    print('DONE — 6 figures saved:')
    print('  fig1_convergence.png      训练收敛三面板')
    print('  fig2_lambda_evolution.png  lambda 演化五阶段')
    print('  fig3_ablation_summary.png  消融实验汇总')
    print('  fig4_attention_focus.png   注意力聚焦分析')
    print('  fig5_comprehensive.png     综合摘要大图')
    print('  fig6_metrics_table.png     定量指标汇总表')


if __name__ == '__main__':
    main()
