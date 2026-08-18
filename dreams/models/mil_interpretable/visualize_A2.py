"""
visualize_A2.py — A0 vs A1 vs A2 vs LR-agg 可视化

读取 outputs/mil_A0vsA1vsA2_TIMESTAMP/training_logs.json 和 summary.json

用法: python dreams/models/mil_interpretable/visualize_A2.py <output_dir>
"""

import json, sys, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path('outputs').glob('mil_A0vsA1vsA2_*'))[-1]
    print(f'Reading: {out_dir}')

    with open(out_dir / 'training_logs.json') as f: logs = json.load(f)
    with open(out_dir / 'summary.json') as f: summary = json.load(f)
    with open(out_dir / 'config.json') as f: config = json.load(f)

    n_folds = summary['n_folds']
    models = ['A0', 'A1', 'A2']
    colors = {'A0': '#e74c3c', 'A1': '#3498db', 'A2': '#2ecc71'}

    # ===== Figure 1: Validation r curves (all folds) =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'MIL A0 vs A1 vs A2 — {n_folds}-fold CV', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for m in models:
        all_rs = []
        for k in range(n_folds):
            fold_key = f'fold_{k}_{m}'
            if fold_key in logs:
                epochs = [e['epoch'] for e in logs[fold_key]]
                rs = [e['val_r'] for e in logs[fold_key]]
                ax.plot(epochs, rs, color=colors[m], alpha=0.25, lw=0.5)
                all_rs.append(rs)
        # Mean curve
        if all_rs:
            min_len = min(len(r) for r in all_rs)
            stacked = np.array([r[:min_len] for r in all_rs])
            mean_r = stacked.mean(axis=0)
            std_r = stacked.std(axis=0)
            x = np.arange(min_len)
            ax.plot(x, mean_r, color=colors[m], lw=2.5, label=f'{m} (mean)')
            ax.fill_between(x, mean_r - std_r, mean_r + std_r, color=colors[m], alpha=0.1)
    # LR-agg baseline
    lr_r = float(summary['LR-agg'].split('+/-')[0])
    ax.axhline(lr_r, color='gray', ls='--', lw=1.5, label=f'LR-agg ({lr_r:.3f})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Pearson r')
    ax.set_title('Validation r Curves (all folds + mean)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ===== Figure 2: Best r fold-by-fold bar chart (ALL FOUR models) =====
    ax = axes[0, 1]
    all_models = ['LR-agg'] + models
    all_colors = {**colors, 'LR-agg': '#95a5a6'}
    x = np.arange(n_folds)
    w = 0.2
    lr_folds_list = json.load(open(out_dir / 'baseline_lr_results.json'))['r_folds']
    for i, m in enumerate(all_models):
        if m == 'LR-agg':
            fold_rs = lr_folds_list
        else:
            fold_rs = []
            for k in range(n_folds):
                fk = f'fold_{k}_{m}'
                fold_rs.append(max(e['val_r'] for e in logs[fk]) if fk in logs else 0)
        if len(fold_rs) == n_folds:
            ax.bar(x + i * w, fold_rs, w, color=all_colors[m], alpha=0.85, label=m)
    ax.set_xlabel('Fold'); ax.set_ylabel('Best Validation r')
    ax.set_title('Best r per Fold — All Models')
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels([str(k) for k in range(n_folds)])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # ===== Figure 3: Learning rate curves (A1 and A2) =====
    ax = axes[1, 0]
    for m in ['A1', 'A2']:
        fold_key = f'fold_0_{m}'
        if fold_key in logs:
            lrs = [e['lr'] for e in logs[fold_key]]
            epochs = [e['epoch'] for e in logs[fold_key]]
            ax.plot(epochs, lrs, color=colors[m], lw=1.5, label=m)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Fold 0)')
    ax.set_yscale('log')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ===== Figure 4: Train vs Val Loss (Fold 0, A2) =====
    ax = axes[1, 1]
    for m in models:
        fold_key = f'fold_0_{m}'
        if fold_key in logs:
            tl = [e['train_loss'] for e in logs[fold_key]]
            vl = [e['val_loss'] for e in logs[fold_key]]
            ep = [e['epoch'] for e in logs[fold_key]]
            ax.plot(ep, tl, color=colors[m], lw=0.8, alpha=0.5, ls='--')
            ax.plot(ep, vl, color=colors[m], lw=2, label=f'{m}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Curves (Fold 0, solid=val, dashed=train)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / 'comparison_plots.png'
    plt.savefig(fig_path, dpi=150)
    print(f'Saved: {fig_path}')

    # ===== Figure 5: A2 restart points =====
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fold_key = f'fold_0_A2'
    if fold_key in logs:
        rs = [e['val_r'] for e in logs[fold_key]]
        epochs = [e['epoch'] for e in logs[fold_key]]
        restarts = [e['epoch'] for e in logs[fold_key] if e.get('restart')]
        ax2.plot(epochs, rs, color=colors['A2'], lw=2, label='A2 val_r')
        for rx in restarts:
            ax2.axvline(rx, color='red', alpha=0.3, lw=0.8)
        ax2.axhline(lr_r, color='gray', ls='--', lw=1.5, label=f'LR-agg ({lr_r:.3f})')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation r')
        ax2.set_title(f'A2 AdaCosine — Restart Points (Fold 0, {len(restarts)} restarts)')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    fig_path2 = out_dir / 'a2_restarts.png'
    
    plt.savefig(fig_path2, dpi=150)
    print(f'Saved: {fig_path2}')

    # ===== Final summary table =====
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    for k, v in summary.items():
        if k != 'n_folds':
            print(f'  {k:12s}: r = {v}')

    # Print per-fold breakdown (all four)
    print(f'\nPer-fold best r (all models):')
    print(f'  {"Fold":6s} {"LR-agg":>8s} {"A0":>8s} {"A1":>8s} {"A2":>8s}')
    lr_folds = json.load(open(out_dir / 'baseline_lr_results.json'))['r_folds']
    for k in range(n_folds):
        a0_r = max(e['val_r'] for e in logs.get(f'fold_{k}_A0', [{'val_r':0}])) if f'fold_{k}_A0' in logs else 0
        a1_r = max(e['val_r'] for e in logs.get(f'fold_{k}_A1', [{'val_r':0}])) if f'fold_{k}_A1' in logs else 0
        a2_r = max(e['val_r'] for e in logs.get(f'fold_{k}_A2', [{'val_r':0}])) if f'fold_{k}_A2' in logs else 0
        print(f'  {k:4d}   {a0_r:8.4f} {a1_r:8.4f} {a2_r:8.4f} {lr_folds[k]:8.4f}')


if __name__ == '__main__':
    main()
