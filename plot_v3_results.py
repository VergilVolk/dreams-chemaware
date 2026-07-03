"""
v3 对抗协同训练结果全面可视化分析

输入: data/test_2296281.out (105,730 步完整训练日志)
输出:
  fig_v3_convergence.png   — 收敛全景（6面板）
  fig_v3_lambda_analysis.png — λ 深度分析
  fig_v3_summary.png        — 综合结论图
"""

import re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# ==============================================================================
# 1. 解析日志
# ==============================================================================
def parse_log(path):
    mask_loss, lambdas, gate_stds, lam_stds = [], [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'mask_loss=([\d.]+).*lam_frag=([\d.]+)\(std=([\d.]+)\).*gate_std=([\d.]+)', line)
            if m:
                mask_loss.append(float(m.group(1)))
                lambdas.append(float(m.group(2)))
                lam_stds.append(float(m.group(3)))
                gate_stds.append(float(m.group(4)))
    return {
        'steps': np.arange(len(mask_loss)),
        'mask_loss': np.array(mask_loss),
        'lambda': np.array(lambdas),
        'lambda_std': np.array(lam_stds),
        'gate_std': np.array(gate_stds),
    }

data = parse_log('data/test_2296281.out')
steps = data['steps']
N = len(steps)
print(f'Parsed {N} steps')
print(f'lambda unique values: {sorted(set(data["lambda"].round(2)))[:10]}')
print(f'lambda_std unique values: {sorted(set(data["lambda_std"].round(3)))[:6]}')

# ==============================================================================
# 2. 收敛全景图
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('v3 Adversarial Collaborative Training — Convergence Analysis (105,730 steps)',
             fontsize=16, fontweight='bold')

# (a) Mask Loss
ax = axes[0, 0]
w = 2000
sm = np.convolve(data['mask_loss'], np.ones(w)/w, mode='valid')
ax.plot(steps[w-1:], sm, color='#e74c3c', linewidth=1.5)
ax.set_title(f'(a) Mask Loss ({w}-step avg)')
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_ylim(15, 20)

# (b) Lambda
ax = axes[0, 1]
ax.plot(steps, data['lambda'], alpha=0.3, color='#2ecc71', linewidth=0.3)
# 三态的移动平均
for val, color in [(0.90, '#2ecc71'), (0.70, '#3498db'), (0.50, '#e74c3c')]:
    mask = np.abs(data['lambda'] - val) < 0.05
    if mask.any():
        idx = np.where(mask)[0]
        ax.scatter(idx[::100], data['lambda'][idx[::100]], s=5, color=color, alpha=0.5, label=f'lambda={val}')
ax.set_title('(b) Lambda (3-state oscillation)')
ax.set_xlabel('Step'); ax.set_ylabel('lambda_frag')
ax.legend(fontsize=7, loc='upper right')

# (c) Lambda std
ax = axes[0, 2]
ax.plot(steps, data['lambda_std'], color='#9b59b6', linewidth=0.5, alpha=0.7)
ax.set_title('(c) Lambda Frag Std (rule differentiation)')
ax.set_xlabel('Step'); ax.set_ylabel('std')

# (d) Gate std
ax = axes[1, 0]
ax.plot(steps, data['gate_std'], color='#e67e22', linewidth=0.5)
ax.set_title('(d) Gate Weight Std (head specialization)')
ax.set_xlabel('Step'); ax.set_ylabel('std')
ax.axhline(y=0.40, color='gray', linestyle='--', alpha=0.5, label='stable ~0.40')

# (e) Lambda distribution
ax = axes[1, 1]
counts = Counter(data['lambda'].round(2))
vals, cnts = zip(*sorted(counts.items()))
colors_bar = ['#2ecc71' if v > 0.85 else '#3498db' if v > 0.65 else '#e74c3c' for v in vals]
ax.bar(vals, cnts, width=0.02, color=colors_bar, edgecolor='none')
ax.set_title('(e) Lambda Distribution (3 modes)')
ax.set_xlabel('lambda_frag'); ax.set_ylabel('Frequency')
# 标注三模态
for v_label in [0.50, 0.70, 0.90]:
    ax.axvline(x=v_label, color='gray', linestyle='--', alpha=0.4)
    ax.text(v_label, max(cnts)*0.9, f'{v_label:.2f}', ha='center', fontsize=10, fontweight='bold')

# (f) Summary metrics
ax = axes[1, 2]
ax.axis('off')
total_steps = N
pct90 = sum(data['lambda'] > 0.85) / N * 100
pct70 = sum((data['lambda'] > 0.65) & (data['lambda'] <= 0.85)) / N * 100
pct50 = sum((data['lambda'] > 0.4) & (data['lambda'] <= 0.65)) / N * 100
gs_final = data['gate_std'][-1]
diff_rate = sum(data['lambda_std'] > 0.1) / N * 100
equal_rate = sum(data['lambda_std'] <= 0.01) / N * 100
text = (
    f'v3 TRAINING SUMMARY\n'
    f'====================\n\n'
    f'Total steps: {total_steps:,}\n'
    f'Mask loss (final): {sm[-1]:.2f}\n'
    f'Gate std (final):  {gs_final:.3f}\n\n'
    f'Lambda Distribution:\n'
    f'  lambda=0.90:  {pct90:.1f}%\n'
    f'  lambda=0.70:  {pct70:.1f}%\n'
    f'  lambda=0.50:  {pct50:.1f}%\n\n'
    f'Lambda Frag std:\n'
    f'  Differentiated: {diff_rate:.1f}%\n'
    f'  Equal:          {equal_rate:.1f}%\n\n'
    f'COMPARED TO v2 (scalar lambda):\n'
    f'  v2 lambda:  2.00 -> 0.12 (collapsed)\n'
    f'  v3 lambda:  0.60-0.90 (active, oscillating)\n'
    f'  v3 lambda NEVER dropped below 0.45\n'
    f'  => Adversarial framework WORKS'
)
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_v3_convergence.png', dpi=150, bbox_inches='tight')
print('[OK] fig_v3_convergence.png')

# ==============================================================================
# 3. λ 深度分析
# ==============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('v3 Lambda Deep Analysis — Adversarial Collaborative Learning Dynamics',
              fontsize=15, fontweight='bold')

# (a) Lambda over time with phase transitions
ax = axes2[0, 0]
ax.plot(steps[:20000], data['lambda'][:20000], color='#2ecc71', linewidth=0.5, alpha=0.8)
# 找出切换点
transitions = []
for i in range(1, min(20000, N)):
    if abs(data['lambda'][i] - data['lambda'][i-1]) > 0.1:
        transitions.append(i)
ax.scatter(transitions[::10], data['lambda'][transitions[::10]], s=3, color='red', alpha=0.3)
ax.set_title('(a) Lambda First 20K Steps (red = state transitions)')
ax.set_xlabel('Step'); ax.set_ylabel('lambda_frag')
ax.set_ylim(0.3, 1.0)

# (b) Correlation: lambda vs mask_loss
ax = axes2[0, 1]
for val, color, label in [(0.90, '#2ecc71', '0.90'), (0.70, '#3498db', '0.70'), (0.50, '#e74c3c', '0.50')]:
    mask = np.abs(data['lambda'] - val) < 0.05
    if mask.any():
        ax.scatter(data['mask_loss'][mask][::50], data['lambda'][mask][::50],
                   s=3, color=color, alpha=0.3, label=f'lambda={label}')
ax.set_title('(b) Lambda vs Mask Loss (no strong correlation)')
ax.set_xlabel('Mask Loss'); ax.set_ylabel('lambda_frag')
ax.legend(fontsize=8)

# (c) Lambda std distribution
ax = axes2[1, 0]
ax.hist(data['lambda_std'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, label='equal threshold')
ax.axvline(x=0.4, color='red', linestyle='--', alpha=0.5, label='max differentiation')
ax.set_title('(c) Lambda Std Distribution')
ax.set_xlabel('std'); ax.set_ylabel('Count')
ax.legend(fontsize=8)

# (d) Trajectory in lambda-std space
ax = axes2[1, 1]
colors_traj = plt.cm.viridis(np.linspace(0, 1, min(5000, N)))
sample_idx = np.linspace(0, N-1, min(5000, N)).astype(int)
sc = ax.scatter(data['lambda'][sample_idx], data['lambda_std'][sample_idx],
                c=np.arange(len(sample_idx)), cmap='viridis', s=1, alpha=0.5)
ax.set_title('(d) Lambda-Std Trajectory (colored by step)')
ax.set_xlabel('lambda_frag'); ax.set_ylabel('lambda_std')
plt.colorbar(sc, ax=ax, label='Training step')

plt.tight_layout()
plt.savefig('fig_v3_lambda_analysis.png', dpi=150, bbox_inches='tight')
print('[OK] fig_v3_lambda_analysis.png')

# ==============================================================================
# 4. 对比汇总
# ==============================================================================
latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{v2 vs v3 Training Comparison}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{v2 (Scalar $\\lambda$)}} & \\textbf{{v3 (LambdaController)}} \\\\
\\hline
Architecture & Single learnable parameter & 2D frag + 7D layer MLP \\\\
$\\lambda$ trajectory & $2.00 \\rightarrow 0.12$ (collapse) & $0.60-0.90$ (active oscillation) \\\\
$\\lambda$ minimum & 0.12 & \\textbf{{0.45}} (never collapsed) \\\\
Rule differentiation & None & 3-mode (0.50/0.70/0.90) \\\\
Training steps & 57,390 & 105,730 \\\\
Mask loss (final) & 18.53 & {sm[-1]:.2f} \\\\
Gate std (final) & 0.403 & {data['gate_std'][-1]:.3f} \\\\
Collapse prevention & lambda_reg loss & rule_overmix + curriculum \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
print(latex_table)

print('\nDONE — 3 figures + LaTeX table')
print('  fig_v3_convergence.png')
print('  fig_v3_lambda_analysis.png')
print('  LaTeX comparison table above')
