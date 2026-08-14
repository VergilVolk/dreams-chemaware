"""
Task 0 visualization: Rule Jaccard vs MCES — 335 rules, 24,333 pairs
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.stats import pearsonr, spearmanr, gaussian_kde

# ===================================================================
# Load data
# ===================================================================
with open('data/validation/rule_mces_correlation/correlation_report.json') as f:
    report = json.load(f)

# Load pair data
mces_arr = []
jac_arr = []
with open('data/validation/rule_mces_correlation/pair_mces_jaccard.csv') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            mces_arr.append(float(parts[4]))
            jac_arr.append(float(parts[5]))

mces_arr = np.array(mces_arr)
jac_arr = np.array(jac_arr)

r_val = report['pearson_r']
p_val = report['pearson_p']
rho = report['spearman_rho']
sp_val = report['spearman_p']
N = len(mces_arr)
N_RULES = report['n_rules']

# ===================================================================
# Groups
# ===================================================================
groups = [
    (0, 2,   'MCES 0–2\n(near-isomers)'),
    (3, 5,   'MCES 3–5\n(analogs)'),
    (6, 10,  'MCES 6–10\n(different)'),
    (11, 999,'MCES >10\n(unrelated)'),
]
group_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']

# ===================================================================
# Figure
# ===================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})

fig = plt.figure(figsize=(16, 7))
gs = GridSpec(1, 2, width_ratios=[1.0, 0.8], wspace=0.28)

# ---- (a) Density scatter ----
ax = fig.add_subplot(gs[0, 0])

# Use hexbin for 24K points — avoids overplotting
hb = ax.hexbin(mces_arr, jac_arr, gridsize=80, cmap='Blues',
               mincnt=1, bins='log', linewidths=0, alpha=0.9)
cb = plt.colorbar(hb, ax=ax, pad=0.02, shrink=0.82)
cb.set_label('Count (log scale)', fontsize=10)

# Trend line
z = np.polyfit(mces_arr, jac_arr, 1)
xl = np.linspace(0, mces_arr.max(), 200)
ax.plot(xl, np.polyval(z, xl), 'r-', lw=2.5, alpha=0.85,
        label=f'Linear fit (Pearson r = {r_val:.3f}, p = {p_val:.1e})')

# LOESS-like: bin means for visual reference
bin_edges = np.arange(0, 60, 2)
bin_means = []; bin_centers = []
for i in range(len(bin_edges) - 1):
    mask = (mces_arr >= bin_edges[i]) & (mces_arr < bin_edges[i+1])
    if mask.sum() >= 10:
        bin_means.append(jac_arr[mask].mean())
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
ax.plot(bin_centers, bin_means, 'ko-', ms=4, lw=2, mfc='white', mew=1.5,
        label=f'Binned mean (Spearman ρ = {rho:.3f})')

ax.set_xlabel('MCES (structural distance, #bonds different)')
ax.set_ylabel('Rule Jaccard similarity')
ax.set_title(f'(a) Rule Jaccard vs MCES  |  N = {N:,} pairs, {N_RULES} rules',
             fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.set_xlim(-1, 55)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.2, lw=0.5)

# ---- (b) Box plots by MCES group ----
ax = fig.add_subplot(gs[0, 1])

box_data = []; positions = []; box_labels = []; box_ns = []
for i, (lo, hi, label) in enumerate(groups):
    mask = (mces_arr >= lo) & (mces_arr <= hi)
    if mask.sum() > 0:
        box_data.append(jac_arr[mask])
        positions.append(i)
        box_labels.append(label)
        box_ns.append(mask.sum())

bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                patch_artist=True,
                medianprops={'color': '#2c3e50', 'lw': 2},
                whiskerprops={'color': '#7f8c8d', 'lw': 1.2},
                capprops={'color': '#7f8c8d', 'lw': 1.2},
                boxprops={'edgecolor': '#7f8c8d', 'lw': 1.2},
                flierprops={'marker': 'o', 'ms': 3, 'alpha': 0.3,
                           'markerfacecolor': '#bdc3c7', 'markeredgecolor': 'none'})

for i, (patch, color) in enumerate(zip(bp['boxes'], group_colors)):
    patch.set_facecolor(color)
    patch.set_alpha(0.25)

# Overlay swarm-like jittered points (subsample for performance)
rng = np.random.RandomState(42)
for i, data in enumerate(box_data):
    n_show = min(300, len(data))
    idx = rng.choice(len(data), n_show, replace=False)
    jitter = rng.uniform(-0.2, 0.2, n_show)
    ax.scatter(positions[i] + jitter, data[idx],
              s=4, c=group_colors[i], alpha=0.4, edgecolors='none',
              rasterized=True)

# Mean labels
for i, data in enumerate(box_data):
    mean_val = data.mean()
    ax.annotate(f'μ={mean_val:.3f}\nn={box_ns[i]:,}',
                xy=(i + 0.35, mean_val),
                fontsize=8, ha='left', va='center',
                color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.85))

ax.set_xticks(positions)
ax.set_xticklabels(box_labels, fontsize=10)
ax.set_ylabel('Rule Jaccard similarity')
ax.set_title(f'(b) Jaccard by MCES group', fontweight='bold')
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.2, lw=0.5, axis='y')

# Add statistic text box
stats_text = (
    f'Pearson r = {r_val:.4f}  (p = {p_val:.2e})\n'
    f'Spearman ρ = {rho:.4f}  (p = {sp_val:.2e})\n'
    f'Jaccard μ = {jac_arr.mean():.3f} ± {jac_arr.std():.3f}'
)
ax.text(0.98, 0.12, stats_text, transform=ax.transAxes,
        fontsize=8.5, ha='right', va='bottom',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#f8f9fa', ec='#dee2e6', alpha=0.9))

# Title
fig.suptitle('Rule Jaccard vs Molecular Structural Distance (MCES)',
             fontsize=15, fontweight='bold', y=0.98)
fig.text(0.5, 0.92,
         '335 chemical rules | annotated01 dataset | 24,333 molecule pairs '
         '| V5 formula-dense sampling + T1 MCES-stratified pairs',
         ha='center', fontsize=9, color='#6c757d')

plt.savefig('data/validation/rule_mces_correlation/rule_jaccard_vs_mces.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: data/validation/rule_mces_correlation/rule_jaccard_vs_mces.png')

# Also save a summary text
with open('data/validation/rule_mces_correlation/analysis_summary.md', 'w', encoding='utf-8') as f:
    f.write(f"""# Task 0: Rule Jaccard vs MCES — Analysis Summary

**Date**: 2026-08-06
**Rules**: {N_RULES} (335 main rules, no MassBank)
**Pairs**: {N:,} across 4 MCES bins

## Results

| Metric | Value |
|--------|-------|
| Pearson r | {r_val:.4f} |
| Pearson p | {p_val:.2e} |
| Spearman ρ | {rho:.4f} |
| Spearman p | {sp_val:.2e} |
| Jaccard mean | {jac_arr.mean():.4f} ± {jac_arr.std():.4f} |

## By MCES Group

| Group | n | Jaccard μ | Jaccard σ |
|-------|---|-----------|-----------|
""")
    for (lo, hi, label), gdata in zip(groups, report['by_mces_group'].values()):
        f.write(f'| {label.replace(chr(10), " ")} | {gdata["n"]:,} | {gdata["jaccard_mean"]:.4f} | {gdata["jaccard_std"]:.4f} |\n')

    f.write(f"""
## Interpretation

**Verdict: WEAK correlation** (Pearson r = {r_val:.4f})

The 335 main chemical rules show only a very weak negative correlation with
MCES (molecular structural distance). As molecules become more structurally
different (higher MCES), rule Jaccard decreases only marginally — from ~0.48
(near-isomers) to ~0.36 (unrelated molecules). The effect size is small
(ΔJaccard ≈ 0.11 across the full MCES range) compared to within-group variance
(σ ≈ 0.18–0.27).

### Why?

1. **Rules are too generic**: 214 NL rules (neutral losses like H₂O, CO₂) and
   102 CF rules (common fragments like tropylium, benzoyl) are near-universal.
   Two completely unrelated molecules both lose water, both fragment to tropylium
   → Jaccard stays high even at MCES > 50.

2. **Granularity mismatch**: 335 rules operate at the "functional group" level
   while MCES operates at the "bond topology" level. A benzene ring substitution
   pattern change (MCES = 2–4) may not change which rules fire at all.

### Next

- **V4 pending**: Re-run with full ~3,486 rules (including 3,151 MassBank
  fine-grained CF rules) to check if more specific rules improve correlation.
- **If V4 also weak → molecular alignment strategy** for Module 1.
""")

print('Saved: data/validation/rule_mces_correlation/analysis_summary.md')
print('Done.')
