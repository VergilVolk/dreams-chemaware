"""
Task 0 Final: 335 vs 3,486 rules comparison visualization
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ===================================================================
# Data from both runs
# ===================================================================
results = {
    '335 rules': {
        'pearson_r': -0.1672, 'pearson_p': 4.67e-152,
        'spearman_rho': -0.1217, 'spearman_p': 5.42e-81,
        'jaccard_mean': 0.3754, 'jaccard_std': 0.1863,
        'groups': {
            '0–2': (939, 0.4782, 0.2677),
            '3–5': (310, 0.4218, 0.2075),
            '6–10': (5057, 0.3910, 0.1858),
            '>10': (18027, 0.3649, 0.1786),
        }
    },
    '3,486 rules': {
        'pearson_r': -0.1937, 'pearson_p': 2.98e-204,
        'spearman_rho': -0.1435, 'spearman_p': 4.22e-112,
        'jaccard_mean': 0.3016, 'jaccard_std': 0.1572,
        'groups': {
            '0–2': (939, 0.4435, 0.2753),
            '3–5': (310, 0.3553, 0.1919),
            '6–10': (5057, 0.3214, 0.1627),
            '>10': (18027, 0.2882, 0.1475),
        }
    },
}

# ===================================================================
# Figure
# ===================================================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white',
})

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, height_ratios=[0.55, 0.45], hspace=0.35, wspace=0.25)

colors = {'335 rules': '#3498db', '3,486 rules': '#e74c3c'}
group_labels = ['MCES 0–2\n(near-isomers)', 'MCES 3–5\n(analogs)',
                'MCES 6–10\n(different)', 'MCES >10\n(unrelated)']

# ---- (a) Grouped bar chart: Jaccard by MCES group ----
ax = fig.add_subplot(gs[0, :])

x = np.arange(len(group_labels))
width = 0.35

for j, (name, data) in enumerate(results.items()):
    means = [data['groups'][k][1] for k in ['0–2', '3–5', '6–10', '>10']]
    stds = [data['groups'][k][2] for k in ['0–2', '3–5', '6–10', '>10']]
    ns = [data['groups'][k][0] for k in ['0–2', '3–5', '6–10', '>10']]
    offset = (j - 0.5) * width
    bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                  color=colors[name], alpha=0.85, capsize=4,
                  label=f'{name} (Pearson r={data["pearson_r"]:.3f}, ρ={data["spearman_rho"]:.3f})')
    # n labels
    for i, (m, s, n) in enumerate(zip(means, stds, ns)):
        ax.text(x[i] + offset, m + s + 0.008, f'n={n}',
                ha='center', fontsize=7, color=colors[name])

ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_ylabel('Mean Rule Jaccard similarity')
ax.set_title('(a) Rule Jaccard by MCES Group — 335 vs 3,486 rules', fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2, lw=0.5, axis='y')
ax.set_ylim(0, 0.82)

# ---- (b) Delta-Jaccard plot (key metric) ----
ax = fig.add_subplot(gs[1, 0])
mces_positions = [1, 4, 8, 20]  # representative MCES values

for name, data in results.items():
    means = [data['groups'][k][1] for k in ['0–2', '3–5', '6–10', '>10']]
    ax.plot(mces_positions, means, 'o-', color=colors[name], lw=2.5, ms=8,
            label=name, markeredgecolor='white', markeredgewidth=1.5)

ax.set_xlabel('MCES (structural distance)')
ax.set_ylabel('Mean Rule Jaccard')
ax.set_title('(b) Jaccard decay with structural distance', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, lw=0.5)

# Add Δ annotations
for name, data in results.items():
    means = [data['groups'][k][1] for k in ['0–2', '3–5', '6–10', '>10']]
    delta = means[0] - means[-1]
    y_pos = 0.25 if '335' in name else 0.18
    ax.annotate(f'Δ = {delta:.3f}\n({name})',
                xy=(15, y_pos), fontsize=8.5, color=colors[name],
                bbox=dict(boxstyle='round', fc='white', ec=colors[name], alpha=0.85))

# ---- (c) Correlation comparison ----
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')

# Stats table
table_data = [
    ['Metric', '335 rules', '3,486 rules', 'Δ'],
    ['Pearson r', '-0.1672', '-0.1937', '-0.0265'],
    ['Spearman ρ', '-0.1217', '-0.1435', '-0.0218'],
    ['Jaccard μ (overall)', '0.3754', '0.3016', '-0.0738'],
    ['Jaccard @ MCES 0–2', '0.4782', '0.4435', '-0.0347'],
    ['Jaccard @ MCES >10', '0.3649', '0.2882', '-0.0767'],
    ['Δ MCES 0–2 to >10', '0.113', '0.155', '0.042'],
    ['Discrimination ratio', '1.31x', '1.54x', '0.23x'],
]

table = ax.table(cellText=table_data, cellLoc='center',
                 loc='center', colWidths=[0.28, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)

# Style header
for j in range(4):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Style rows
for i in range(1, len(table_data)):
    for j in range(4):
        if j == 0:
            table[i, j].set_facecolor('#ecf0f1')
        elif j == 3:
            val_str = str(table_data[i][j])
            try:
                val = float(val_str)
                c = '#27ae60' if val > 0 else '#e74c3c'
            except ValueError:
                c = '#7f8c8d'
            table[i, j].set_text_props(color=c, fontweight='bold')

ax.set_title('(c) Metrics comparison', fontweight='bold', y=1.02)

# Suptitle with verdict
fig.suptitle('Rule Jaccard vs MCES — Final Results\n'
             'Verdict: WEAK correlation for BOTH 335 and 3,486 rules',
             fontsize=15, fontweight='bold', y=1.01)

fig.text(0.5, 0.96, 'annotated01 | 24,333 pairs | formula-dense sampling + T1 MCES-stratified',
         ha='center', fontsize=9, color='#6c757d')

plt.savefig('data/validation/rule_mces_correlation/final_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: data/validation/rule_mces_correlation/final_comparison.png')

# ===================================================================
# Final analysis markdown
# ===================================================================
with open('data/validation/rule_mces_correlation/FINAL_ANALYSIS.md', 'w', encoding='utf-8') as f:
    f.write(f"""# Task 0: Rule Jaccard vs MCES — FINAL ANALYSIS

**Date**: 2026-08-06
**Data**: annotated01, 24,333 molecule pairs
**Sampling**: V5 formula-dense + T1 MCES-stratified

---

## Results Summary

| Metric | 335 main rules | 3,486 rules (incl. MassBank) |
|--------|---------------|------------------------------|
| Pearson r | -0.1672 | **-0.1937** |
| Spearman ρ | -0.1217 | **-0.1435** |
| Jaccard μ (overall) | 0.3754 | 0.3016 |
| Jaccard @ MCES 0–2 | 0.4782 | 0.4435 |
| Jaccard @ MCES 3–5 | 0.4218 | 0.3553 |
| Jaccard @ MCES 6–10 | 0.3910 | 0.3214 |
| Jaccard @ MCES >10 | 0.3649 | 0.2882 |
| Δ (0–2 → >10) | 0.113 | **0.155** |
| Discrimination ratio | 1.31× | **1.54×** |

## Key Findings

### 1. Adding MassBank rules improves discrimination but correlation remains weak

MassBank's 3,072 fine-grained CF rules increase the Jaccard separation between
near-isomers and unrelated molecules from 1.31× to 1.54× (a 23% improvement in
discrimination ratio). The Pearson r improves from -0.167 to -0.194.

However, **both correlations are still very weak** (|r| < 0.2). The within-group
variance (σ ≈ 0.15–0.28) remains much larger than the between-group effect
(Δ ≈ 0.11–0.16).

### 2. Why rules cannot proxy MCES

Chemical fragmentation rules operate at the **functional group** level:
- "Does this molecule lose H₂O? Does it fragment to tropylium?"
- Two molecules with completely different carbon skeletons can both lose water,
  both produce benzoyl fragments → high rule Jaccard despite high MCES.

MCES operates at the **bond topology** level:
- "How many bonds differ between the maximum common substructures?"
- A simple substitution (ortho → para, MCES ≈ 2) may not change which rules
  fire at all → low rule Jaccard discrimination for structurally similar molecules.

The **granularity mismatch** is fundamental: rules see functional groups,
MCES sees bonds. These are different levels of abstraction that don't
correlate strongly.

### 3. MCES 3–5 is naturally rare

Even with formula-dense sampling (500 groups × up to 30 pairs), we found only
310 pairs in MCES 3–5 out of 24,333 total (1.3%). This range — between
near-isomers and clearly different molecules — is inherently sparse in
chemical space. Molecules tend to be either very similar (isomers, same
formula) or clearly different (different formulas).

## Architecture Decision

**Verdict: Module 1 MUST use molecular alignment, NOT rule alignment.**

Chemical rules are insufficient as a structural similarity proxy for
contrastive learning. The rule Jaccard vs MCES correlation (r ≈ -0.17 to -0.19)
is too weak to guide meaningful triplet formation.

### Recommended path forward

1. **Module 1**: Molecular embedding alignment (ChemBERTa / MolFormer /
   SpecBridge-style) — align spectra to a pretrained molecular encoder's
   latent space.

2. **Module 2**: Chemical rules for post-hoc interpretability — after
   the model makes predictions, use rule matching vectors to explain
   which chemical transformations the model learned to recognize.

3. **Chemical rules as auxiliary loss**: Rules can still serve as a
   weak regularization signal (e.g., encourage spectra of molecules
   with similar rule vectors to have similar embeddings), but NOT as
   the primary alignment target.
""")
print('Saved: data/validation/rule_mces_correlation/FINAL_ANALYSIS.md')
print('Done.')
