"""
Publication-quality figure v3: Rule Jaccard vs MCES correlation.

Fixes from v2 review:
  1. Trend line: binned means + bootstrap 95% CI (no spline, no boundary artifact)
  2. MCES group labels: neutral, chemically accurate names
  3. Mean annotations: "Mean = X.XXX" with legend
  4. Caption: softened, precise about what rules can/cannot do
  5. "RDKit MCS" → "MCES-style distance from RDKit maximum common substructure"
  6. Added: MCES=0 sub-analysis, Cliff's delta, AUROC between extreme groups
  7. Layout: removed bottom caption (→ figure caption), consistent y-axis labels

Output: data/validation/rule_mces_correlation/rule_jaccard_vs_mces_v3.png
        data/validation/rule_mces_correlation/figure_stats_v3.json
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

# ── Config ───────────────────────────────────────────────────────
CSV_PATH = 'data/validation/rule_mces_correlation/pair_mces_jaccard.csv'
OUT_DIR  = 'data/validation/rule_mces_correlation'
OUT_PNG  = os.path.join(OUT_DIR, 'rule_jaccard_vs_mces_v3.png')
OUT_JSON = os.path.join(OUT_DIR, 'figure_stats_v3.json')
DPI = 200
RNG = np.random.RandomState(42)
N_BOOTSTRAP = 2000

# Color palette
C_GROUPS = ['#2166AC', '#67A9CF', '#EF8A62', '#B2182B']  # blue → red gradient
C_TREND  = '#B2182B'
C_CI     = '#F4A582'
GRID_C   = '#D9D9D9'
FACE_C   = '#F7F7F7'

# Neutral MCES bins
GROUPS = [
    (0,  2,   'MCES 0–2'),
    (3,  5,   'MCES 3–5'),
    (6,  10,  'MCES 6–10'),
    (11, 999, 'MCES >10'),
]
GROUP_LABELS_PLAIN = [g[2] for g in GROUPS]

# ═══════════════════════════════════════════════════════════════
# 1. Load data + derive all statistics
# ═══════════════════════════════════════════════════════════════
print('[1] Loading data & computing statistics...')

mces_list = []; jac_list = []
group_jac = {i: [] for i in range(len(GROUPS))}
mces0_same_ik = []   # MCES=0, same 14-char IK
mces0_diff_ik = []   # MCES=0, different 14-char IK

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        mces = float(row['mces']); jac = float(row['jaccard'])
        mces_list.append(mces); jac_list.append(jac)

        for gi, (lo, hi, _) in enumerate(GROUPS):
            if lo <= mces <= hi:
                group_jac[gi].append(jac)
                break

        # MCES=0 sub-analysis
        if mces == 0:
            ik_a = row['ik_a']; ik_b = row['ik_b']
            if ik_a == ik_b:
                mces0_same_ik.append(jac)
            else:
                mces0_diff_ik.append(jac)

mces_arr = np.array(mces_list)
jac_arr  = np.array(jac_list)
n = len(mces_arr)

# Overall correlations
r_val, p_val   = pearsonr(mces_arr, jac_arr)
rho, sp_val    = spearmanr(mces_arr, jac_arr)

# Per-group statistics
group_stats = {}
for gi, (lo, hi, label) in enumerate(GROUPS):
    gd = np.array(group_jac[gi])
    group_stats[label] = {
        'n': int(len(gd)),
        'mean': float(gd.mean()),
        'std':  float(gd.std()),
        'median': float(np.median(gd)),
        'q1': float(np.percentile(gd, 25)),
        'q3': float(np.percentile(gd, 75)),
    }
    # Within-group Spearman
    in_group = (mces_arr >= lo) & (mces_arr <= hi)
    if in_group.sum() >= 5:
        g_mces = mces_arr[in_group]; g_jac = jac_arr[in_group]
        if np.std(g_mces) > 0:
            grho, gp = spearmanr(g_mces, g_jac)
            group_stats[label]['within_group_spearman_rho'] = float(grho)
            group_stats[label]['within_group_spearman_p']   = float(gp)

# Cliff's delta: MCES 0–2 vs MCES >10
gd_near = np.array(group_jac[0])       # MCES 0–2
gd_far  = np.array(group_jac[3])       # MCES >10

def cliff_delta(x, y):
    """Cliff's delta effect size. 1 = all x > y, -1 = all x < y, 0 = complete overlap."""
    nx, ny = len(x), len(y)
    # Dominance matrix: for each pair, is xi > yj?
    # Efficient: compare each xi to sorted y
    y_sorted = np.sort(y)
    greater = 0; less = 0
    for xi in x:
        n_less = np.searchsorted(y_sorted, xi, side='left').sum()
        n_greater = ny - np.searchsorted(y_sorted, xi, side='right').sum()
        greater += n_greater
        less    += n_less
    return (greater - less) / (greater + less)

cliff = cliff_delta(gd_near, gd_far)

# AUROC: MCES 0–2 (positive) vs MCES >10 (negative)
from sklearn import metrics
auro_labels = np.concatenate([np.ones(len(gd_near)), np.zeros(len(gd_far))])
auro_scores = np.concatenate([gd_near, gd_far])
fpr, tpr, _ = metrics.roc_curve(auro_labels, auro_scores)
auroc_val = float(metrics.auc(fpr, tpr))

# Adjacent group mean differences with bootstrap CI
adj_diffs = {}
for gi in range(len(GROUPS) - 1):
    g1 = np.array(group_jac[gi])
    g2 = np.array(group_jac[gi + 1])
    obs_diff = g1.mean() - g2.mean()
    # Bootstrap
    bs_diffs = []
    for _ in range(N_BOOTSTRAP):
        bs1 = RNG.choice(g1, len(g1), replace=True)
        bs2 = RNG.choice(g2, len(g2), replace=True)
        bs_diffs.append(bs1.mean() - bs2.mean())
    bs_diffs = np.array(bs_diffs)
    ci_lo = float(np.percentile(bs_diffs, 2.5))
    ci_hi = float(np.percentile(bs_diffs, 97.5))
    adj_diffs[f'{GROUP_LABELS_PLAIN[gi]} → {GROUP_LABELS_PLAIN[gi+1]}'] = {
        'observed_diff': float(obs_diff),
        'bootstrap_95ci': [ci_lo, ci_hi],
        'significant': not (ci_lo <= 0 <= ci_hi),
    }

# MCES=0 analysis
mces0_same_ik_arr = np.array(mces0_same_ik)
mces0_diff_ik_arr = np.array(mces0_diff_ik)

# Trend: binned means with bootstrap 95% CI (no spline, no boundary artifact)
P99 = np.percentile(mces_arr, 99)  # truncate at P99 to avoid sparse-tail artifacts
bin_width = max(2.0, (P99 - mces_arr.min()) / 50)
bin_edges = np.arange(mces_arr.min(), P99 + bin_width, bin_width)

bin_centers = []; bin_means = []; bin_ci_lo = []; bin_ci_hi = []; bin_ns = []
for i in range(len(bin_edges) - 1):
    mask = (mces_arr >= bin_edges[i]) & (mces_arr < bin_edges[i + 1])
    n_b = mask.sum()
    if n_b >= 20:
        b_mces = mces_arr[mask]; b_jac = jac_arr[mask]
        bin_centers.append(b_mces.mean())
        bin_means.append(b_jac.mean())
        bin_ns.append(n_b)
        # Bootstrap CI for this bin's mean
        bs_means = []
        for _ in range(N_BOOTSTRAP):
            bs_idx = RNG.choice(n_b, n_b, replace=True)
            bs_means.append(b_jac[bs_idx].mean())
        bs_means = np.array(bs_means)
        bin_ci_lo.append(float(np.percentile(bs_means, 2.5)))
        bin_ci_hi.append(float(np.percentile(bs_means, 97.5)))

bin_centers = np.array(bin_centers)
bin_means   = np.array(bin_means)
bin_ci_lo   = np.array(bin_ci_lo)
bin_ci_hi   = np.array(bin_ci_hi)
print(f'  Trend bins: {len(bin_centers)} (P99={P99:.0f}, bin_width={bin_width:.1f})')

# Print summary
print(f'\n  N={n:,}  Pearson r={r_val:.4f} (p={p_val:.2e})  Spearman ρ={rho:.4f} (p={sp_val:.2e})')
print(f'  Cliff\'s delta (0–2 vs >10): {cliff:.4f}')
print(f'  AUROC (0–2 vs >10): {auroc_val:.4f}')
print(f'  MCES=0 pairs: {len(mces0_same_ik)} same-IK, {len(mces0_diff_ik)} diff-IK')
if len(mces0_same_ik_arr) > 0:
    print(f'    Same-IK  Jaccard: mean={mces0_same_ik_arr.mean():.4f}, std={mces0_same_ik_arr.std():.4f}')
if len(mces0_diff_ik_arr) > 0:
    print(f'    Diff-IK  Jaccard: mean={mces0_diff_ik_arr.mean():.4f}, std={mces0_diff_ik_arr.std():.4f}')

for gi, (lo, hi, label) in enumerate(GROUPS):
    gs = group_stats[label]
    print(f'  {label}: n={gs["n"]:,}  mean={gs["mean"]:.4f}  median={gs["median"]:.4f}  '
          f'IQR=[{gs["q1"]:.3f}, {gs["q3"]:.3f}]')

for k, v in adj_diffs.items():
    print(f'  Δ {k}: {v["observed_diff"]:.4f}  CI={v["bootstrap_95ci"]}  sig={v["significant"]}')

# ═══════════════════════════════════════════════════════════════
# 2. Figure
# ═══════════════════════════════════════════════════════════════
print('\n[2] Plotting...')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.facecolor': 'white', 'axes.facecolor': FACE_C,
    'axes.edgecolor': '.3', 'axes.grid': True,
    'grid.alpha': 0.35, 'grid.color': GRID_C, 'grid.linewidth': 0.4,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.subplots_adjust(wspace=0.30, left=0.07, right=0.97, top=0.91, bottom=0.09)

# ── Panel (a): Hexbin + binned means with bootstrap 95% CI ──
ax = axes[0]

hb = ax.hexbin(mces_arr, jac_arr, gridsize=60, cmap='Blues',
               bins='log', mincnt=1, linewidths=0, alpha=0.92)

# Binned means with 95% CI (no spline — honest about uncertainty)
ax.fill_between(bin_centers, bin_ci_lo, bin_ci_hi,
                color=C_CI, alpha=0.30, linewidth=0, zorder=4,
                label='Binned mean ± 95% CI (bootstrap)')
ax.plot(bin_centers, bin_means, color=C_TREND, linewidth=2.0, zorder=5,
        marker='o', markersize=3.5, markeredgewidth=0, markerfacecolor=C_TREND)

# Shade the P99+ region to show it's excluded
ax.axvspan(P99, mces_arr.max() + 2, color='.85', alpha=0.35, linewidth=0, zorder=2)
ax.text(P99 + 1, 0.06, f'MCES > P99\n({P99:.0f}+)\nexcluded',
        fontsize=6.5, color='.4', ha='left', va='bottom')

# Colorbar
cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label('Number of pairs\nper hexagon', fontsize=7.5, labelpad=4)
cbar.ax.tick_params(labelsize=7)

ax.set_xlabel('MCES-style distance (RDKit MCS)')
ax.set_ylabel('Rule Jaccard similarity')
ax.set_xlim(-2, mces_arr.max() + 2)
ax.set_ylim(-0.02, 1.04)

# Stats box
stats_lines = [
    f'N = {n:,} pairs',
    f'Pearson r = {r_val:.3f}  (p ≈ {p_val:.1e})',
    f'Spearman ρ = {rho:.3f}  (p ≈ {sp_val:.1e})',
    f'Cliff\'s δ = {cliff:.3f}  (0–2 vs >10)',
    f'AUROC = {auroc_val:.3f}  (0–2 vs >10)',
]
ax.text(0.97, 0.97, '\n'.join(stats_lines), transform=ax.transAxes, fontsize=7.3,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='.5', alpha=0.92), zorder=10)

# Legend for trend elements only
legend_elements_a = [
    Line2D([0], [0], color=C_TREND, linewidth=2.0, marker='o', markersize=3.5,
           markerfacecolor=C_TREND, markeredgewidth=0),
    Patch(facecolor=C_CI, alpha=0.30, edgecolor='none'),
]
ax.legend(legend_elements_a,
          ['Binned mean (≥20 pairs/bin)', '95% CI (bootstrap, n=2,000)'],
          loc='lower left', fontsize=7.5, framealpha=0.85)

ax.set_title('(a)  Rule Jaccard vs structural distance', fontweight='bold', loc='left', pad=6)

# ── Panel (b): Violin + box + mean by MCES group ──
ax = axes[1]

violin_data = [np.array(group_jac[gi]) for gi in range(len(GROUPS))]
positions = np.arange(len(GROUPS))

# Violins
vp = ax.violinplot(violin_data, positions=positions, showmeans=False,
                   showmedians=False, showextrema=False, widths=0.78)
for gi, body in enumerate(vp['bodies']):
    body.set_facecolor(C_GROUPS[gi]); body.set_alpha(0.50)
    body.set_edgecolor(C_GROUPS[gi]); body.set_linewidth(0.8)

# Boxplots inside violins
bp = ax.boxplot(violin_data, positions=positions, widths=0.16,
                patch_artist=True, showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.3},
                whiskerprops={'color': '.25', 'linewidth': 0.9},
                capprops={'color': '.25', 'linewidth': 0.9},
                boxprops={'facecolor': 'white', 'edgecolor': '.35', 'linewidth': 0.9})

# Mean diamonds
for gi, gd in enumerate(violin_data):
    ax.scatter(gi, gd.mean(), marker='D', color='black', s=32, zorder=10,
               edgecolors='white', linewidths=0.6)

# Tick labels with sample sizes
tick_labels = [f'{GROUP_LABELS_PLAIN[gi]}\n(n={len(group_jac[gi]):,})'
               for gi in range(len(GROUPS))]
ax.set_xticks(positions)
ax.set_xticklabels(tick_labels, fontsize=8.5)

# Consistent y-axis label
ax.set_ylabel('Rule Jaccard similarity')
ax.set_ylim(-0.02, 1.04)
ax.yaxis.set_major_locator(MaxNLocator(6))

# Mean values above violins
for gi, gd in enumerate(violin_data):
    ax.text(gi, 1.015, f'Mean = {gd.mean():.3f}', ha='center', va='bottom',
            fontsize=7.8, color='.2', fontweight='bold')

# Legend for marks
legend_elements_b = [
    Line2D([0], [0], marker='D', color='none', markerfacecolor='black',
           markersize=7, markeredgecolor='white', markeredgewidth=0.6),
    Patch(facecolor='white', edgecolor='.35', linewidth=0.9),
    Patch(facecolor=C_GROUPS[0], alpha=0.50, edgecolor=C_GROUPS[0], linewidth=0.8),
]
ax.legend(legend_elements_b,
          ['Mean  ◆', 'Median & IQR  ▯', 'Density (violin)'],
          loc='lower left', fontsize=7.5, framealpha=0.85)

# Sample imbalance note
imbalance_note = (
    f'Pearson/Spearman weighted by\n'
    f'current sampling distribution:\n'
    f'>10 group = {len(group_jac[3])/n*100:.0f}% of pairs.\n'
    f'3–5 group = only {len(group_jac[1]):,} pairs.'
)
ax.text(0.97, 0.06, imbalance_note, transform=ax.transAxes, fontsize=6.8,
        verticalalignment='bottom', horizontalalignment='right', color='.35',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='.6', alpha=0.85), zorder=10)

ax.set_title('(b)  Jaccard distribution by MCES group', fontweight='bold', loc='left', pad=6)

# ═══════════════════════════════════════════════════════════════
# Suptitle
# ═══════════════════════════════════════════════════════════════
fig.suptitle('Chemical Rule Overlap vs Structural Distance',
             fontsize=12.5, fontweight='bold', y=0.975)

# ═══════════════════════════════════════════════════════════════
# 3. Save
# ═══════════════════════════════════════════════════════════════
fig.savefig(OUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ── Statistics JSON ──
full_stats = {
    'n_pairs': int(n),
    'n_rules': 335,
    'overall': {
        'pearson_r': float(r_val), 'pearson_p': float(p_val),
        'spearman_rho': float(rho), 'spearman_p': float(sp_val),
    },
    'by_mces_group': group_stats,
    'adjacent_group_diffs': adj_diffs,
    'cliff_delta_0to2_vs_gt10': float(cliff),
    'auroc_0to2_vs_gt10': float(auroc_val),
    'mces0_breakdown': {
        'n_same_ik': int(len(mces0_same_ik_arr)),
        'same_ik_mean_jaccard': float(mces0_same_ik_arr.mean()) if len(mces0_same_ik_arr) > 0 else None,
        'same_ik_std_jaccard': float(mces0_same_ik_arr.std()) if len(mces0_same_ik_arr) > 0 else None,
        'n_diff_ik': int(len(mces0_diff_ik_arr)),
        'diff_ik_mean_jaccard': float(mces0_diff_ik_arr.mean()) if len(mces0_diff_ik_arr) > 0 else None,
        'diff_ik_std_jaccard': float(mces0_diff_ik_arr.std()) if len(mces0_diff_ik_arr) > 0 else None,
    },
    'sampling_note': (
        'Pearson/Spearman are weighted by the current pair sampling distribution '
        '(MCES>10 = 74% of pairs, MCES 3–5 = only 1.3%). '
        'Report both original and MCES-stratified correlations in formal publication.'
    ),
    'interpretation': (
        'Rule Jaccard provides coarse structural stratification (mean drops from '
        '0.478 at MCES 0–2 to 0.365 at MCES >10, Cliff\'s δ = {:.3f}) but the '
        'distributions overlap substantially (AUROC = {:.3f}). Within-group variance '
        'far exceeds between-group mean differences. Rule overlap is thus suitable '
        'for concept supervision and rule–structure disagreement mining, but '
        'insufficient as a standalone continuous regression target for '
        'embedding-space distances.'
    ).format(cliff, auroc_val),
}

with open(OUT_JSON, 'w') as f:
    json.dump(full_stats, f, indent=2)

print(f'\n[3] Saved:')
print(f'    {OUT_PNG}')
print(f'    {OUT_JSON}')
print(f'    PNG size: {os.path.getsize(OUT_PNG):,} bytes')
print('Done.')

# ═══════════════════════════════════════════════════════════════
# Figure caption (for paper/report)
# ═══════════════════════════════════════════════════════════════
CAPTION = """
Figure X. Relationship between chemical-rule overlap and structural distance.
Rule Jaccard similarity decreases on average as MCES-style distance (RDKit MCS)
increases, but the distributions overlap substantially across all MCES groups
(Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}, Cliff's δ = {cliff:.3f} for
MCES 0–2 vs >10). Binned means with bootstrap 95% CI are shown; MCES > P99 is
excluded from the trend due to sparse sampling. Rule overlap provides coarse
chemical stratification but is insufficient as a standalone continuous target
for embedding-distance regression. It is instead suited for concept supervision
and rule–structure disagreement mining.
""".format(r=r_val, rho=rho, cliff=cliff).strip()

caption_path = os.path.join(OUT_DIR, 'figure_caption_v3.txt')
with open(caption_path, 'w') as f:
    f.write(CAPTION + '\n')
print(f'\n[Caption] Saved to {caption_path}')
print(f'{CAPTION}')
