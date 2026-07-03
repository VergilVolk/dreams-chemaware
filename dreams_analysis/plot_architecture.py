"""DreaMS 架构可视化 — 生成配套图片"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ======================================================================
# Fig 1: Architecture Overview
# ======================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
fig.suptitle('DreaMS Architecture Overview', fontsize=16, fontweight='bold', y=0.98)

# Input
ax.add_patch(plt.Rectangle((0.5, 8.2), 2, 0.8, fill=True, facecolor='#e8f4f8', edgecolor='#2980b9', lw=2))
ax.text(1.5, 8.6, 'Input Spectrum', ha='center', fontsize=11, fontweight='bold')
ax.text(1.5, 8.3, '(n_peaks, 2)\n[m/z, intensity]', ha='center', fontsize=8)

# PeakEncoder
ax.add_patch(plt.Rectangle((3.5, 6.5), 3, 2.5, fill=True, facecolor='#d5f5e3', edgecolor='#27ae60', lw=2))
ax.text(5.0, 8.5, 'PeakEncoder', ha='center', fontsize=12, fontweight='bold')
ax.text(5.0, 8.0, 'm/z -> FourierFeatures -> FFN_F (980-dim)', ha='center', fontsize=8)
ax.text(5.0, 7.6, '[m/z, int] -> FFN_P (44-dim)', ha='center', fontsize=8)
ax.text(5.0, 7.1, 'Concat -> (n, 980+44=1024)', ha='center', fontsize=8, fontweight='bold')
ax.text(5.0, 6.7, 'Prepend Precursor Peak (+1 token)', ha='center', fontsize=7, style='italic')

# SpectrumEncoder
ax.add_patch(plt.Rectangle((3.5, 3.2), 3, 2.5, fill=True, facecolor='#fef9e7', edgecolor='#f39c12', lw=2))
ax.text(5.0, 5.2, 'SpectrumEncoder', ha='center', fontsize=12, fontweight='bold')
ax.text(5.0, 4.7, '7x Transformer Blocks', ha='center', fontsize=9, fontweight='bold')
ax.text(5.0, 4.3, 'MultiheadAttention (8 heads)', ha='center', fontsize=8)
ax.text(5.0, 4.0, '+ Graphormer bias (m/z diffs)', ha='center', fontsize=8, color='#e74c3c', fontweight='bold')
ax.text(5.0, 3.7, '+ FeedForward + ScaleNorm', ha='center', fontsize=8)
ax.text(5.0, 3.4, 'Pre-Norm architecture', ha='center', fontsize=7, style='italic')

# PeakDecoder
ax.add_patch(plt.Rectangle((3.5, 0.8), 3, 1.5, fill=True, facecolor='#ebdef0', edgecolor='#8e44ad', lw=2))
ax.text(5.0, 1.8, 'PeakDecoder', ha='center', fontsize=12, fontweight='bold')
ax.text(5.0, 1.4, 's_0 (precursor) -> Task Head', ha='center', fontsize=8)
ax.text(5.0, 1.0, 's_k (masked) -> m/z classifier', ha='center', fontsize=8)

# Arrows
ax.annotate('', xy=(5, 6.5), xytext=(5, 7.3), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.annotate('', xy=(5, 3.2), xytext=(5, 5.7), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.annotate('', xy=(5, 0.8), xytext=(5, 2.3), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Side annotations
ax.text(7.2, 7.7, 'd_model = 1024', fontsize=9, fontfamily='monospace', color='#555')
ax.text(7.2, 4.5, 'Graphormer = key innovation', fontsize=9, fontfamily='monospace', color='#e74c3c')
ax.text(7.2, 4.2, 'Learns m/z diff -> attention bias', fontsize=8, color='#777')
ax.text(7.2, 1.5, 'Linear probe = interpretability', fontsize=9, fontfamily='monospace', color='#555')

# Legend
ax.text(0.5, 0.2, 'Input -> Fourier + Peak Embeddings -> 7-layer Graphormer Transformer -> Task-specific Decoder',
        fontsize=9, fontfamily='monospace', color='#333')

plt.savefig('dreams_analysis/fig1_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] fig1_architecture.png')

# ======================================================================
# Fig 2: Graphormer Attention Detail
# ======================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
fig.suptitle('Graphormer Self-Attention Mechanism', fontsize=16, fontweight='bold', y=0.98)

# Formula box
ax.add_patch(plt.Rectangle((1, 6.5), 8, 2, fill=True, facecolor='#fdf2e9', edgecolor='#e67e22', lw=3))
ax.text(5, 8.0, 'alpha_ij = (q_i * k_j) / sqrt(d_k)  +  phi(F_i - F_j)', ha='center', fontsize=16, fontfamily='monospace', fontweight='bold')
ax.text(5, 7.4, 'Standard Dot-Product Attention         Graphormer Term (m/z difference bias)', ha='center', fontsize=11)
ax.text(1.5, 6.9, 'F_i, F_j: Fourier features of peaks i, j', fontsize=9, fontfamily='monospace')
ax.text(1.5, 6.65, 'phi: learnable linear projection (d_fourier -> 1 or n_heads)', fontsize=9, fontfamily='monospace')

# Chemical meaning
ax.text(0.5, 5.8, 'Chemical Interpretation:', fontsize=12, fontweight='bold')
chemical_data = [
    ('18.01 Da', 'H2O loss', 'High positive bias', '#27ae60'),
    ('27.99 Da', 'CO loss', 'High positive bias', '#27ae60'),
    ('17.03 Da', 'NH3 loss', 'Medium positive bias', '#f39c12'),
    ('25.00 Da', 'No chemical meaning', 'Zero or negative bias', '#e74c3c'),
]
for i, (mz, meaning, bias, color) in enumerate(chemical_data):
    y = 5.3 - i * 0.4
    ax.text(1.5, y, f'Delta-m/z = {mz}', fontsize=10, fontfamily='monospace')
    ax.text(4.5, y, f'-> {meaning}', fontsize=10)
    ax.text(7.0, y, f'-> {bias}', fontsize=10, color=color, fontweight='bold')

# Key insight
ax.add_patch(plt.Rectangle((0.5, 2.5), 9, 1.5, fill=True, facecolor='#ebf5fb', edgecolor='#2980b9', lw=2))
ax.text(5, 3.6, 'KEY INSIGHT', ha='center', fontsize=12, fontweight='bold', color='#2980b9')
ax.text(5, 3.2, 'Graphormer encodes chemical fragmentation knowledge directly into the attention mechanism.', ha='center', fontsize=10)
ax.text(5, 2.9, 'Model learns to focus on chemically meaningful peak pairs (neutral losses) without explicit rules.', ha='center', fontsize=10)

# vs chem_aware
ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 1.5, fill=True, facecolor='#fdedec', edgecolor='#e74c3c', lw=2))
ax.text(5, 1.6, 'CHEM_AWARE EXTENSION', ha='center', fontsize=12, fontweight='bold', color='#e74c3c')
ax.text(5, 1.2, 'DreaMS: learns phi from data -> statistical (needs many examples)', ha='center', fontsize=10)
ax.text(5, 0.9, 'ChemAware: encodes phi from chemical rules -> causal (zero-shot, physics-grounded)', ha='center', fontsize=10)

plt.savefig('dreams_analysis/fig2_graphormer.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] fig2_graphormer.png')

# ======================================================================
# Fig 3: Pretraining Objectives
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('DreaMS Pre-training Objectives', fontsize=16, fontweight='bold')

# Mask prediction
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis('off')
ax1.text(5, 9.5, 'Objective 1: Masked Peak Prediction', ha='center', fontsize=13, fontweight='bold', color='#2980b9')
# Spectrum visualization
peaks_x = np.linspace(1, 9, 8)
peaks_y = np.array([0.3, 0.8, 0.5, 0.95, 0.4, 0.7, 0.2, 0.6]) * 5 + 2
for px, py in zip(peaks_x, peaks_y):
    ax1.vlines(px, 2, py, colors='#2980b9', lw=2)
    ax1.scatter(px, py, s=60, c='#2980b9', zorder=5)
# Masked peaks
for i in [2, 5]:
    ax1.vlines(peaks_x[i], 2, peaks_y[i], colors='#e74c3c', lw=3, linestyle='--')
    ax1.scatter(peaks_x[i], peaks_y[i], s=120, c='white', edgecolors='#e74c3c', lw=2, zorder=6)
    ax1.annotate('MASK', xy=(peaks_x[i], peaks_y[i]+0.3), ha='center', fontsize=9, color='#e74c3c', fontweight='bold')
ax1.set_title('Mask 30% of peaks -> Predict missing m/z', fontsize=11)
ax1.text(5, 0.5, 'Classification over binned mass range\nFocal Loss for class imbalance', ha='center', fontsize=9, color='#555')

# Retention order
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis('off')
ax2.text(5, 9.5, 'Objective 2: Retention Order Prediction', ha='center', fontsize=13, fontweight='bold', color='#27ae60')
ax2.text(5, 8.5, 'Spectrum A (masked)', ha='center', fontsize=10, fontweight='bold')
ax2.text(5, 5.5, 'Spectrum B (masked)', ha='center', fontsize=10, fontweight='bold')
ax2.annotate('', xy=(5, 6), xytext=(5, 8), arrowprops=dict(arrowstyle='->', lw=2))
ax2.annotate('', xy=(5, 6), xytext=(5, 5), arrowprops=dict(arrowstyle='<-', lw=2))
ax2.text(5, 6.5, 'Concat Precursor Embs', ha='center', fontsize=9)
ax2.text(5, 4.5, 'Binary Classifier: A before B?', ha='center', fontsize=10, fontweight='bold')
ax2.text(5, 3.5, 'Learns molecular POLARITY\n(orthogonal to fragmentation chemistry)', ha='center', fontsize=9, color='#555')
ax2.set_title('Classify elution order from 2 spectra', fontsize=11)

plt.savefig('dreams_analysis/fig3_pretraining.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] fig3_pretraining.png')

# ======================================================================
# Fig 4: ChemAware vs DreaMS Comparison
# ======================================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
fig.suptitle('DreaMS Graphormer vs ChemAware ChemicalRuleEngine', fontsize=16, fontweight='bold')

# DreaMS side
ax.add_patch(plt.Rectangle((0.2, 0.5), 4.6, 8.5, fill=True, facecolor='#ebf5fb', edgecolor='#2980b9', lw=3))
ax.text(2.5, 8.5, 'DreaMS Graphormer', ha='center', fontsize=14, fontweight='bold', color='#2980b9')
ax.text(2.5, 7.5, 'Learns m/z diff -> attention bias', ha='center', fontsize=10)
ax.text(2.5, 7.0, 'from 700M spectra', ha='center', fontsize=10)
ax.text(2.5, 6.0, 'PROS:', fontsize=11, fontweight='bold', color='#27ae60')
for i, p in enumerate(['Data-driven, adapts to any fragmentation', 'Learns rare patterns from stats', 'Proven SOTA performance', 'No manual rule curation needed']):
    ax.text(0.5, 5.5-i*0.4, f'+ {p}', fontsize=8)
ax.text(2.5, 3.5, 'CONS:', fontsize=11, fontweight='bold', color='#e74c3c')
for i, c in enumerate(['Linear projection only (monotonic)', 'Needs many examples for rare cases', 'No causal explanation (why?)', 'Cannot verify chemical plausibility']):
    ax.text(0.5, 3.0-i*0.4, f'- {c}', fontsize=8)

# ChemAware side
ax.add_patch(plt.Rectangle((5.2, 0.5), 4.6, 8.5, fill=True, facecolor='#fdedec', edgecolor='#e74c3c', lw=3))
ax.text(7.5, 8.5, 'ChemAware RuleEngine', ha='center', fontsize=14, fontweight='bold', color='#e74c3c')
ax.text(7.5, 7.5, 'Encodes chemical rules -> bias', ha='center', fontsize=10)
ax.text(7.5, 7.0, 'explicit, physics-grounded', ha='center', fontsize=10)
ax.text(7.5, 6.0, 'PROS:', fontsize=11, fontweight='bold', color='#27ae60')
for i, p in enumerate(['Zero-shot (no data needed)', 'Causal explanation (knows WHY)', 'Can verify/correct attention', 'Works for rare/unseen patterns']):
    ax.text(5.5, 5.5-i*0.4, f'+ {p}', fontsize=8)
ax.text(7.5, 3.5, 'CONS:', fontsize=11, fontweight='bold', color='#e74c3c')
for i, c in enumerate(['Rule coverage limited (~20 NL, ~22 CF)', 'Cannot discover new patterns', 'Manual curation needed', 'Redundant with well-trained Graphormer']):
    ax.text(5.5, 3.0-i*0.4, f'- {c}', fontsize=8)

# Bridge
ax.annotate('', xy=(5.2, 5), xytext=(4.8, 5), arrowprops=dict(arrowstyle='<->', lw=2, color='#8e44ad'))
ax.text(5, 5.5, 'A-B\nAdversarial', ha='center', fontsize=8, fontweight='bold', color='#8e44ad')
ax.text(5, 1.5, 'Best of both:\nGraphormer learns stats\n+ Rules verify & explain', ha='center', fontsize=9, fontweight='bold', color='#333')

plt.savefig('dreams_analysis/fig4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] fig4_comparison.png')

print('\nAll 4 figures generated in dreams_analysis/')
