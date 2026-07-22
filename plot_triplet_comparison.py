"""triplet_comparison.png 的生成脚本。用法：python plot_triplet_comparison.py"""
import re, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(path):
    steps, aucs, masks, contras, press = [], [], [], [], []
    pending_step = None
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m0 = re.search(r'AUC Check @ Step (\d+)', line)
            if m0: pending_step = int(m0.group(1)); continue
            if pending_step and 'AUC=' in line:
                m = re.search(r'AUC=([\d.]+)', line)
                if m: steps.append(pending_step); aucs.append(float(m.group(1)))
                pending_step = None; continue
            m3 = re.search(r'mask=([\d.]+) contra=([\d.]+) pres=([\d.]+)', line)
            if m3: masks.append(float(m3.group(1))); contras.append(float(m3.group(2))); press.append(float(m3.group(3)))
    return steps, aucs, masks, contras, press

s_old, a_old, m_old, c_old, p_old = parse_log('chemaware_experiments/triplet_2304669.out')
s_new, a_new, m_new, c_new, p_new = parse_log('chemaware_experiments/triplet_2308525.out')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(s_old, a_old, 'b-o', ms=5, label='Old (th=0.3/0.1, NL+CF+ISO, 5 epochs)')
ax.plot(s_new, a_new, 'r-o', ms=5, label='New (th=0.23/0.09, +HR, 2 epochs)')
ax.axhline(0.8828, color='gray', ls='--', label='Baseline (0.883)')
ax.set_xlabel('Step'); ax.set_ylabel('AUC'); ax.set_title('Held-out Retrieval AUC')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
xs_om = np.arange(0, len(m_old)*10, 10)[:len(m_old)]
xs_nm = np.arange(0, len(m_new)*10, 10)[:len(m_new)]
w = 200
ax.plot(xs_om[w-1:], np.convolve(m_old, np.ones(w)/w, mode='valid'), 'b-', lw=1.5, label='Mask loss (old)')
ax.plot(xs_nm[w-1:], np.convolve(m_new, np.ones(w)/w, mode='valid'), 'r-', lw=1.5, label='Mask loss (new)')
ax.plot(xs_om[w-1:], np.convolve(c_old, np.ones(w)/w, mode='valid'), 'b--', lw=1.5, label='Triplet loss (old)')
ax.plot(xs_nm[w-1:], np.convolve(c_new, np.ones(w)/w, mode='valid'), 'r--', lw=1.5, label='Triplet loss (new)')
ax.set_xlabel('Step'); ax.set_ylabel('Loss (smoothed)'); ax.set_title('Loss Curves')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('chemaware_experiments/triplet_comparison.png', dpi=150)
print('Saved: chemaware_experiments/triplet_comparison.png')
