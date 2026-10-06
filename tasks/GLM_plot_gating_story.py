# GLM figure: why confidence gates instead of routes.
# Panel A: risk-coverage curves (accept the top-c% most confident queries).
# Panel B: the oracle headroom hides where winners are least confident --
# histogram of winning methods' gap percentiles on recoverable queries vs
# the gap percentile of WSE on queries it answers correctly.
# All numbers recomputed from the frozen 15-method bundle (run15).
from pathlib import Path
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats  # noqa: E402

OUT = Path("deliverables/figures")
INK, GRAY, GREEN, RED, BLUE = "#1a1a1a", "#9a9a9a", "#1e8449", "#c0392b", "#1f618d"

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

# ---- Panel A: risk-coverage ------------------------------------------------
ax = axes[0]
styles = {
    "weighted_spectral_entropy": (GREEN, "-", "WSE · identity"),
    "noise_v1": (RED, "-", "noise V1 (ours) · identity"),
    "weighted_spectral_entropy_f": (GREEN, "--", "WSE · formula"),
    "noise_v1_f": (RED, "--", "noise V1 (ours) · formula"),
}
data = {}
for panel in ("identity_disjoint", "formula_disjoint"):
    g, methods, mol = load_panel(panel)
    rank, gap, top1 = per_query_stats(g, mol)
    data[panel] = (methods, rank, gap)

for panel, suffix in (("identity_disjoint", ""), ("formula_disjoint", "_f")):
    methods, rank, gap = data[panel]
    for m in ("weighted_spectral_entropy", "noise_v1"):
        mi = methods.index(m)
        order = np.argsort(-gap[mi], kind="stable")
        c_sorted = (rank[mi] == 1)[order]
        n = len(c_sorted)
        covs = np.arange(1, n + 1) / n * 100
        accs = np.cumsum(c_sorted) / np.arange(1, n + 1) * 100
        thin = np.concatenate([np.arange(0, n, max(1, n // 400)), [n - 1]])
        color, ls, label = styles[m + suffix]
        ax.plot(covs[thin], accs[thin], color=color, ls=ls, lw=1.8, label=label)
ax.axhline(99.0, color=GRAY, lw=0.8, ls=":")
ax.axhline(99.5, color=GRAY, lw=0.8, ls=":")
ax.text(99, 99.06, " 99%", fontsize=7.5, color=GRAY, ha="right")
ax.text(99, 99.56, " 99.5%", fontsize=7.5, color=GRAY, ha="right")
ax.set_xlabel("coverage (% of queries answered)", fontsize=9.5)
ax.set_ylabel("Top-1 accuracy among answered", fontsize=9.5)
ax.set_title("A · truth-blind confidence gates quality in", fontsize=10)
ax.set_xlim(30, 100.5); ax.set_ylim(86, 100.2)
ax.legend(fontsize=7.6, loc="lower left", frameon=False)
ax.grid(color="#eeeeee", lw=0.6)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# ---- Panel B: where the oracle headroom lives ------------------------------
ax = axes[1]
methods, rank, gap = data["identity_disjoint"]
N = rank.shape[1]
correct = rank == 1
wse_i = methods.index("weighted_spectral_entropy")
oracle = correct.any(axis=0)
recoverable = (~correct[wse_i]) & oracle
win_pct = []
for i in np.flatnonzero(recoverable):
    for mi in range(len(methods)):
        if correct[mi, i]:
            win_pct.append(np.argsort(np.argsort(gap[mi]))[i] / (N - 1))
win_pct = np.asarray(win_pct)
wse_correct_pct = (np.argsort(np.argsort(gap[wse_i]))[correct[wse_i]]
                   / (N - 1))
bins = np.linspace(0, 1, 21)
ax.hist(wse_correct_pct, bins=bins, color=GREEN, alpha=0.55,
        label=f"WSE when correct (n={len(wse_correct_pct):,})", density=True)
ax.hist(win_pct, bins=bins, color=RED, alpha=0.65,
        label=f"winning method on recoverable queries (n={len(win_pct):,})",
        density=True)
ax.axvline(win_pct.mean(), color=RED, lw=1.2, ls="--")
ax.text(win_pct.mean() + 0.02, ax.get_ylim()[1] * 0.92,
        f"mean {win_pct.mean():.2f}", fontsize=8, color=RED)
ax.set_xlabel("the method's own gap percentile (batch)", fontsize=9.5)
ax.set_ylabel("density", fontsize=9.5)
ax.set_title("B · oracle headroom hides where winners are least confident",
             fontsize=10)
ax.legend(fontsize=7.6, frameon=False)
ax.grid(color="#eeeeee", lw=0.6)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.suptitle("Confidence should gate, not route (sealed GNPS benchmark, "
             "15 methods)", fontsize=11.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"GLM_gating_story.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight",
                facecolor="white")
print("written:", OUT / "GLM_gating_story.{svg,pdf,png}")
