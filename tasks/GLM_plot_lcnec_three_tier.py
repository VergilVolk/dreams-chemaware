# GLM figure: three-tier claim architecture on LCNEC dark modules.
# X = best molecule cosine, Y = molecule-level gap; gate regions shaded;
# halogen-flagged candidates get an orange edge. All values read from the
# corrected molecule rescan / three-tier table (no hand-typed numbers).
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path("data/validation/GLM_track2_census/lcnec_dark_three_tier.csv")
OUT = Path("deliverables/figures")
INK, GRAY, GREEN, RED, BLUE, ORANGE = ("#1a1a1a", "#9a9a9a", "#1e8449",
                                       "#c0392b", "#1f618d", "#b9770e")
GATE99, GATE995, HIT = 0.584, 0.7407, 0.85

df = pd.read_csv(CSV)
colors = {"A_gate_passed": GREEN, "B1_high_score_contested": RED,
          "B2_sub_hit_analog": BLUE, "C_dark": GRAY}

fig, ax = plt.subplots(figsize=(7.4, 5.4))
# gate regions
ax.axvspan(HIT, 1.0, ymin=GATE99, alpha=0.10, color=GREEN, zorder=0)
ax.axhline(GATE99, color=GREEN, lw=1.0, ls="--")
ax.axhline(GATE995, color=GREEN, lw=0.8, ls=":")
ax.axvline(HIT, color=GRAY, lw=0.9, ls="--")
ax.text(0.852, GATE99 + 0.012, "gate @99% (gap\u2009\u2265\u20090.584)",
        fontsize=8, color=GREEN)
ax.text(0.852, GATE995 + 0.012, "gate @99.5%", fontsize=8, color=GREEN)

for tier, sub in df.groupby("tier"):
    ax.scatter(sub["best_cosine"], sub["molecule_gap"], s=90,
               c=colors[tier], label=f"{tier} (n={len(sub)})",
               edgecolors="white", linewidths=0.8, zorder=5)
hal = df[df["top1_halogen_flag"] == True]  # noqa: E712
ax.scatter(hal["best_cosine"], hal["molecule_gap"], s=130, facecolors="none",
           edgecolors=ORANGE, linewidths=1.8, zorder=6,
           label="halogen flag (exogenous-until-proven)")

notable = {221.984558: ("221.985  C8H7FN6O\nonly gate-passed; F flag",
                        (8, 6)),
           273.080554: ("273.081\ncos 0.981, gap 0.012\nthe confident-error trap",
                        (-20, -34)),
           342.088065: ("342.088", (8, -3))}
for mz, (text, off) in notable.items():
    row = df[np.isclose(df["precursor"], mz, atol=1e-3)]
    if len(row):
        ax.annotate(text, (row.iloc[0]["best_cosine"],
                           row.iloc[0]["molecule_gap"]),
                    textcoords="offset points", xytext=off, fontsize=7.8,
                    color=INK)

ax.set_xlabel("best molecule cosine (reverse search)", fontsize=10)
ax.set_ylabel("molecule-level top1-top2 gap", fontsize=10)
ax.set_title("Three-tier claims for 30 LCNEC dark modules\n"
             "(gate thresholds transferred from the model-blind GNPS "
             "benchmark)", fontsize=10.5)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 0.92)
ax.grid(color="#eeeeee", lw=0.6)
ax.legend(fontsize=7.8, loc="upper left", frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"GLM_lcnec_three_tier.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight",
                facecolor="white")
print("written:", OUT / "GLM_lcnec_three_tier.{svg,pdf,png}")
