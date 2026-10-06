# GLM figure: metric split — Top-1 retrieval vs pooled pairwise discrimination.
# Every point is read from the verified ladder CSV (dual-source cross-checked);
# no number is hand-typed.  Outputs svg/pdf/600dpi png.
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LADDER = Path("deliverables/GLM_gnps_article_ladder/run15/ladder_full.csv")
OUT = Path("deliverables/figures")
INK, GRAY, GREEN, BLUE, RED, ORANGE = ("#1a1a1a", "#9a9a9a", "#1e8449",
                                       "#1f618d", "#c0392b", "#b9770e")

ladder = pd.read_csv(LADDER)
GROUPS = {
    "official_dreams": (INK, "o", "official DreaMS"),
    "noise_v1": (RED, "*", "noise V1 (ours)"),
    "cosine_greedy": (GRAY, "o", "cosine"),
    "modified_cosine": (GRAY, "o", "modified cosine"),
    "weighted_spectral_entropy": (GREEN, "o", "weighted spectral entropy"),
    "p2b_sqrt_cosine": (BLUE, "s", "sqrt-cosine (P2b)"),
    "p2b_unweighted_entropy": (BLUE, "s", "unweighted entropy (P2b)"),
    "neutral_loss_sqrt_cosine": (BLUE, "s", "neutral-loss sqrt-cosine"),
    "p2b_official_frozen": (BLUE, "D", "P2b frozen on official"),
    "p2b_noise_v1_frozen": (BLUE, "D", "P2b frozen on noise V1"),
    "ms2deepscore_2x_public": (GRAY, "^", "MS2DeepScore 2.x (public)"),
    "spec2vec_gnps_public": (GRAY, "^", "Spec2Vec 2020 (public)"),
    "entropy_raw_public": (GREEN, "v", "raw entropy (public)"),
    "denoising_search_public": (GREEN, "P", "denoising search (public)"),
    "spec2vec_2026_retrained": (ORANGE, "^", "Spec2Vec 2026 (public)"),
}
LABEL_DX = {"noise_v1": (6, 4), "weighted_spectral_entropy": (-8, 6),
            "official_dreams": (6, -10), "spec2vec_gnps_public": (6, 3),
            "ms2deepscore_2x_public": (6, -3), "cosine_greedy": (6, 2)}

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
for ax, panel, title in zip(
        axes, ("identity_disjoint", "formula_disjoint"),
        ("identity-disjoint (n=10,995)", "formula-disjoint (n=5,261)")):
    sub = ladder[ladder.panel == panel]
    for _, row in sub.iterrows():
        color, marker, label = GROUPS[row["method"]]
        size = 220 if row["method"] == "noise_v1" else 70
        ax.scatter(row["recall@1"] * 100, row["gnps_10ppm_pooled_pairwise_auroc"],
                   c=color, marker=marker, s=size, zorder=5,
                   edgecolors="white", linewidths=0.8,
                   alpha=1.0 if row["method"] in ("noise_v1", "official_dreams",
                                                  "weighted_spectral_entropy") else 0.85)
        if row["method"] in LABEL_DX:
            dx, dy = LABEL_DX[row["method"]]
            ax.annotate(label, (row["recall@1"] * 100,
                                row["gnps_10ppm_pooled_pairwise_auroc"]),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=8.2, color=color,
                        ha="left" if dx > 0 else "right")
    ax.set_xlabel("Recall@1 (%) — identification", fontsize=9.5)
    ax.set_ylabel("pooled pairwise AUROC — discrimination", fontsize=9.5)
    ax.set_title(title, fontsize=10)
    ax.grid(color="#eeeeee", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # annotate the split: rank-1 method on each axis
    best_r1 = sub.loc[sub["recall@1"].idxmax()]
    best_auc = sub.loc[sub["gnps_10ppm_pooled_pairwise_auroc"].idxmax()]
    if best_r1["method"] != best_auc["method"]:
        ax.annotate("", xy=(best_r1["recall@1"] * 100,
                            best_r1["gnps_10ppm_pooled_pairwise_auroc"]),
                    xytext=(best_auc["recall@1"] * 100,
                            best_auc["gnps_10ppm_pooled_pairwise_auroc"]),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0, ls="--"))
fig.suptitle("Metric split on the sealed GNPS benchmark: the best discriminator "
             "is not the best identifier (15 methods, verified numbers)",
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"GLM_gnps_metric_split.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight",
                facecolor="white")
print("written:", OUT / "GLM_gnps_metric_split.{svg,pdf,png}")
