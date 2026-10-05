"""Generate the code-native six-panel figure for report Sections 2.3 and 2.4.

All plotted values are fixed from the report text.  The figure deliberately
separates held-out spectral-fusion evidence, development-set reranking evidence,
and same-domain embedding fine-tuning evidence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"

BURGUNDY = "#92283B"
BURGUNDY_DARK = "#741C2D"
BURGUNDY_LIGHT = "#D8A5AE"
GRAY = "#B9B9B9"
GRAY_DARK = "#666666"
GRID = "#E2E2E2"
BLACK = "#151515"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def rounded_box(
    ax: plt.Axes,
    xy,
    wh,
    text,
    edge,
    face="#FAFAFA",
    lw=1.2,
    fs=8.5,
    text_color=BLACK,
    text_weight="normal",
):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fs,
        color=text_color,
        fontweight=text_weight,
    )
    return box


def panel_a(ax: plt.Axes) -> None:
    panel_label(ax, "a")
    ax.set_title("Spectral evidence fusion", loc="left", pad=18)
    ax.text(0.0, 1.02, "ordinary MS/MS input; no candidate structure required", transform=ax.transAxes, fontsize=8.2)
    ax.set_axis_off()

    # Deterministic schematic spectrum (not experimental data).
    mz = np.array([0.03, 0.07, 0.10, 0.14, 0.19, 0.23, 0.27, 0.31, 0.36, 0.40, 0.45])
    intensity = np.array([0.20, 0.37, 0.74, 0.25, 1.00, 0.46, 0.70, 0.31, 0.89, 0.24, 0.52])
    base_x, base_y, width, height = 0.00, 0.25, 0.36, 0.50
    ax.plot([base_x, base_x + width], [base_y, base_y], color=BLACK, lw=0.9, transform=ax.transAxes, clip_on=False)
    ax.plot([base_x, base_x], [base_y, base_y + height], color=BLACK, lw=0.9, transform=ax.transAxes, clip_on=False)
    for i, (x, h) in enumerate(zip(mz, intensity)):
        color = BURGUNDY if i in {2, 4, 6, 8} else GRAY_DARK
        ax.plot([base_x + x, base_x + x], [base_y, base_y + 0.43 * h], color=color, lw=1.7 if color == BURGUNDY else 1.0, transform=ax.transAxes)
    ax.text(0.18, 0.18, "Query MS/MS spectrum", transform=ax.transAxes, ha="center", fontsize=8.5)

    boxes = [
        ((0.47, 0.68), "DreaMS\nglobal similarity", GRAY_DARK, "#F5F5F5"),
        ((0.47, 0.42), "Neutral-loss\nsimilarity", BURGUNDY, "#FFF7F8"),
        ((0.47, 0.16), "Spectral\nentropy", GRAY_DARK, "#F5F5F5"),
    ]
    for (x, y), label, edge, face in boxes:
        rounded_box(ax, (x, y), (0.28, 0.17), label, edge=edge, face=face, lw=1.5 if edge == BURGUNDY else 1.0)
        ax.add_patch(FancyArrowPatch((0.36, 0.50), (x, y + 0.085), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=GRAY_DARK))

    rounded_box(
        ax,
        (0.83, 0.36),
        (0.16, 0.30),
        "Candidate\nrank fusion",
        edge=BURGUNDY_DARK,
        face=BURGUNDY,
        lw=1.2,
        fs=8.6,
        text_color="white",
        text_weight="bold",
    )
    for _, y in [(0, 0.765), (0, 0.505), (0, 0.245)]:
        ax.add_patch(FancyArrowPatch((0.75, y), (0.83, 0.51), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=GRAY_DARK))


def panel_b(ax: plt.Axes) -> None:
    panel_label(ax, "b")
    ax.set_title("Spectral fusion benchmarks", loc="left", pad=18)
    cohorts = ["Development\nn = 5,037", "Held-out\nn = 3,000", "Near structure\nn = 496"]
    official = np.array([86.06, 87.93, 48.79])
    fusion = np.array([89.97, 89.00, 44.56])
    delta = fusion - official
    y = np.arange(3)[::-1]
    ax.hlines(y, np.minimum(official, fusion), np.maximum(official, fusion), color=GRAY_DARK, lw=1.2, zorder=1)
    ax.scatter(official, y, s=55, color=GRAY, edgecolor=GRAY_DARK, linewidth=0.7, label="Official", zorder=3)
    ax.scatter(fusion, y, s=58, color=BURGUNDY, edgecolor=BURGUNDY_DARK, linewidth=0.7, label="Spectral fusion", zorder=3)
    for i in range(3):
        ax.text(official[i], y[i] + 0.16, f"{official[i]:.2f}", ha="center", va="bottom", fontsize=8.0)
        ax.text(fusion[i], y[i] - 0.18, f"{fusion[i]:.2f}", ha="center", va="top", fontsize=8.0, color=BURGUNDY_DARK, fontweight="bold")
        sign = "+" if delta[i] >= 0 else "−"
        ax.text(max(official[i], fusion[i]) + 1.8, y[i], f"{sign}{abs(delta[i]):.2f} pp", va="center", fontsize=8.2, fontweight="bold", color=BURGUNDY_DARK if delta[i] >= 0 else GRAY_DARK)
    ax.set_yticks(y, cohorts)
    ax.set_xlim(39, 97)
    ax.set_xlabel("Recall@1 (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False, ncol=2)


def panel_c(ax: plt.Axes) -> None:
    panel_label(ax, "c")
    ax.set_title("Paired rank changes", loc="left", pad=18)
    labels = ["Development", "Held-out", "Near structure"]
    corrected = np.array([280, 89, 20])
    introduced = np.array([83, 57, 41])
    y = np.arange(3)[::-1]
    ax.barh(y, -introduced, height=0.42, color=GRAY, edgecolor="none", label="Introduced")
    ax.barh(y, corrected, height=0.42, color=BURGUNDY, edgecolor="none", label="Corrected")
    for yy, c, i in zip(y, corrected, introduced):
        ax.text(c + 7, yy, str(c), va="center", fontsize=8.5)
        ax.text(-i - 7, yy, str(i), va="center", ha="right", fontsize=8.5)
    ax.axvline(0, color=BLACK, lw=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlim(-110, 315)
    ax.set_xlabel("Number of queries")
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False, ncol=2)
    ax.text(0.98, 0.42, "Held-out 95% CI\n+0.24 to +1.89 pp", transform=ax.transAxes, ha="right", va="center", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7F8", ec=BURGUNDY_LIGHT, lw=0.8))


def panel_d(ax: plt.Axes) -> None:
    panel_label(ax, "d")
    ax.set_title("Fragmentation-rule evidence asset", loc="left", pad=18)
    ax.set_axis_off()
    ax.text(0.50, 0.88, "16,272 candidate rules", transform=ax.transAxes, ha="center", fontsize=11.5, fontweight="bold")
    ax.text(0.50, 0.79, "from 139,240 MassBank 2026.03 records", transform=ax.transAxes, ha="center", fontsize=8.8)
    left, bottom, total_width, height = 0.05, 0.56, 0.90, 0.17
    frac = 9396 / 16272
    ax.add_patch(Rectangle((left, bottom), total_width * frac, height, transform=ax.transAxes, facecolor=BURGUNDY, edgecolor="none"))
    ax.add_patch(Rectangle((left + total_width * frac, bottom), total_width * (1 - frac), height, transform=ax.transAxes, facecolor=BURGUNDY_LIGHT, edgecolor="none"))
    ax.text(left + total_width * frac / 2, bottom + height / 2, "9,396\nproduct-ion rules", transform=ax.transAxes, ha="center", va="center", color="white", fontweight="bold", fontsize=8.8)
    ax.text(left + total_width * (frac + (1 - frac) / 2), bottom + height / 2, "6,876\nneutral-loss rules", transform=ax.transAxes, ha="center", va="center", fontsize=8.8)
    rounded_box(ax, (0.11, 0.28), (0.32, 0.14), "Formula-disjoint\ndevelopment", edge=GRAY_DARK, face="#F5F5F5")
    rounded_box(ax, (0.57, 0.28), (0.32, 0.14), "Frozen formula-held\nconfirmation", edge=GRAY_DARK, face="#F5F5F5")
    ax.add_patch(FancyArrowPatch((0.43, 0.35), (0.57, 0.35), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10, lw=0.9, color=GRAY_DARK))
    ax.text(0.50, 0.15, "mass-position and structure-condition permutation controls; CIs > 0", transform=ax.transAxes, ha="center", fontsize=7.9)
    ax.text(0.50, 0.05, "Evidence qualification; not an end-to-end accuracy estimate", transform=ax.transAxes, ha="center", fontsize=7.8, style="italic", color=GRAY_DARK)


def panel_e(ax: plt.Axes) -> None:
    panel_label(ax, "e")
    ax.set_title("Structure-conditioned candidate reranking", loc="left", pad=18)
    vals = [90.46, 94.40]
    x = np.array([0, 1])
    ax.bar(x, vals, color=[GRAY, BURGUNDY], width=0.62, edgecolor=[GRAY_DARK, BURGUNDY_DARK], linewidth=0.7)
    for xx, v in zip(x, vals):
        ax.text(xx, v + 0.22, f"{v:.2f}", ha="center", fontsize=9.0, fontweight="bold")
    ax.plot([0, 0, 1, 1], [95.20, 95.45, 95.45, 95.20], color=BLACK, lw=0.8)
    ax.text(0.5, 95.55, "+3.94 pp [2.82, 5.19]", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax.set_ylim(88.5, 96.2)
    ax.set_ylabel("Recall@1 (%)")
    ax.set_xticks(x, ["Official", "ChemAware\nreranker"])
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.text(0.50, -0.20, "n = 1,929   •   93 corrected / 17 introduced", transform=ax.transAxes, ha="center", fontsize=8.2)
    ax.text(0.50, -0.31, "True mapping vs candidate-rotated control:\n+3.32 pp [2.31, 4.38]", transform=ax.transAxes, ha="center", fontsize=8.0, color=BURGUNDY_DARK)


def panel_f(ax: plt.Axes) -> None:
    panel_label(ax, "f")
    ax.set_title("Chemical information enters two ways", loc="left", pad=18)
    labels = [
        "Candidate reranking\n(development)",
        "Native hard-triplet\nfine-tuning\nn = 1,929",
        "Max-reference boundary\nalignment\nn = 1,975",
    ]
    means = np.array([3.94, 1.81, 2.13])
    lo = np.array([2.82, 0.82, 1.28])
    hi = np.array([5.19, 2.87, 3.03])
    colors = [BURGUNDY_DARK, BURGUNDY, BURGUNDY_LIGHT]
    y = np.arange(3)[::-1]
    for yy, m, low, high, color in zip(y, means, lo, hi, colors):
        ax.errorbar(m, yy, xerr=[[m - low], [high - m]], fmt="s", markersize=7.5, color=color, ecolor=color, elinewidth=1.2, capsize=3, capthick=1.0)
        ax.text(high + 0.18, yy, f"+{m:.2f} [{low:.2f}, {high:.2f}]", va="center", fontsize=8.0)
    ax.axvline(0, color=BLACK, lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 6.4)
    ax.set_xlabel("Recall@1 improvement (percentage points)")
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.text(0.00, -0.23, r"$\bf{Reranking:}$ structure available at inference", transform=ax.transAxes, fontsize=7.9)
    ax.text(0.00, -0.32, r"$\bf{Fine\!\!\!-tuning:}$ ordinary MS/MS only at inference", transform=ax.transAxes, fontsize=7.9)


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.0, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.94, bottom=0.095, wspace=0.34, hspace=0.56)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]
    for draw, ax in zip((panel_a, panel_b, panel_c, panel_d, panel_e, panel_f), axes):
        draw(ax)

    stem = OUT / "figure_2_3_2_4_spectral_chemical_evidence_code"
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
