from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


OUT_DIR = Path(__file__).resolve().parent

INK = "#202020"
MID = "#777777"
LIGHT = "#D8D8D8"
PALE = "#F2F2F2"
ACCENT = "#8B2F3C"
ACCENT_LIGHT = "#C98E96"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2.5, color=INK)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def draw_peak_attribution(ax: plt.Axes) -> None:
    mz = np.array(
        [54, 67, 73, 81, 91, 105, 117, 129, 143, 157, 171, 185, 199, 213,
         229, 243, 259, 273, 289, 305, 321, 337, 353, 371, 389, 407, 425]
    )
    intensity = np.array(
        [0.12, 0.27, 0.08, 0.43, 0.22, 0.74, 0.16, 0.35, 0.92, 0.18, 0.54,
         0.31, 0.14, 0.66, 0.25, 0.47, 0.19, 0.84, 0.11, 0.38, 0.29, 0.57,
         0.17, 0.41, 0.23, 0.13, 0.08]
    )
    response = np.array(
        [-0.02, 0.01, 0.00, 0.03, -0.01, 0.12, 0.01, -0.03, 0.18, 0.00,
         0.04, -0.02, 0.01, 0.15, -0.01, 0.02, -0.04, 0.10, 0.00, -0.02,
         0.01, 0.06, -0.01, 0.03, -0.02, 0.00, 0.01]
    )
    selected = response >= 0.10

    ax.vlines(mz[~selected], 0, intensity[~selected], color=MID, lw=1.0)
    ax.vlines(mz[selected], 0, intensity[selected], color=ACCENT, lw=1.8)
    ax.scatter(mz[selected], intensity[selected] + 0.035, marker="v", s=12, color=ACCENT,
               clip_on=False, zorder=3)
    ax.set_xlim(40, 440)
    ax.set_ylim(-0.18, 1.02)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("Peak-wise error attribution", loc="left", pad=5, fontweight="bold")
    ax.text(0.98, 0.96, "schematic", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=MID, style="italic")

    heat_ax = ax.inset_axes([0.0, 0.02, 1.0, 0.075])
    cmap = LinearSegmentedColormap.from_list("response", ["#A7A7A7", "#F6F6F6", ACCENT])
    heat_ax.imshow(response[np.newaxis, :], aspect="auto", cmap=cmap, vmin=-0.18, vmax=0.18,
                   extent=(mz.min(), mz.max(), 0, 1))
    heat_ax.set_xlim(40, 440)
    heat_ax.set_xticks([])
    heat_ax.set_yticks([])
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    clean_axis(ax)
    panel_label(ax, "a")


def draw_intervention_benchmark(container: plt.Axes) -> None:
    container.axis("off")
    panel_label(container, "b")
    container.set_title("Frozen intervention benchmark", loc="left", pad=5, fontweight="bold")
    gs = container.get_subplotspec().subgridspec(1, 2, width_ratios=[1.05, 1], wspace=0.48)

    ax = container.figure.add_subplot(gs[0, 0])
    rates = [4.52, 10.74]
    ax.bar([0, 1], rates, width=0.58, color=[LIGHT, ACCENT], edgecolor=INK, linewidth=0.55)
    ax.set_xticks([0, 1], ["Matched\nrandom", "Error-\nguided"])
    ax.set_ylabel("Errors corrected (%)")
    ax.set_ylim(0, 12.6)
    ax.set_yticks([0, 4, 8, 12])
    ax.axhline(0, color=INK, lw=0.6)
    for x, y in enumerate(rates):
        ax.text(x, y + 0.35, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5)
    clean_axis(ax)

    ax2 = container.figure.add_subplot(gs[0, 1])
    estimate, lo, hi = 0.0305, 0.0229, 0.0386
    ax2.axvline(0, color=LIGHT, lw=0.8)
    ax2.errorbar(estimate, 0, xerr=[[estimate - lo], [hi - estimate]], fmt="o",
                 color=ACCENT, ecolor=ACCENT, capsize=2.5, ms=4.3, lw=1.1)
    ax2.set_xlim(-0.004, 0.044)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.set_xlabel("Net margin change")
    ax2.set_xticks([0.00, 0.02, 0.04])
    ax2.text(0.5, 0.78, "formula-cluster 95% CI", transform=ax2.transAxes,
             ha="center", va="center", fontsize=6.1, color=MID)
    ax2.text(0.98, 0.04, "n = 551", transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=6.1, color=MID)
    clean_axis(ax2, left=False)


def draw_internal_benchmark(ax: plt.Axes) -> None:
    values = np.array([93.1875, 93.6835, 94.2399])
    x = np.arange(3)
    colors = [MID, ACCENT_LIGHT, ACCENT]
    ax.plot(x, values, color=INK, lw=0.85, zorder=1)
    ax.scatter(x, values, s=[28, 34, 38], c=colors, edgecolor=INK, linewidth=0.55, zorder=2)
    ax.set_xticks(x, ["Official", "Stage-1", "V1"])
    ax.set_ylim(92.95, 94.45)
    ax.set_yticks([93.0, 93.5, 94.0, 94.5])
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Internal held benchmark", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", lw=0.55, zorder=0)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + 0.09, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.3)
    ax.text(0.98, 0.05, "18,333 queries", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.1, color=MID)
    clean_axis(ax)
    panel_label(ax, "c")


def draw_transition_heatmaps(container: plt.Axes) -> None:
    container.axis("off")
    panel_label(container, "d")
    container.set_title("Paired rank transitions", loc="left", pad=5, fontweight="bold")
    gs = container.get_subplotspec().subgridspec(1, 2, wspace=0.30)
    matrices = [
        np.array([[98.72, 1.28], [24.74, 75.26]]),
        np.array([[98.92, 1.08], [24.87, 75.13]]),
    ]
    titles = ["Official → Stage-1", "Stage-1 → V1"]
    count_notes = ["309 corrected\n218 introduced", "288 corrected\n186 introduced"]
    cmap = LinearSegmentedColormap.from_list("grayred", ["#F5F5F5", "#BDBDBD", "#565656"])
    for j, (matrix, title, note) in enumerate(zip(matrices, titles, count_notes)):
        ax = container.figure.add_subplot(gs[0, j])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        ax.set_xticks([0, 1], ["Correct", "Wrong"], rotation=25, ha="right")
        ax.set_yticks([0, 1], ["Correct", "Wrong"] if j == 0 else ["", ""])
        if j == 0:
            ax.set_ylabel("Before")
        ax.set_title(title, fontsize=7, pad=4)
        for row in range(2):
            for col in range(2):
                color = "white" if matrix[row, col] > 60 else INK
                ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center",
                        fontsize=6.4, color=color)
        ax.text(0.5, -0.34, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.0, color=ACCENT if j == 1 else MID)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)


def draw_external_benchmark(ax: plt.Axes) -> None:
    labels = ["Identity", "Formula", "Identity near", "Formula near"]
    official = np.array([85.357, 86.809, 79.939, 77.989])
    stage1 = np.array([86.066, 87.797, 80.822, 79.416])
    v1 = np.array([86.558, 88.063, 81.441, 79.959])
    y = np.arange(len(labels))
    h = 0.19
    ax.barh(y + h, official, height=h, color=LIGHT, edgecolor=INK, linewidth=0.45, label="Official")
    ax.barh(y, stage1, height=h, color=ACCENT_LIGHT, edgecolor=INK, linewidth=0.45, label="Stage-1")
    ax.barh(y - h, v1, height=h, color=ACCENT, edgecolor=INK, linewidth=0.45, label="V1")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(76, 90)
    ax.set_xticks([76, 80, 84, 88])
    ax.set_xlabel("GNPS Recall@1 (%)")
    ax.set_title("Independent GNPS benchmark", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.55, zorder=0)
    ax.legend(frameon=False, ncol=1, loc="center right", bbox_to_anchor=(1.0, 0.39),
              handlelength=1.1, labelspacing=0.35, borderaxespad=0)
    clean_axis(ax)
    panel_label(ax, "e")


def draw_external_forest(ax: plt.Axes) -> None:
    labels = ["Identity", "Formula", "Identity near", "Formula near", "AUROC identity", "AUROC formula"]
    effects = np.array([1.201, 1.255, 1.502, 1.970, 1.258, 2.662])
    lows = np.array([0.460, 0.229, 0.589, 0.251, np.nan, np.nan])
    highs = np.array([1.890, 2.299, 2.464, 3.674, np.nan, np.nan])
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color=INK, lw=0.7)
    for i in range(4):
        ax.errorbar(effects[i], y[i], xerr=[[effects[i] - lows[i]], [highs[i] - effects[i]]],
                    fmt="s", color=ACCENT, ecolor=ACCENT, capsize=2, ms=3.7, lw=1.0)
    ax.scatter(effects[4:], y[4:], marker="o", s=18, facecolor="white", edgecolor=INK,
               linewidth=0.8, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.1, 4.0)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("V1 − official (percentage points)")
    ax.set_title("External effect sizes", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.55, zorder=0)
    ax.text(0.98, 0.98, "Recall@1: 95% CI", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=MID)
    clean_axis(ax)
    panel_label(ax, "f")


def main() -> None:
    style()
    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.065,
        right=0.985,
        bottom=0.095,
        top=0.965,
        wspace=0.42,
        hspace=0.62,
        width_ratios=[1.03, 1.12, 1.03],
        height_ratios=[0.92, 1.08],
    )

    draw_peak_attribution(fig.add_subplot(grid[0, 0]))
    draw_intervention_benchmark(fig.add_subplot(grid[0, 1]))
    draw_internal_benchmark(fig.add_subplot(grid[0, 2]))
    draw_transition_heatmaps(fig.add_subplot(grid[1, 0]))
    draw_external_benchmark(fig.add_subplot(grid[1, 1]))
    draw_external_forest(fig.add_subplot(grid[1, 2]))

    for suffix, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(
            OUT_DIR / f"noise_encoder_robustness_figure_v1.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
