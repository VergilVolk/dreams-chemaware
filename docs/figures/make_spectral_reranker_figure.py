from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from make_noise_encoder_robustness_figure import (
    ACCENT,
    ACCENT_LIGHT,
    INK,
    LIGHT,
    MID,
    clean_axis,
    panel_label,
    style,
)


OUT_DIR = Path(__file__).resolve().parent


def feature_weights(ax: plt.Axes) -> None:
    labels = ["Neutral loss", "Spectral entropy", "DreaMS", "Cosine"]
    values = [0.80, 0.10, 0.10, 0.00]
    y = np.arange(len(labels))
    ax.barh(y, values, color=[ACCENT, ACCENT_LIGHT, MID, LIGHT], edgecolor=INK, linewidth=0.55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.86)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_xlabel("Frozen fusion weight")
    ax.set_title("P2b evidence composition", loc="left", pad=5, fontweight="bold")
    for yi, value in zip(y, values):
        ax.text(value + 0.025, yi, f"{value:.1f}", va="center", ha="left", fontsize=6.3)
    clean_axis(ax)
    panel_label(ax, "a")


def development_benchmark(ax: plt.Axes) -> None:
    values = [86.06, 89.97]
    x = np.arange(2)
    ax.plot(x, values, color=INK, lw=0.9)
    ax.scatter(x, values, s=[34, 42], c=[MID, ACCENT], edgecolor=INK, linewidth=0.55, zorder=3)
    ax.set_xticks(x, ["DreaMS", "P2b"])
    ax.set_ylim(84.8, 90.7)
    ax.set_yticks([85, 87, 89])
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("P2 development benchmark", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", lw=0.55)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + 0.22, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.text(0.98, 0.05, "n = 5,037\n280 / 83", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, color=MID)
    clean_axis(ax)
    panel_label(ax, "b")


def sealed_benchmark(ax: plt.Axes) -> None:
    labels = ["P3 main", "Near-core"]
    baseline = [87.93, 48.79]
    p2b = [89.00, 44.56]
    y = np.arange(2)
    h = 0.28
    ax.barh(y + h / 2, baseline, height=h, color=LIGHT, edgecolor=INK, linewidth=0.55,
            label="DreaMS")
    ax.barh(y - h / 2, p2b, height=h, color=ACCENT, edgecolor=INK, linewidth=0.55,
            label="P2b")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(40, 92)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_xlabel("Recall@1 (%)")
    ax.set_title("P2b sealed evaluation", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.55)
    ax.legend(frameon=False, loc="lower right", handlelength=1.1)
    clean_axis(ax)
    panel_label(ax, "c")


def transition_heatmaps(container: plt.Axes) -> None:
    container.axis("off")
    container.set_title("P2b paired transitions", loc="left", pad=5, fontweight="bold")
    panel_label(container, "d")
    gs = container.get_subplotspec().subgridspec(1, 2, wspace=0.30)
    matrices = [
        np.array([[97.84, 2.16], [24.59, 75.41]]),
        np.array([[83.06, 16.94], [7.87, 92.13]]),
    ]
    titles = ["P3 main", "Near-core"]
    notes = ["89 corrected\n57 introduced", "20 corrected\n41 introduced"]
    cmap = LinearSegmentedColormap.from_list("gray", ["#F5F5F5", "#BDBDBD", "#565656"])
    for j, (matrix, title, note) in enumerate(zip(matrices, titles, notes)):
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


def residual_benchmark(ax: plt.Axes) -> None:
    labels = ["Official", "Direct + backoff", "Residual V2"]
    values = [90.4614, 94.1939, 94.4012]
    x = np.arange(3)
    ax.plot(x, values, color=INK, lw=0.9)
    ax.scatter(x, values, s=[30, 36, 42], c=[MID, ACCENT_LIGHT, ACCENT],
               edgecolor=INK, linewidth=0.55, zorder=3)
    ax.set_xticks(x, ["Official", "Direct +\nbackoff", "Residual\nV2"])
    ax.set_ylim(89.8, 95.0)
    ax.set_yticks([90, 92, 94])
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Multi-null residual reranker", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", lw=0.55)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + 0.18, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.3)
    ax.text(0.98, 0.05, "role-3 development, n = 1,929", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.0, color=MID)
    clean_axis(ax)
    panel_label(ax, "e")


def effect_summary(ax: plt.Axes) -> None:
    labels = ["P2 development", "P2b main", "P2b near-core", "Residual V2", "V2 vs rotated null"]
    effects = np.array([3.91, 1.07, -4.23, 3.9399, 3.3178])
    lows = np.array([np.nan, 0.24, np.nan, 2.8191, 2.3060])
    highs = np.array([np.nan, 1.89, np.nan, 5.1921, 4.3776])
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color=INK, lw=0.7)
    for i in range(len(labels)):
        if np.isfinite(lows[i]):
            ax.errorbar(effects[i], y[i], xerr=[[effects[i] - lows[i]], [highs[i] - effects[i]]],
                        fmt="s", color=ACCENT, ecolor=ACCENT, capsize=2, ms=3.7, lw=1.0)
        else:
            marker = "v" if effects[i] < 0 else "o"
            ax.scatter(effects[i], y[i], marker=marker, s=22, facecolor="white",
                       edgecolor=INK, linewidth=0.8, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(-5.5, 6.0)
    ax.set_xticks([-4, -2, 0, 2, 4, 6])
    ax.set_xlabel("Recall@1 change (percentage points)")
    ax.set_title("Effect-size summary", loc="left", pad=5, fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.55)
    clean_axis(ax)
    panel_label(ax, "f")


def main() -> None:
    style()
    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.07,
        right=0.985,
        bottom=0.095,
        top=0.965,
        wspace=0.48,
        hspace=0.62,
        width_ratios=[1.0, 1.0, 1.12],
        height_ratios=[0.92, 1.08],
    )
    feature_weights(fig.add_subplot(grid[0, 0]))
    development_benchmark(fig.add_subplot(grid[0, 1]))
    sealed_benchmark(fig.add_subplot(grid[0, 2]))
    transition_heatmaps(fig.add_subplot(grid[1, 0]))
    residual_benchmark(fig.add_subplot(grid[1, 1]))
    effect_summary(fig.add_subplot(grid[1, 2]))

    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"spectral_reranker_figure_v1.{suffix}",
                    bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
