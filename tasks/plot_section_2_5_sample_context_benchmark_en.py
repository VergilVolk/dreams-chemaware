"""English benchmarking figure for Section 2.5.

The left axis reports Top-1 accuracy gains within matched benchmarks. The right
axis reports truth-blind candidate-specific event yields. The two quantities
are deliberately separated because event yield is not an accuracy estimate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
OUT_STEM = "section_2_5_sample_context_benchmark_en_v1"

FONT = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = FontProperties(fname=r"C:\Windows\Fonts\arialbd.ttf")

INK = "#202020"
MID = "#737373"
LIGHT = "#D3D3D3"
GRID = "#E8E8E8"
ACCENT = "#92283B"
ACCENT_LIGHT = "#D1A0AA"
PALE = "#F8F0F2"
WARNING = "#A23A32"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT.get_name(),
            "font.size": 8.0,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.3,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def draw_accuracy_benchmark(ax: plt.Axes) -> None:
    ax.set_title("Top-1 accuracy gain within each benchmark", loc="left", pad=17, fontproperties=FONT_BOLD)
    ax.text(
        0.0,
        1.015,
        "Percentage-point change relative to the matched spectral baseline",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=MID,
        fontproperties=FONT,
    )

    labels = [
        "Six-source development panel",
        "Real-library retrieval | all queries",
        "Real-library retrieval | negative mode",
    ]
    effects = np.array([5.93, 2.8817, 6.24])
    y = np.arange(3)[::-1]

    # Six-source whisker is the observed source range, not a confidence interval.
    source_low, source_high = 3.70, 8.67
    # Formula-cluster bootstrap interval for the complete real-library benchmark.
    all_low, all_high = 1.5528, 4.3880

    ax.barh(y, effects, height=0.50, color=[ACCENT_LIGHT, ACCENT, ACCENT], edgecolor="none", zorder=2)
    ax.errorbar(
        effects[0],
        y[0],
        xerr=[[effects[0] - source_low], [source_high - effects[0]]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )
    ax.errorbar(
        effects[1],
        y[1],
        xerr=[[effects[1] - all_low], [all_high - effects[1]]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )

    annotations = [
        "+5.93 pp\nsource range: +3.70 to +8.67",
        "+2.88 pp\n75.41% to 78.30%",
        "+6.24 pp\n64.41% to 70.65%",
    ]
    annotation_x = [source_high + 0.20, all_high + 0.20, effects[2] + 0.24]
    for yy, xx, note in zip(y, annotation_x, annotations):
        ax.text(xx, yy, note, ha="left", va="center", fontsize=6.8, color=INK, linespacing=1.15, fontproperties=FONT_BOLD)

    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, labels, fontproperties=FONT)
    ax.set_xlim(0, 11.3)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Top-1 accuracy gain (percentage points)", fontproperties=FONT)
    clean(ax)

    ax.text(
        0.0,
        -0.22,
        "The six-source whisker shows the observed source range; the all-query whisker is a 95% formula-cluster CI.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=MID,
        fontproperties=FONT,
    )
    ax.text(
        0.0,
        -0.34,
        "Coverage- and degree-matched controls attenuated the static-network gain;\nindependent MassBank transfer was not robust.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=WARNING,
        fontproperties=FONT_BOLD,
    )


def draw_event_benchmark(ax: plt.Axes) -> None:
    ax.set_title("Candidate-specific biochemical support in real samples", loc="left", pad=17, fontproperties=FONT_BOLD)
    ax.text(
        0.0,
        1.015,
        "Truth-blind event yield; not an annotation-accuracy estimate",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=WARNING,
        fontproperties=FONT_BOLD,
    )

    labels = [
        "Observed sample context",
        "Metabolite-to-sample permutation\n(maximum of 20 runs)",
        "Degree-preserving network rewiring\n(maximum of 20 runs)",
    ]
    values = np.array([4996, 4072, 91])
    colors = [ACCENT, ACCENT_LIGHT, LIGHT]
    y = np.arange(3)[::-1]
    ax.barh(y, values, height=0.52, color=colors, edgecolor=INK, linewidth=0.45, zorder=2)
    for yy, value in zip(y, values):
        ax.text(value + 85, yy, f"{value:,}", ha="left", va="center", fontsize=7.0, color=INK, fontproperties=FONT_BOLD)

    ax.text(
        3850,
        y[0],
        "+22.69% vs permutation maximum",
        ha="center",
        va="center",
        fontsize=6.6,
        color="white",
        fontproperties=FONT_BOLD,
    )
    ax.set_yticks(y, labels, fontproperties=FONT)
    ax.set_xlim(0, 5700)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel("Queries with candidate-specific biochemical support", fontproperties=FONT)
    clean(ax)

    ax.text(
        0.99,
        0.055,
        "51,976 queries | 329 reaction classes | 2,239 potential rank changes",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=MID,
        fontproperties=FONT,
    )
    ax.text(
        0.0,
        -0.22,
        "Observed excess over the permutation maximum: +23.18% and +19.91% in the two studies.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=INK,
        fontproperties=FONT,
    )
    ax.text(
        0.0,
        -0.34,
        "The event-concentration gate was not met; structure truth remained unopened;\nannotation accuracy was not evaluated.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=WARNING,
        fontproperties=FONT_BOLD,
    )


def main() -> None:
    configure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.9, 4.55), facecolor="white", constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        left=0.17,
        right=0.985,
        bottom=0.24,
        top=0.90,
        wspace=0.60,
        width_ratios=[1.0, 1.0],
    )
    draw_accuracy_benchmark(fig.add_subplot(grid[0, 0]))
    draw_event_benchmark(fig.add_subplot(grid[0, 1]))
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
