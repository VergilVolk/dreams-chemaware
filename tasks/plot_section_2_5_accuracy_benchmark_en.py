"""Generate the compact accuracy benchmark for Section 2.5."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
OUT_STEM = "section_2_5_accuracy_benchmark_en_v2"

FONT = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = FontProperties(fname=r"C:\Windows\Fonts\arialbd.ttf")

INK = "#202020"
GRID = "#E8E8E8"
ACCENT = "#92283B"
ACCENT_LIGHT = "#D1A0AA"


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT.get_name(),
            "font.size": 8.2,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    labels = [
        "Six-source development",
        "Real-library | all queries",
        "Real-library | negative mode",
    ]
    effects = np.array([5.93, 2.8817, 6.24])
    y = np.arange(3)[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.45), facecolor="white")
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.19, top=0.88)

    ax.barh(y, effects, height=0.50, color=[ACCENT_LIGHT, ACCENT, ACCENT], edgecolor="none", zorder=2)

    # Source range for the six-source panel.
    ax.errorbar(
        effects[0],
        y[0],
        xerr=[[effects[0] - 3.70], [8.67 - effects[0]]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )
    # Formula-cluster 95% confidence interval for all real-library queries.
    ax.errorbar(
        effects[1],
        y[1],
        xerr=[[effects[1] - 1.5528], [4.3880 - effects[1]]],
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )

    notes = [
        (8.85, "+5.93 pp  [3.70, 8.67] range"),
        (4.56, "+2.88 pp  [1.55, 4.39] 95% CI"),
        (6.47, "+6.24 pp"),
    ]
    for yy, (xx, text) in zip(y, notes):
        ax.text(xx, yy, text, ha="left", va="center", fontsize=7.0, color=INK, fontproperties=FONT_BOLD)

    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, labels, fontproperties=FONT)
    ax.set_xlim(0, 11.4)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Top-1 accuracy gain (percentage points)", fontproperties=FONT)
    ax.set_title("Sample-context biochemical evidence", loc="left", pad=9, fontproperties=FONT_BOLD)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
