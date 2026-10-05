"""图1：非靶向 LC-MS/MS 代谢物注释的核心瓶颈。

A：代码排版的注释流程；LC-MS 图来自 NIH/NIAID BioArt 公共领域素材。
B：代码复现 DreaMS 图1b所示的分子空间覆盖关系。
C：代码绘制公开谱图与已注释谱库的数量差距。

数据来源：Bushuiev et al., Nature Biotechnology (2025),
https://doi.org/10.1038/s41587-025-02663-3
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
LCMS_ASSET = ROOT / "deliverables" / "assets" / "nih_lcms_public_domain.png"
OUT_DIR = ROOT / "deliverables" / "figures"
OUT_STEM = "GLM_figure1_annotation_bottleneck"

CN = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf")
CN_BOLD = FontProperties(fname=r"C:\Windows\Fonts\arialbd.ttf")

INK = "#202020"
MID = "#777777"
GRID = "#E5E5E5"
BLUE = "#2C78B4"
YELLOW = "#F2B544"
RED = "#C6473A"
TEAL = "#BFDCD8"
PINK = "#EAA6B5"
PURPLE = "#C5B7D8"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": CN.get_name(),
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def panel_title(ax, tag: str, title: str) -> None:
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=13, color=INK,
            fontproperties=CN_BOLD, va="bottom")
    ax.text(0.08, 1.02, title, transform=ax.transAxes, fontsize=10.8, color=INK,
            fontproperties=CN_BOLD, va="bottom")


def down_arrow(ax, y0: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (50, y0), (50, y1), arrowstyle="-|>", mutation_scale=11,
            color=MID, linewidth=1.1, shrinkA=0, shrinkB=0,
        )
    )


def draw_tubes(ax) -> None:
    colors = ["#F1A2AE", "#72A8D4", "#62B9A7"]
    xs = [42, 50, 58]
    for x, color in zip(xs, colors):
        ax.add_patch(Rectangle((x - 2.2, 87), 4.4, 7.5, facecolor="white",
                               edgecolor=MID, linewidth=0.8))
        ax.add_patch(Rectangle((x - 2.0, 87), 4.0, 4.6, facecolor=color,
                               edgecolor="none"))
        ax.add_patch(Rectangle((x - 2.6, 94.5), 5.2, 1.3, facecolor=MID,
                               edgecolor="none"))
    ax.text(50, 83.5, "Biological samples", ha="center", fontsize=9.6,
            color=INK, fontproperties=CN_BOLD)


def draw_spectrum(ax, x: float, y: float, w: float, h: float) -> None:
    positions = np.array([0.05, 0.13, 0.24, 0.31, 0.43, 0.56, 0.67, 0.82, 0.94])
    intensities = np.array([0.22, 0.66, 0.34, 1.00, 0.48, 0.28, 0.74, 0.42, 0.18])
    ax.plot([x, x + w], [y, y], color=INK, lw=0.9)
    ax.plot([x, x], [y, y + h], color=INK, lw=0.9)
    for p, intensity in zip(positions, intensities):
        ax.plot([x + p * w, x + p * w], [y, y + intensity * h],
                color=BLUE, lw=1.5, solid_capstyle="butt")
    ax.text(x + w + 1, y - 0.5, "m/z", fontsize=6.7, color=MID)


def molecule_image() -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    molecule = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    AllChem.Compute2DCoords(molecule)
    return np.asarray(Draw.MolToImage(molecule, size=(360, 280), fitImage=True).convert("RGBA"))


def draw_panel_a(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    panel_title(ax, "a", "From biological samples to candidate structures")

    draw_tubes(ax)
    down_arrow(ax, 81, 75)

    instrument = mpimg.imread(LCMS_ASSET)
    ax.imshow(instrument, extent=(24, 76, 57, 75), interpolation="lanczos", aspect="auto")
    ax.text(50, 54, "LC–MS/MS acquisition", ha="center", fontsize=9.3,
            color=INK, fontproperties=CN_BOLD)
    down_arrow(ax, 51, 46)

    draw_spectrum(ax, 31, 34, 38, 11)
    ax.text(50, 29.5, "Hundreds of MS/MS spectra per minute", ha="center", fontsize=8.8,
            color=INK, fontproperties=CN_BOLD)
    down_arrow(ax, 27, 22)

    ax.add_patch(
        FancyBboxPatch(
            (31, 13), 38, 8, boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor="#EEF4F8", edgecolor=BLUE, linewidth=1.0,
        )
    )
    ax.text(50, 17, "Spectral model and candidate ranking", ha="center", va="center",
            fontsize=8.7, color=INK, fontproperties=CN_BOLD)
    down_arrow(ax, 12, 8)

    ax.imshow(molecule_image(), extent=(39, 61, -2, 9), interpolation="bilinear", aspect="auto")
    ax.text(67, 4.2, "Candidate structure", fontsize=8.5, color=INK,
            fontproperties=CN_BOLD, va="center")

    ax.text(73, 39.5, "Most spectra", fontsize=8.0, color=MID, va="center")
    ax.text(73, 35.0, "lack reliable\nstructure annotations", fontsize=7.7, color=RED,
            va="center", fontproperties=CN_BOLD)


def draw_panel_b(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    panel_title(ax, "b", "Annotated libraries cover limited molecular space")

    # Code-drawn reproduction of the coverage relationships in DreaMS Fig. 1b.
    ax.add_patch(Ellipse((49, 49), 94, 72, facecolor=TEAL, edgecolor="none", alpha=0.42))
    ax.add_patch(Ellipse((38, 51), 48, 58, facecolor="#5A88C8", edgecolor="none", alpha=0.82))
    ax.add_patch(Ellipse((59, 50), 21, 31, facecolor=PINK, edgecolor="none", alpha=0.78))
    ax.add_patch(Ellipse((61, 34), 34, 40, facecolor=PURPLE, edgecolor="none", alpha=0.66))

    ax.text(38, 55, "353K", ha="center", fontsize=9.4, color=INK, fontproperties=CN_BOLD)
    ax.text(54, 57, "8K", ha="center", fontsize=7.8, color=INK)
    ax.text(63, 58, "15K", ha="center", fontsize=7.8, color=INK)
    ax.text(58, 49, "6K", ha="center", fontsize=7.6, color=INK)
    ax.text(66, 48, "2K", ha="center", fontsize=7.6, color=INK)
    ax.text(48, 35, "18K", ha="center", fontsize=7.8, color=INK)
    ax.text(68, 28, "77K", ha="center", fontsize=8.4, color=INK, fontproperties=CN_BOLD)

    ax.text(29, 79, "Molecules represented by unannotated spectra\nMassIVE GNPS", ha="center", fontsize=7.5,
            color=INK, fontproperties=CN_BOLD)
    ax.text(79, 57, "Molecules in annotated spectral libraries\nMoNA and NIST20", ha="center", fontsize=7.5,
            color=INK, fontproperties=CN_BOLD)
    ax.text(25, 17, "Natural-product space\nCOCONUT", ha="center", fontsize=7.8, color=MID)
    ax.text(76, 12, "Human metabolites\nHMDB", ha="center", fontsize=7.8, color=MID)

    ax.text(50, 3, "Annotated libraries cover only a small fraction of molecular space", ha="center",
            fontsize=8.3, color=RED, fontproperties=CN_BOLD)


def draw_panel_c(ax) -> None:
    labels = ["Available public spectra", "Annotated library spectra"]
    values = np.array([714_000_000, 2_000_000], dtype=float)
    x = np.arange(2)
    bars = ax.bar(x, values, width=0.30, color=[BLUE, YELLOW], edgecolor="none", zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(1_000_000, 1_350_000_000)
    ax.set_yticks([1e6, 1e7, 1e8, 1e9], ["1M", "10M", "100M", "1B"])
    ax.set_xticks(x, labels, fontproperties=CN_BOLD)
    ax.set_ylabel("Number of MS/MS spectra (log scale)", fontproperties=CN)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, color=INK)
    panel_title(ax, "c", "Public spectra vastly outnumber annotated library spectra")

    for bar, value, label in zip(bars, values, ["714M", "~2M"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, label,
                ha="center", va="bottom", fontsize=9.8, color=INK,
                fontproperties=CN_BOLD)

    ax.text(0.98, 0.90, "~357-fold gap", transform=ax.transAxes, ha="right", va="center",
            color=RED, fontsize=10.4, fontproperties=CN_BOLD)
    ax.text(0.50, -0.18, "Fewer than 10% of small-molecule MS/MS spectra can be reliably annotated",
            transform=ax.transAxes, ha="center", va="center", color=MID, fontsize=8.0)


def main() -> None:
    configure()
    fig = plt.figure(figsize=(15.0, 6.2), facecolor="white")
    ax_a = fig.add_axes([0.04, 0.12, 0.25, 0.75])
    ax_b = fig.add_axes([0.34, 0.12, 0.30, 0.75])
    ax_c = fig.add_axes([0.70, 0.20, 0.27, 0.59])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    # Light separators preserve the clean style used in the performance figure.
    fig.add_artist(mpl.lines.Line2D([0.315, 0.315], [0.12, 0.87], color=GRID, lw=0.8))
    fig.add_artist(mpl.lines.Line2D([0.67, 0.67], [0.12, 0.87], color=GRID, lw=0.8))

    fig.suptitle("Figure 1. The core bottleneck in untargeted LC–MS/MS metabolite annotation",
                 x=0.50, y=0.965, fontsize=15.0, color=INK, fontproperties=CN_BOLD)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight",
                    facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
