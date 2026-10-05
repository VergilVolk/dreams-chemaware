"""代码绘制：本课题代谢物注释方法框架（中文第二版）。

顶部谱图读取仓库内 ST001122 的真实 MS/MS 谱；化学结构由 RDKit
根据 SMILES 生成。所有框、箭头和图标均由 matplotlib 代码绘制。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables" / "figures"
OUT_STEM = "metabolite_annotation_framework_cn_v2"
MGF_PATH = ROOT / "data" / "validation" / "bioaware_b47_st001122_truthblind_ms2_20260914_v1" / "mgf" / "IC35_56.mgf"

CN = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
CN_BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

NAVY = "#173F67"
BLUE = "#2E6FA9"
BLUE_MID = "#79A9D1"
BLUE_LIGHT = "#DCECF8"
BLUE_PALE = "#F4F9FD"
INK = "#182A3A"
WHITE = "#FFFFFF"


def read_mgf_scan(path: Path, target_scan: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Read one real spectrum from an MGF file by scan number."""
    mz, intensity = [], []
    active = False
    matched = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                active, matched = True, False
                mz, intensity = [], []
                continue
            if not active:
                continue
            if line.startswith("SCANS="):
                matched = int(line.split("=", 1)[1]) == target_scan
                continue
            if line == "END IONS":
                if matched and mz:
                    return np.asarray(mz), np.asarray(intensity)
                active = False
                continue
            if matched and line and line[0].isdigit() and " " in line:
                parts = line.split()
                if len(parts) >= 2:
                    mz.append(float(parts[0]))
                    intensity.append(float(parts[1]))
    raise RuntimeError(f"scan {target_scan} not found in {path}")


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": CN.get_name(),
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def add_box(ax, x, y, w, h, *, face=BLUE_PALE, edge=NAVY, lw=1.4, radius=1.3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, *, lw=1.4, head=11):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=lw,
            color=BLUE,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=0,
        )
    )


def draw_spectrum(ax, x, y, w, h, mz, intensity, *, color=BLUE, axes=True):
    keep = np.argsort(intensity)[-28:]
    sx = mz[keep]
    sy = intensity[keep] / intensity[keep].max()
    order = np.argsort(sx)
    sx, sy = sx[order], sy[order]
    lo, hi = sx.min(), sx.max()
    px = x + (sx - lo) / max(hi - lo, 1e-9) * w
    if axes:
        ax.plot([x, x], [y, y + h], color=INK, lw=0.75, zorder=3)
        ax.plot([x, x + w], [y, y], color=INK, lw=0.75, zorder=3)
    for xx, yy in zip(px, sy):
        ax.plot([xx, xx], [y, y + yy * h], color=color, lw=1.15, solid_capstyle="butt", zorder=3)


def molecule_images():
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    smiles = [
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "O=c1[nH]cnc2[nH]cnc12",
    ]
    images = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)
        images.append(Draw.MolToImage(mol, size=(260, 200), fitImage=True).convert("RGBA"))
    return images


def paste_image(ax, image, cx, cy, width, height):
    ax.imshow(np.asarray(image), extent=(cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2), interpolation="bilinear", aspect="auto", zorder=4)


def draw_condition_icon(ax, cx, cy):
    # Three code-drawn spectra indicate instrument/energy/ion-condition variation.
    heights = ([0.25, 0.75, 0.42, 0.95, 0.35, 0.62], [0.52, 0.92, 0.30, 0.66, 0.82, 0.38], [0.38, 0.58, 0.88, 0.44, 0.70, 0.94])
    for row, vals in enumerate(heights):
        yy = cy + 2.1 - row * 2.1
        ax.plot([cx - 4.3, cx + 4.2], [yy, yy], color=BLUE_MID, lw=0.55)
        for i, val in enumerate(vals):
            xx = cx - 3.7 + i * 1.35
            ax.plot([xx, xx], [yy, yy + val * 1.45], color=BLUE, lw=1.0)


def draw_fragment_icon(ax, cx, cy):
    # Product-ion and neutral-loss relationships, not a generic molecule network.
    ax.plot([cx - 4.2, cx + 4.2], [cy - 2.1, cy - 2.1], color=BLUE_MID, lw=0.65)
    peaks = [(-3.5, 2.0), (-1.5, 4.0), (0.2, 1.5), (2.0, 3.1), (3.6, 1.1)]
    for dx, height in peaks:
        ax.plot([cx + dx, cx + dx], [cy - 2.1, cy - 2.1 + height], color=BLUE, lw=1.35)
    ax.annotate("Δm", xy=(cx + 2.0, cy + 1.0), xytext=(cx - 1.5, cy + 2.4), color=NAVY, fontsize=7.0, ha="center", arrowprops={"arrowstyle": "<->", "color": NAVY, "lw": 0.8})


def draw_context_icon(ax, cx, cy):
    nodes = [(0, 0), (-3.6, 2.0), (-3.1, -2.0), (3.4, 2.4), (3.8, -1.7), (0.5, 3.5)]
    links = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (3, 4)]
    for a, b in links:
        ax.plot([cx + nodes[a][0], cx + nodes[b][0]], [cy + nodes[a][1], cy + nodes[b][1]], color=BLUE, lw=0.9)
    for i, (dx, dy) in enumerate(nodes):
        ax.add_patch(Circle((cx + dx, cy + dy), 1.05 if i == 0 else 0.72, facecolor=BLUE_MID if i else BLUE, edgecolor=NAVY, lw=0.8, zorder=3))


def draw_match_icon(ax, cx, cy):
    vals_a = [0.3, 0.8, 0.5, 1.0, 0.4, 0.7]
    vals_b = [0.2, 0.75, 0.45, 0.92, 0.35, 0.65]
    for i, (a, b) in enumerate(zip(vals_a, vals_b)):
        xx = cx - 4.0 + i * 1.4
        ax.plot([xx, xx], [cy + 0.3, cy + 0.3 + a * 2.8], color=BLUE, lw=1.0)
        ax.plot([xx, xx], [cy - 0.3, cy - 0.3 - b * 2.4], color=BLUE_MID, lw=1.0)
    ax.plot([cx - 4.5, cx + 4.5], [cy, cy], color=INK, lw=0.6)


def draw_rank_icon(ax, cx, cy):
    widths = [7.2, 5.8, 4.3]
    for i, width in enumerate(widths):
        yy = cy + 2.0 - i * 2.0
        ax.add_patch(FancyBboxPatch((cx - width / 2, yy - 0.48), width, 0.96, boxstyle="round,pad=0.05,rounding_size=0.35", facecolor=BLUE_LIGHT if i else BLUE_MID, edgecolor=NAVY, lw=0.7))


def draw_funnel_icon(ax, cx, cy):
    ax.add_patch(Polygon([(cx - 4.0, cy + 2.3), (cx + 4.0, cy + 2.3), (cx + 1.4, cy - 0.2), (cx + 1.4, cy - 2.8), (cx - 1.4, cy - 2.8), (cx - 1.4, cy - 0.2)], closed=True, facecolor=BLUE_MID, edgecolor=NAVY, lw=0.9))


def draw_flask_icon(ax, cx, cy):
    ax.plot([cx - 1.0, cx - 1.0, cx - 3.2, cx + 3.2, cx + 1.0, cx + 1.0], [cy + 3.0, cy + 1.1, cy - 3.0, cy - 3.0, cy + 1.1, cy + 3.0], color=NAVY, lw=1.2)
    ax.plot([cx - 1.2, cx + 1.2], [cy + 3.0, cy + 3.0], color=NAVY, lw=1.2)
    ax.add_patch(Polygon([(cx - 2.6, cy - 2.4), (cx + 2.6, cy - 2.4), (cx + 1.7, cy - 0.9), (cx - 1.7, cy - 0.9)], closed=True, facecolor=BLUE_MID, edgecolor="none"))
    ax.add_patch(Circle((cx + 0.7, cy - 1.6), 0.28, facecolor=WHITE, edgecolor="none"))


def main() -> None:
    configure()
    mz, intensity = read_mgf_scan(MGF_PATH, target_scan=4275)
    mols = molecule_images()

    # Portrait-oriented canvas prevents the boxes from becoming flat and wide.
    fig = plt.figure(figsize=(9.6, 11.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97.1, "本课题代谢物注释方法框架", ha="center", va="center", fontsize=22, color=NAVY, fontproperties=CN_BOLD)

    # Input with a real experimental MS/MS spectrum.
    add_box(ax, 18, 84.5, 64, 9.0, face=BLUE_PALE, lw=1.5)
    draw_spectrum(ax, 21.5, 86.0, 16.0, 5.5, mz, intensity)
    ax.text(61.0, 89.0, "非靶向代谢组学 LC–MS/MS 数据", ha="center", va="center", fontsize=14.5, color=NAVY, fontproperties=CN_BOLD)

    arrow(ax, 50, 84.2, 50, 81.6)

    # One unified annotation/retrieval block.
    add_box(ax, 16, 71.0, 68, 10.0, face=BLUE_PALE, lw=1.5)
    draw_spectrum(ax, 19.2, 73.0, 10.0, 5.2, mz, intensity, color=BLUE_MID)
    for image, cx in zip(mols, [33.0, 39.2, 45.4]):
        paste_image(ax, image, cx, 76.0, 5.8, 5.2)
    ax.text(65.0, 76.0, "待注释谱图、候选结构\n与基础谱图检索", ha="center", va="center", fontsize=13.2, color=NAVY, linespacing=1.35, fontproperties=CN_BOLD)

    arrow(ax, 50, 70.7, 50, 68.0)
    ax.plot([16, 84], [67.0, 67.0], color=BLUE, lw=1.35)
    for cx in [17, 50, 83]:
        arrow(ax, cx, 67.0, cx, 64.1)

    # Three research components.
    boxes = [(3.0, 49.5, 28.0, 14.0), (36.0, 49.5, 28.0, 14.0), (69.0, 49.5, 28.0, 14.0)]
    for x, y, w, h in boxes:
        add_box(ax, x, y, w, h, face=BLUE_PALE, lw=1.35)

    # Compact, interpretable code-drawn illustrations above larger labels.
    draw_condition_icon(ax, 17.0, 59.1)
    draw_fragment_icon(ax, 50.0, 59.0)
    draw_context_icon(ax, 83.0, 58.9)
    ax.text(17.0, 52.8, "面向实验条件变化的\n自监督谱图模型训练", ha="center", va="center", fontsize=14.0, color=NAVY, linespacing=1.38, fontproperties=CN_BOLD)
    ax.text(50.0, 52.8, "质谱碎裂化学规律\n辅助候选结构判断", ha="center", va="center", fontsize=14.0, color=NAVY, linespacing=1.38, fontproperties=CN_BOLD)
    ax.text(83.0, 52.8, "生物样本内代谢关系\n辅助结构判断", ha="center", va="center", fontsize=14.0, color=NAVY, linespacing=1.38, fontproperties=CN_BOLD)

    # Merge the three components into the ordered downstream research route.
    for cx in [17, 50, 83]:
        ax.plot([cx, cx], [49.2, 47.2], color=BLUE, lw=1.3)
    ax.plot([17, 83], [47.2, 47.2], color=BLUE, lw=1.3)
    arrow(ax, 50, 47.2, 50, 44.5)

    # The first two downstream steps are the methodological focus.
    downstream = [
        (35.0, 9.0, 22.0, 56.0, "引入谱学信息辅助\n近结构分子判断", 15.0, BLUE_LIGHT, 1.9, draw_match_icon),
        (23.0, 9.0, 22.0, 56.0, "引入质谱碎裂化学规律\n进行错误重排", 15.0, BLUE_LIGHT, 1.9, draw_rank_icon),
        (13.5, 6.5, 25.0, 50.0, "重点结构筛选与真实数据应用", 13.2, BLUE_PALE, 1.35, draw_funnel_icon),
        (4.5, 6.5, 25.0, 50.0, "结合非靶向代谢组学实验确认", 13.2, BLUE_PALE, 1.35, draw_flask_icon),
    ]
    for i, (y, h, x, w, text, fs, face, lw, icon_fn) in enumerate(downstream):
        add_box(ax, x, y, w, h, face=face, lw=lw)
        icon_x = x + 7.0
        icon_fn(ax, icon_x, y + h / 2)
        ax.text(x + w * 0.60, y + h / 2, text, ha="center", va="center", fontsize=fs, color=NAVY, linespacing=1.35, fontproperties=CN_BOLD)
        if i < len(downstream) - 1:
            next_y, next_h = downstream[i + 1][0], downstream[i + 1][1]
            arrow(ax, 50, y - 0.2, 50, next_y + next_h + 0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
