# GLM Figure 2 - Multi-level improvement of LC-MS/MS metabolite annotation
# One base annotation task + three complementary evidence streams entering at
# different positions. Paper-style vector figure, code-drawn (matplotlib + RDKit).
# Outputs: SVG + PDF (vector) and 600 dpi PNG.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

INK = "#1a1a1a"
GRAY = "#9a9a9a"
LIGHT = "#d9d9d9"
ACCENT = "#c0392b"
GREEN = "#1e8449"
BLUE = "#1f618d"

_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(_FONT_PATH)
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyhbd.ttc")
matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT_PATH).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join("deliverables", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
BASE = os.path.splitext(os.path.basename(__file__))[0]

MOL_SMILES = [
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O",       # dimethylxanthine
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",     # caffeine
    "O=c1[nH]cnc2[nH]cnc12",            # hypoxanthine
]


def mol_images(smiles_list, size=(240, 220)):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    imgs = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise RuntimeError(f"invalid SMILES: {smi}")
        AllChem.Compute2DCoords(m)
        imgs.append(Draw.MolToImage(m, size=size, fitImage=True))
    return imgs


fig = plt.figure(figsize=(13.6, 6.0))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc="white", ec=GRAY, lw=1.2, r=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, color=GRAY, lw=1.5, ls="-", head=8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=head, color=color, lw=lw,
                                 linestyle=ls, shrinkA=2, shrinkB=5,
                                 connectionstyle=f"arc3,rad={rad}"))


def stick(x, y, w, h, n=16, seed=0, color=INK, lw=1.1):
    r = np.random.default_rng(seed)
    mzs = np.sort(r.choice(np.linspace(0.08, 0.95, n * 2), n, replace=False))
    ints = (r.uniform(0.12, 1.0, n) ** 1.6)
    ints = ints / ints.max()
    ints = ints[r.permutation(n)] * 0.95 + 0.05
    for m, i in zip(mzs, ints):
        ax.plot([x + m * w, x + m * w], [y, y + i * h], color=color, lw=lw,
                solid_capstyle="butt")
    ax.plot([x, x + w], [y, y], color=color, lw=0.8)


def paste(img, cx, cy, w, alpha=1.0):
    im = np.asarray(img.convert("RGBA"))
    # Compensate for the wide figure so molecular structures retain their
    # native aspect ratio in data coordinates.
    y_scale = fig.get_figwidth() / fig.get_figheight()
    ax.imshow(im, extent=(cx - w / 2, cx + w / 2,
                          cy - w / 2 * im.shape[0] / im.shape[1] * y_scale,
                          cy + w / 2 * im.shape[0] / im.shape[1] * y_scale),
              zorder=6, alpha=alpha, interpolation="bilinear", aspect="auto")


# =============== left: shared input =====================================
stick(3.0, 65, 12, 15, seed=3)
ax.text(9, 60.5, "查询 MS/MS", fontsize=8.5, ha="center", color=INK)
stick(3.0, 31, 12, 14, seed=4, color=BLUE)
ax.text(9, 24.5, "候选参考谱", fontsize=8.5, ha="center", color=INK)
try:
    mols = mol_images(MOL_SMILES)
except Exception as e:  # noqa
    print("mol fallback:", e)
    mols = []

# pretrained DreaMS trunk
box(19, 39, 16, 25, fc="#eef3f7", ec=INK, lw=1.4)
ax.text(27, 57, "自监督预训练 DreaMS", fontsize=9.4, ha="center", color=INK,
        fontweight="bold")
ax.text(27, 47, "谱图表征与初始候选排序", fontsize=7.2,
        ha="center", color=GRAY)
arrow(15.5, 72, 18.9, 59, color=INK)
arrow(15.5, 36, 18.9, 43, color=INK)

# =============== upper branch: improve the model itself ==================
box(43, 73, 24, 15, fc="white", ec=GRAY, lw=1.1)
ax.text(55, 83.8, "共享编码器训练", fontsize=9.2, ha="center", color=INK,
        fontweight="bold")
ax.text(55, 77.5, "多条件实测谱图 · 定向谱图动作 · 困难候选",
        fontsize=6.8, ha="center", color=BLUE)
arrow(35.5, 59, 42.5, 80.5, color=INK, lw=1.5)
box(70, 73.5, 17, 14, fc="#f4faf5", ec=GREEN, lw=1.3)
ax.text(78.5, 84.5, "Noise Top-1", fontsize=7.0, ha="center", color=GRAY)
ax.text(78.5, 80.3, "93.19 $\\rightarrow$ 94.24%", fontsize=9.0, ha="center",
        color=GREEN, fontweight="bold")
ax.text(78.5, 76.2, "ChemAware 原生训练  +1.81 个百分点", fontsize=5.8, ha="center",
        color=GRAY)
arrow(67.5, 80.5, 69.5, 80.5, color=GRAY)

# =============== middle main road: local chemical evidence ===============
box(43, 43, 24, 16, fc="white", ec=GRAY, lw=1.2)
ax.text(55, 54.5, "候选后融合重排", fontsize=9.2, ha="center", color=INK,
        fontweight="bold")
ax.text(55, 48.0, "特征碎片 · 中性丢失\n· 候选条件化化学证据",
        fontsize=6.8,
        ha="center", color=GRAY)
arrow(35.5, 51.5, 42.5, 51.5, color=INK, lw=1.5)
box(70, 43.5, 17, 14, fc="#f4faf5", ec=GREEN, lw=1.3)
ax.text(78.5, 54.0, "V2 开发集：Top-1 误判率", fontsize=6.4, ha="center", color=GRAY)
ax.text(78.5, 50.0, "−41%", fontsize=10.5, ha="center", color=GREEN,
        fontweight="bold")
ax.text(78.5, 46.2, "P2b 封存评价  +1.07 个百分点", fontsize=5.9, ha="center", color=GRAY)
arrow(67.5, 51.0, 69.5, 51.0, color=GRAY)

# =============== lower branch: biochemical evidence from the sample =======
box(43, 12, 24, 17, fc="white", ec=GRAY, lw=1.2)
ax.text(55, 24.5, "样本局部生化证据", fontsize=9.2, ha="center", color=INK,
        fontweight="bold")
ax.text(55, 18.7, "高置信度代谢物\n+ 精确生化反应关系",
        fontsize=7.0, ha="center", color=GRAY)
arrow(35.5, 44, 42.5, 20.5, color=INK, lw=1.5)
box(70, 12.5, 17, 15, fc="#f4faf5", ec=GREEN, lw=1.3)
ax.text(78.5, 24.0, "B30 开放开发集", fontsize=6.3, ha="center", color=GRAY)
ax.text(78.5, 20.0, "75.41 $\\rightarrow$ 78.30%", fontsize=8.8, ha="center",
        color=GREEN, fontweight="bold")
ax.text(78.5, 15.8, "B47：4,996 个候选特异事件查询", fontsize=5.8,
        ha="center", color=GRAY)
arrow(67.5, 20.5, 69.5, 20.5, color=GRAY)

# =============== right: unified output ===================================
box(90, 31, 9.5, 43, fc="#eef3f7", ec=INK, lw=1.4)
ax.text(94.7, 69.0, "候选结构\n排序", fontsize=8.6, ha="center",
        color=INK, fontweight="bold")
try:
    for k, ry in enumerate((61.5, 51.5, 41.5)):
        paste(mols[k % 3], 94.7, ry, 4.6, alpha=1.0 if k == 0 else 0.72)
except Exception as e:  # noqa
    print("output mol fallback:", e)
ax.text(94.7, 33.8, "未来证据整合", fontsize=6.5,
        ha="center", color=GRAY)
arrow(87.5, 80.5, 89.8, 65, color=GRAY, rad=-0.15)
arrow(87.5, 51.0, 89.8, 51.0, color=GRAY)
arrow(87.5, 20.5, 89.8, 36, color=GRAY, rad=0.15)
# dashed: authentic-standard validation (next stage)
box(90, 15, 9.5, 10, fc="white", ec=GRAY, lw=1.0)
ax.text(94.7, 20.0, "同法标准品验证", fontsize=6.8, ha="center",
        color=GRAY)
arrow(94.7, 30.5, 94.7, 25.5, color=GRAY, ls=(0, (3, 3)), lw=1.2)
ax.text(55, 5.5, "不同结果来自不同评价协议，效应量不可直接相加。",
        fontsize=7.2, ha="center", color=GRAY, style="italic")

fig.suptitle("图2  LC–MS/MS 代谢物注释的多层次改进框架",
             fontsize=12.5, y=0.985, color=INK, fontweight="bold")

for ext in ("svg", "pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"{BASE}.{ext}"), dpi=600 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print("written:", OUT_DIR, BASE)
