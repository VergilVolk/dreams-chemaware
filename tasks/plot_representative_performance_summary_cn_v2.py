"""绘制本阶段代表性性能提升：左侧共享谱图模型训练，右侧候选重排。"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables" / "figures"
OUT_STEM = "representative_performance_summary_cn_v2"

CN = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
CN_BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

INK = "#202020"
MID = "#777777"
GRID = "#E8E8E8"
BLUE = "#2C78B4"
YELLOW = "#F2B544"
SEPARATOR = "#C9C9C9"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": CN.get_name(),
            "font.size": 8.4,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def clean(ax, grid_axis):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.7, color=INK)


def draw_model_training(ax):
    ax.set_title("a  自监督谱图模型训练", loc="left", pad=18, fontproperties=CN_BOLD)

    labels = [
        "新分子检索\n10,995次",
        "新分子式检索\n5,261次",
        "质谱化学规则评价\n1,929次",
    ]
    baseline = np.array([85.36, 86.81, 90.46])
    improved = np.array([86.56, 88.06, 92.28])
    gains = ["+1.20个百分点", "+1.25个百分点", "+1.81个百分点"]
    x = np.array([0.00, 0.92, 2.12])
    width = 0.30

    bars0 = ax.bar(x - width / 2, baseline, width, color=BLUE, edgecolor="none", label="基础模型", zorder=2)
    bars1 = ax.bar(x + width / 2, improved, width, color=YELLOW, edgecolor="none", label="改进后", zorder=2)

    for group, (b0, b1, gain) in enumerate(zip(bars0, bars1, gains)):
        ax.text(b0.get_x() + b0.get_width() / 2, b0.get_height() - 3.1, f"{baseline[group]:.2f}%", ha="center", va="center", fontsize=7.6, color="white", fontproperties=CN_BOLD)
        ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height() - 3.1, f"{improved[group]:.2f}%", ha="center", va="center", fontsize=7.6, color="white", fontproperties=CN_BOLD)
        ax.text(x[group], max(baseline[group], improved[group]) + 3.0, gain, ha="center", va="bottom", fontsize=7.6, color=INK, fontproperties=CN_BOLD)

    # 前两组与第三组分别对应两条不同的共享模型训练路线。
    ax.text(0.46, 106.2, "面向实验条件变化的微调", ha="center", va="center", fontsize=8.6, color=MID, fontproperties=CN_BOLD)
    ax.text(2.12, 106.2, "利用质谱碎裂化学规则微调", ha="center", va="center", fontsize=8.6, color=MID, fontproperties=CN_BOLD)
    ax.axvline(1.52, color=SEPARATOR, lw=0.8, ymin=0.03, ymax=0.94)

    ax.set_xticks(x, labels, fontproperties=CN)
    ax.set_ylabel("首位准确率（%）", fontproperties=CN)
    ax.set_xlim(-0.42, 2.56)
    ax.set_ylim(0, 109)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    clean(ax, "y")


def draw_evidence_reranking(ax):
    ax.set_title("b  多类证据辅助候选重排", loc="left", pad=18, fontproperties=CN_BOLD)

    labels = [
        "质谱碎裂化学规律重排",
        "局部谱学信息重排",
        "完整谱库检索（1,631次）",
        "多来源样本（细胞、脑、肝、血浆等）",
        "负离子检索（753次）",
    ]
    baseline = np.array([90.46, 87.93, 75.41, 66.05, 64.41])
    improved = np.array([94.40, 89.00, 78.30, 71.98, 70.65])
    gains = ["+3.94个百分点", "+1.07个百分点", "+2.88个百分点", "+5.93个百分点", "+6.24个百分点"]
    centers = np.array([8.40, 6.55, 4.70, 2.85, 1.00])
    offset = 0.18
    height = 0.36

    ax.barh(centers + offset, baseline, height=height, color=BLUE, edgecolor="none", label="基础排序", zorder=2)
    ax.barh(centers - offset, improved, height=height, color=YELLOW, edgecolor="none", label="引入后", zorder=2)

    for i, center in enumerate(centers):
        ax.text(baseline[i] + 0.7, center + offset, f"{baseline[i]:.2f}%", ha="left", va="center", fontsize=7.8, fontproperties=CN_BOLD)
        ax.text(improved[i] + 0.7, center - offset, f"{improved[i]:.2f}%", ha="left", va="center", fontsize=7.8, fontproperties=CN_BOLD)
        ax.text(101.8, center, gains[i], ha="left", va="center", fontsize=8.2, color=INK, fontproperties=CN_BOLD)
        ax.text(0.8, center + 0.58, labels[i], ha="left", va="center", fontsize=8.4, color=INK, fontproperties=CN_BOLD)

    # 分隔碎裂化学、局部谱学与代谢知识三类证据。
    ax.axhline(7.48, color=SEPARATOR, lw=0.8)
    ax.axhline(5.62, color=SEPARATOR, lw=0.8)

    ax.set_yticks([])
    ax.set_xlim(0, 119)
    ax.set_ylim(0.45, 9.18)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("首位准确率（%）", fontproperties=CN)
    clean(ax, "x")


def main() -> None:
    configure()
    fig = plt.figure(figsize=(12.5, 6.7), facecolor="white")

    # 两个面板等高；左侧收紧柱间距，右侧给五组配对基准留出完整行距。
    ax_left = fig.add_axes([0.065, 0.13, 0.39, 0.68])
    ax_right = fig.add_axes([0.535, 0.13, 0.425, 0.68])

    draw_model_training(ax_left)
    draw_evidence_reranking(ax_right)
    fig.suptitle("本阶段代表性模型性能提升", x=0.52, y=0.992, fontsize=15.0, color=INK, fontproperties=CN_BOLD)
    handles = [
        mpl.patches.Patch(color=BLUE, label="基础模型"),
        mpl.patches.Patch(color=YELLOW, label="算法改进后"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.956), frameon=False, ncol=2, columnspacing=1.4, handlelength=1.4, prop=CN)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
