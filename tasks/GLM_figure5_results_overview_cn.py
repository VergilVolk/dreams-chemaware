"""绘制图5：本课题主要方法的结果总览。

四个面板只展示各自协议内部可比较的结果，不跨数据集拼接增益：
  a. 共享谱图模型的内部与GNPS外部评价；
  b. 固定谱学后融合与候选化学证据重排；
  c. 开放开发结果与真值盲样本事件产额；
  d. GNPS近结构子集外部评价。

输出 PNG、PDF 和 SVG 到仓库 deliverables 目录。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
OUT_STEM = "figure5_results_overview_cn_v1"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
CN = FontProperties(fname=str(FONT_REGULAR))
CN_BOLD = FontProperties(fname=str(FONT_BOLD))

INK = "#202020"
MID = "#6F6F6F"
LIGHT = "#D5D5D5"
PALE = "#F4F4F4"
GRID = "#E7E7E7"
GREEN = "#23864B"
GREEN_LIGHT = "#8FC6A5"
GREEN_PALE = "#EAF4ED"
RED = "#B13B32"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": CN.get_name(),
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, color=INK)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
        fontproperties=CN_BOLD,
    )


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=13, color=INK, fontproperties=CN_BOLD)


def container_title(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.0,
        1.12,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        color=INK,
        fontproperties=CN_BOLD,
    )


def add_delta_box(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        color=GREEN,
        fontsize=8.2,
        fontweight="bold",
        fontproperties=CN_BOLD,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": GREEN_PALE, "edgecolor": "none"},
        zorder=5,
    )


def draw_panel_a(ax: plt.Axes) -> None:
    panel_label(ax, "a")
    panel_title(ax, "共享谱图模型的内部与外部评价")

    labels = [
        "内部固定评价\n(n=18,333)",
        "GNPS身份隔离\n(n=10,995)",
        "GNPS分子式隔离\n(n=5,261)",
    ]
    original = np.array([93.1875, 85.357, 86.809])
    stage1 = np.array([93.6835, 86.066, 87.797])
    final = np.array([94.2399, 86.558, 88.063])
    deltas = ["+1.05", "+1.20", "+1.26"]

    x = np.arange(3)
    offsets = np.array([-0.18, 0.0, 0.18])
    series = [
        (original, "原始DreaMS", MID, "o"),
        (stage1, "第一阶段模型", GREEN_LIGHT, "s"),
        (final, "最终模型", GREEN, "D"),
    ]
    for offset, (values, name, color, marker) in zip(offsets, series):
        ax.scatter(
            x + offset,
            values,
            s=48,
            marker=marker,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.55,
            label=name,
            zorder=3,
        )
        for xx, yy in zip(x + offset, values):
            ax.text(xx, yy + 0.32, f"{yy:.2f}", ha="center", va="bottom", fontsize=7.5)
    for i in range(3):
        ax.plot(x[i] + offsets, [original[i], stage1[i], final[i]], color="#AFAFAF", lw=0.9, zorder=1)
        add_delta_box(ax, x[i] + 0.18, final[i] + 1.25, f"提高{deltas[i]}个百分点")

    ax.set_xticks(x, labels, fontproperties=CN)
    ax.set_ylabel("首位准确率（%）", fontproperties=CN)
    ax.set_ylim(82.5, 97.2)
    ax.set_yticks([84, 88, 92, 96])
    clean_axis(ax)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, -0.27),
        frameon=False,
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.2,
        prop=CN,
    )
    ax.text(
        0.99,
        -0.24,
        "GNPS：独立外部评价",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color=MID,
        fontproperties=CN,
    )


def draw_panel_b(container: plt.Axes) -> None:
    container.axis("off")
    panel_label(container, "b")
    container_title(container, "谱学后融合与候选化学证据重排")
    sub = container.get_subplotspec().subgridspec(1, 2, wspace=0.44)

    ax1 = container.figure.add_subplot(sub[0, 0])
    values = [87.93, 89.00]
    ax1.bar(
        [0, 1],
        values,
        width=0.58,
        color=[LIGHT, GREEN],
        edgecolor=INK,
        linewidth=0.55,
        zorder=2,
    )
    ax1.set_ylim(84.5, 91.3)
    ax1.set_yticks([85, 87, 89, 91])
    ax1.set_xticks([0, 1], ["基础排序", "固定谱学\n后融合"], fontproperties=CN)
    ax1.set_ylabel("首位准确率（%）", fontproperties=CN)
    ax1.set_title("封存评价", loc="left", pad=8, fontsize=9.5, fontproperties=CN_BOLD)
    for xx, yy in enumerate(values):
        ax1.text(xx, yy + 0.16, f"{yy:.2f}", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax1.text(
        0.5,
        90.55,
        "提高1.07个百分点\n95%置信区间：[0.24，1.89]",
        ha="center",
        va="center",
        fontsize=7.8,
        color=GREEN,
        fontproperties=CN_BOLD,
    )
    ax1.text(0.02, -0.19, "n=3,000；89个纠正 / 57个新增误判", transform=ax1.transAxes, fontsize=7.4, color=MID, fontproperties=CN)
    ax1.text(0.02, -0.30, "近结构核心子集：下降4.23个百分点", transform=ax1.transAxes, fontsize=7.4, color=RED, fontproperties=CN_BOLD)
    clean_axis(ax1)

    ax2 = container.figure.add_subplot(sub[0, 1])
    errors = [9.54, 5.60]
    ax2.bar(
        [0, 1],
        errors,
        width=0.58,
        color=[LIGHT, GREEN],
        edgecolor=INK,
        linewidth=0.55,
        zorder=2,
    )
    ax2.set_ylim(0, 11.2)
    ax2.set_yticks([0, 2, 4, 6, 8, 10])
    ax2.set_xticks([0, 1], ["基础排序", "化学证据\n重排"], fontproperties=CN)
    ax2.set_ylabel("首位误判率（%）", fontproperties=CN)
    ax2.set_title("开发评价", loc="left", pad=8, fontsize=9.5, fontproperties=CN_BOLD)
    for xx, yy in enumerate(errors):
        ax2.text(xx, yy + 0.22, f"{yy:.2f}", ha="center", va="bottom", fontsize=8.1, fontweight="bold")
    ax2.annotate(
        "相对下降41%",
        xy=(1, 5.75),
        xytext=(0.55, 9.0),
        ha="center",
        va="center",
        fontsize=8.4,
        color=GREEN,
        fontproperties=CN_BOLD,
        arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.1},
    )
    ax2.text(0.02, -0.19, "n=1,929；93个纠正 / 17个新增误判", transform=ax2.transAxes, fontsize=7.4, color=MID, fontproperties=CN)
    ax2.text(0.02, -0.30, "准确率增益：+3.94 [2.82，5.19] 个百分点", transform=ax2.transAxes, fontsize=7.4, color=MID, fontproperties=CN)
    clean_axis(ax2)


def draw_panel_c(container: plt.Axes) -> None:
    container.axis("off")
    panel_label(container, "c")
    container_title(container, "样本生化信息的排序结果与事件对照")
    sub = container.get_subplotspec().subgridspec(1, 2, width_ratios=[0.88, 1.42], wspace=0.40)

    ax1 = container.figure.add_subplot(sub[0, 0])
    values = [75.41, 78.30]
    ax1.plot([0, 1], values, color="#AAAAAA", lw=1.1, zorder=1)
    ax1.scatter([0, 1], values, s=[62, 72], c=[LIGHT, GREEN], edgecolor=INK, linewidth=0.6, zorder=3)
    ax1.set_xlim(-0.45, 1.45)
    ax1.set_ylim(73.7, 79.8)
    ax1.set_yticks([74, 76, 78])
    ax1.set_xticks([0, 1], ["基础排序", "加入样本\n生化信息"], fontproperties=CN)
    ax1.set_ylabel("首位准确率（%）", fontproperties=CN)
    ax1.set_title("开发数据上的排序结果", loc="left", pad=8, fontsize=9.5, fontproperties=CN_BOLD)
    for xx, yy in enumerate(values):
        ax1.text(xx, yy + 0.22, f"{yy:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax1.text(0.5, 79.25, "提高2.89个百分点", ha="center", color=GREEN, fontproperties=CN_BOLD, fontsize=8.2)
    ax1.text(0.02, -0.19, "n=1,631；50个纠正 / 3个新增误判", transform=ax1.transAxes, fontsize=7.4, color=MID, fontproperties=CN)
    ax1.text(0.02, -0.30, "开发数据，尚非独立验证", transform=ax1.transAxes, fontsize=7.4, color=RED, fontproperties=CN_BOLD)
    clean_axis(ax1)

    ax2 = container.figure.add_subplot(sub[0, 1])
    counts = [4996, 4072, 91]
    colors = [GREEN, GREEN_LIGHT, LIGHT]
    x = np.arange(3)
    ax2.bar(x, counts, width=0.60, color=colors, edgecolor=INK, linewidth=0.55, zorder=2)
    ax2.set_ylim(0, 5650)
    ax2.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax2.set_xticks(
        x,
        ["真实样本\n事件", "高置信代谢物上下文置换\n（20次最大值）", "反应网络重连\n（最大值）"],
        fontproperties=CN,
    )
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.set_ylabel("候选特异事件查询数", fontproperties=CN)
    ax2.set_title("两个真实研究中的事件数量", loc="left", pad=8, fontsize=9.5, fontproperties=CN_BOLD)
    for xx, yy in zip(x, counts):
        ax2.text(xx, yy + 105, f"{yy:,}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax2.plot([0, 0, 1, 1], [5230, 5350, 5350, 5230], color=INK, lw=0.8)
    ax2.text(0.5, 5390, "较最严格空模型高22.69%", ha="center", va="bottom", fontsize=7.8, color=GREEN, fontproperties=CN_BOLD)
    ax2.text(0.01, -0.19, "51,976次检索；2,239个潜在排序机会", transform=ax2.transAxes, fontsize=7.4, color=MID, fontproperties=CN)
    ax2.text(0.01, -0.30, "未读取正确结构；无准确率；事件分布集中度未达要求", transform=ax2.transAxes, fontsize=7.4, color=RED, fontproperties=CN_BOLD)
    clean_axis(ax2)


def draw_panel_d(ax: plt.Axes) -> None:
    panel_label(ax, "d")
    panel_title(ax, "GNPS近结构子集上的外部验证")

    labels = ["身份隔离近结构\n(n=9,592)", "分子式隔离近结构\n(n=2,944)"]
    original = np.array([79.939, 77.989])
    stage1 = np.array([80.822, 79.416])
    final = np.array([81.441, 79.959])
    effects = [
        "提高1.50个百分点\n95%置信区间：[0.59，2.46]",
        "提高1.97个百分点\n95%置信区间：[0.25，3.67]",
    ]
    x = np.arange(2)
    offsets = np.array([-0.18, 0.0, 0.18])
    series = [
        (original, "原始DreaMS", MID, "o"),
        (stage1, "第一阶段模型", GREEN_LIGHT, "s"),
        (final, "最终模型", GREEN, "D"),
    ]
    for offset, (values, name, color, marker) in zip(offsets, series):
        ax.scatter(
            x + offset,
            values,
            s=55,
            marker=marker,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.55,
            label=name,
            zorder=3,
        )
        for xx, yy in zip(x + offset, values):
            ax.text(xx, yy + 0.18, f"{yy:.2f}", ha="center", va="bottom", fontsize=7.8)
    for i in range(2):
        ax.plot(x[i] + offsets, [original[i], stage1[i], final[i]], color="#AAAAAA", lw=1.0, zorder=1)
        ax.text(
            x[i],
            83.05,
            effects[i],
            ha="center",
            va="top",
            color=GREEN,
            fontsize=8.1,
            fontproperties=CN_BOLD,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": GREEN_PALE, "edgecolor": "none"},
        )
    ax.set_xticks(x, labels, fontproperties=CN)
    ax.set_ylabel("首位准确率（%）", fontproperties=CN)
    ax.set_ylim(76.8, 83.4)
    ax.set_yticks([77, 79, 81, 83])
    clean_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        frameon=False,
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.1,
        prop=CN,
    )
    ax.text(
        0.99,
        0.025,
        "GNPS Gold/Silver；[M+H]+；质量误差10 ppm",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MID,
        fontproperties=CN,
    )


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.2, 9.3), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.28,
        hspace=0.52,
        width_ratios=[1.0, 1.08],
    )

    draw_panel_a(fig.add_subplot(grid[0, 0]))
    draw_panel_b(fig.add_subplot(grid[0, 1]))
    draw_panel_c(fig.add_subplot(grid[1, 0]))
    draw_panel_d(fig.add_subplot(grid[1, 1]))

    for suffix, kwargs in {
        "png": {"dpi": 450},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(
            OUT_DIR / f"{OUT_STEM}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
