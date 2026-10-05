"""绘制图5 v2：紧凑六面板实验结果总览。

设计原则：
1. 每个面板只比较同一评价协议、同一分母上的结果；
2. 开发结果、预留独立测试和外部评价分开呈现；
3. 样本信息的准确率结果与不读取正确结构的事件计数分属两个面板；
4. 所有增益均由未四舍五入的原始值计算，避免二次舍入。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
OUT_STEM = "figure5_results_overview_cn_v2"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
CN = FontProperties(fname=str(FONT_REGULAR))
CN_BOLD = FontProperties(fname=str(FONT_BOLD))

INK = "#202020"
MID = "#777777"
LIGHT = "#D2D2D2"
PALE = "#F3F3F3"
GRID = "#E8E8E8"
ACCENT = "#92283B"
ACCENT_DARK = "#741C2D"
ACCENT_LIGHT = "#D3A0AA"
WARNING = "#A13A32"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": CN.get_name(),
            "font.size": 7.3,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def clean(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=7, color=INK, fontproperties=CN_BOLD)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=INK,
        fontproperties=CN_BOLD,
    )


def draw_a(ax: plt.Axes) -> None:
    panel(ax, "a")
    title(ax, "共享谱图模型：内部固定测试集")
    values = np.array([93.1875, 93.6835, 94.2399])
    x = np.arange(3)
    colors = [MID, ACCENT_LIGHT, ACCENT]
    markers = ["o", "s", "D"]
    ax.plot(x, values, color="#AFAFAF", lw=0.9, zorder=1)
    for xx, yy, color, marker in zip(x, values, colors, markers):
        ax.scatter(xx, yy, s=38, marker=marker, facecolor=color, edgecolor=INK, linewidth=0.55, zorder=3)
        ax.text(xx, yy + 0.075, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.6)
    ax.text(1.0, 94.39, "累计提高1.05个百分点", ha="center", va="bottom", color=ACCENT_DARK, fontsize=6.8, fontproperties=CN_BOLD)
    ax.set_xticks(x, ["原始DreaMS", "第一阶段", "最终模型"], fontproperties=CN)
    ax.set_ylim(92.95, 94.55)
    ax.set_yticks([93.0, 93.5, 94.0, 94.5])
    ax.set_ylabel("首位准确率（%）", fontproperties=CN)
    ax.text(0.98, 0.04, "n=18,333", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=MID, fontproperties=CN)
    clean(ax)


def draw_b(ax: plt.Axes) -> None:
    panel(ax, "b")
    title(ax, "共享谱图模型：GNPS外部评价")
    labels = [
        "身份隔离\n(n=10,995)",
        "分子式隔离\n(n=5,261)",
        "身份隔离近结构\n(n=9,592)",
        "分子式隔离近结构\n(n=2,944)",
    ]
    original = np.array([85.357, 86.809, 79.939, 77.989])
    stage1 = np.array([86.066, 87.797, 80.822, 79.416])
    final = np.array([86.558, 88.063, 81.441, 79.959])
    gains = ["+1.20", "+1.25", "+1.50", "+1.97"]
    y = np.arange(4)[::-1]
    for yy, base, mid, end, gain in zip(y, original, stage1, final, gains):
        ax.plot([base, mid, end], [yy, yy, yy], color="#AFAFAF", lw=0.85, zorder=1)
        ax.scatter(base, yy, s=25, marker="o", facecolor=MID, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.scatter(mid, yy, s=25, marker="s", facecolor=ACCENT_LIGHT, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.scatter(end, yy, s=29, marker="D", facecolor=ACCENT, edgecolor=INK, linewidth=0.45, zorder=3)
        ax.text(end + 0.45, yy, f"{gain}个百分点", ha="left", va="center", fontsize=6.2, color=ACCENT_DARK, fontproperties=CN_BOLD)
    ax.set_yticks(y, labels, fontproperties=CN)
    ax.set_xlim(76.5, 91.0)
    ax.set_xticks([78, 82, 86, 90])
    ax.set_xlabel("首位准确率（%）", fontproperties=CN)
    clean(ax, grid="x")
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=4.2, mfc=MID, mec=INK, mew=0.45, label="原始DreaMS"),
        plt.Line2D([], [], marker="s", ls="", ms=4.2, mfc=ACCENT_LIGHT, mec=INK, mew=0.45, label="第一阶段"),
        plt.Line2D([], [], marker="D", ls="", ms=4.2, mfc=ACCENT, mec=INK, mew=0.45, label="最终模型"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, -0.34), frameon=False, ncol=3, columnspacing=0.8, handletextpad=0.3, prop=CN)


def draw_c(ax: plt.Axes) -> None:
    panel(ax, "c")
    title(ax, "固定谱学后融合：预留独立测试")
    labels = ["主要测试集\n(n=3,000)", "极端近结构子集\n(n=496)"]
    baseline = np.array([87.93, 48.79])
    fusion = np.array([89.00, 44.56])
    changes = ["+1.07个百分点", "−4.23个百分点"]
    colors = [ACCENT_DARK, WARNING]
    y = np.arange(2)[::-1]
    for yy, base, end, change, color in zip(y, baseline, fusion, changes, colors):
        ax.plot([base, end], [yy, yy], color="#AFAFAF", lw=1.0, zorder=1)
        ax.scatter(base, yy, s=31, facecolor=LIGHT, edgecolor=INK, linewidth=0.5, zorder=3)
        ax.scatter(end, yy, s=34, facecolor=ACCENT, edgecolor=INK, linewidth=0.5, zorder=3)
        ax.text(base, yy + 0.17, f"{base:.2f}", ha="center", va="bottom", fontsize=6.4)
        ax.text(end, yy - 0.18, f"{end:.2f}", ha="center", va="top", fontsize=6.4, color=ACCENT_DARK, fontproperties=CN_BOLD)
        ax.text(max(base, end) + 2.2, yy, change, ha="left", va="center", fontsize=6.4, color=color, fontproperties=CN_BOLD)
    ax.set_yticks(y, labels, fontproperties=CN)
    ax.set_xlim(39, 98)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_xlabel("首位准确率（%）", fontproperties=CN)
    clean(ax, grid="x")
    ax.text(0.98, 0.04, "主要测试集：89个纠正 / 57个新增误判", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=MID, fontproperties=CN)


def draw_d(ax: plt.Axes) -> None:
    panel(ax, "d")
    title(ax, "候选化学证据重排：开发集评价")
    values = [90.4614, 94.4012]
    ax.bar([0, 1], values, width=0.58, color=[LIGHT, ACCENT], edgecolor=INK, linewidth=0.55, zorder=2)
    for xx, yy in enumerate(values):
        ax.text(xx, yy + 0.16, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.8, fontproperties=CN_BOLD)
    ax.plot([0, 0, 1, 1], [95.18, 95.34, 95.34, 95.18], color=INK, lw=0.7)
    ax.text(0.5, 95.42, "提高3.94个百分点 [2.82，5.19]", ha="center", va="bottom", fontsize=6.4, color=ACCENT_DARK, fontproperties=CN_BOLD)
    ax.set_ylim(89.0, 95.9)
    ax.set_yticks([90, 92, 94])
    ax.set_xticks([0, 1], ["基础排序", "化学证据重排"], fontproperties=CN)
    ax.set_ylabel("首位准确率（%）", fontproperties=CN)
    clean(ax)
    ax.text(0.50, -0.19, "n=1,929；93个纠正 / 17个新增误判", transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=MID, fontproperties=CN)
    ax.text(0.50, -0.30, "首位误判率：9.54% → 5.60%（相对下降41%）", transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=ACCENT_DARK, fontproperties=CN_BOLD)


def draw_e(ax: plt.Axes) -> None:
    panel(ax, "e")
    title(ax, "样本生化信息：开发集评价")
    baseline = 0.7541385652973636 * 100
    candidate = 0.78295524218271 * 100
    gain = candidate - baseline
    ax.plot([0, 1], [baseline, candidate], color="#AFAFAF", lw=1.0, zorder=1)
    ax.scatter(0, baseline, s=43, facecolor=LIGHT, edgecolor=INK, linewidth=0.55, zorder=3)
    ax.scatter(1, candidate, s=47, facecolor=ACCENT, edgecolor=INK, linewidth=0.55, zorder=3)
    ax.text(0, baseline + 0.16, f"{baseline:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.text(1, candidate + 0.16, f"{candidate:.2f}", ha="center", va="bottom", fontsize=6.8, fontproperties=CN_BOLD)
    ax.text(0.5, 79.10, f"按原始值计算：提高{gain:.2f}个百分点", ha="center", va="bottom", fontsize=6.5, color=ACCENT_DARK, fontproperties=CN_BOLD)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(74.4, 79.7)
    ax.set_yticks([75, 76, 77, 78, 79])
    ax.set_xticks([0, 1], ["基础排序", "加入样本生化信息"], fontproperties=CN)
    ax.set_ylabel("首位准确率（%）", fontproperties=CN)
    clean(ax)
    ax.text(0.50, -0.19, "n=1,631；50个纠正 / 3个新增误判", transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=MID, fontproperties=CN)
    ax.text(0.50, -0.30, "开发数据，尚非独立外部验证", transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=WARNING, fontproperties=CN_BOLD)


def draw_f(ax: plt.Axes) -> None:
    panel(ax, "f")
    title(ax, "真实样本事件对照：不评价准确率")
    values = [4996, 4072, 91]
    x = np.arange(3)
    ax.bar(x, values, width=0.60, color=[ACCENT, ACCENT_LIGHT, LIGHT], edgecolor=INK, linewidth=0.5, zorder=2)
    for xx, yy in zip(x, values):
        ax.text(xx, yy + 100, f"{yy:,}", ha="center", va="bottom", fontsize=6.8, fontproperties=CN_BOLD)
    ax.plot([0, 0, 1, 1], [5250, 5360, 5360, 5250], color=INK, lw=0.7)
    ax.text(0.5, 5400, "较最严格对照高22.69%", ha="center", va="bottom", fontsize=6.3, color=ACCENT_DARK, fontproperties=CN_BOLD)
    ax.set_ylim(0, 5750)
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticks(
        x,
        ["真实样本\n事件", "打乱高置信代谢物与\n样本的对应关系\n（20次最大值）", "保持网络度数的\n随机重连对照\n（最大值）"],
        fontproperties=CN,
    )
    ax.tick_params(axis="x", labelsize=5.9)
    ax.set_ylabel("候选特异事件查询数", fontproperties=CN)
    clean(ax)
    ax.text(0.50, -0.26, "51,976次检索；2,239个潜在排序机会", transform=ax.transAxes, ha="center", va="top", fontsize=6.1, color=MID, fontproperties=CN)
    ax.text(0.50, -0.37, "未读取正确结构；未计算准确率；事件分布集中度未达后续评价要求", transform=ax.transAxes, ha="center", va="top", fontsize=5.9, color=WARNING, fontproperties=CN_BOLD)


def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.2, 6.55), facecolor="white", constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.065,
        right=0.988,
        bottom=0.125,
        top=0.965,
        wspace=0.52,
        hspace=0.64,
        width_ratios=[0.92, 1.18, 1.0],
    )
    draw_a(fig.add_subplot(grid[0, 0]))
    draw_b(fig.add_subplot(grid[0, 1]))
    draw_c(fig.add_subplot(grid[0, 2]))
    draw_d(fig.add_subplot(grid[1, 0]))
    draw_e(fig.add_subplot(grid[1, 1]))
    draw_f(fig.add_subplot(grid[1, 2]))

    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
