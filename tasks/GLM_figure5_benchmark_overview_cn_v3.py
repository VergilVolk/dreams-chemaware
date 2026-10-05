"""绘制图5 v3：以效应量为中心的结果总览。

主面板展示各评价任务内相对各自基线的首位准确率变化；不同任务不
共用分母，不能相加。下方分别展示每千次查询的纠正/新增误判，以及
不读取正确结构的真实样本事件对照。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables"
OUT_STEM = "figure5_benchmark_overview_cn_v3"

CN = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
CN_BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

INK = "#202020"
MID = "#777777"
LIGHT = "#D5D5D5"
GRID = "#E7E7E7"
EXTERNAL = "#8E263B"
SEALED = "#B76070"
DEVELOPMENT = "#D5A2AB"
NEGATIVE = "#A23A32"
PALE_EXTERNAL = "#FAF2F4"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": CN.get_name(),
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
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


def clean(ax: plt.Axes, *, grid: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=INK,
        fontproperties=CN_BOLD,
    )


def draw_main_benchmark(ax: plt.Axes) -> None:
    panel_label(ax, "a")
    ax.set_title("各评价任务内相对各自基线的首位准确率变化", loc="left", pad=15, fontproperties=CN_BOLD)
    ax.text(
        0.0,
        1.015,
        "GNPS外部结果为本图重点；其余结果按证据阶段分层展示，不跨任务相加",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=MID,
        fontproperties=CN,
    )

    rows = [
        ("GNPS身份隔离｜外部（n=10,995）", 1.201, 0.460, 1.890, EXTERNAL, "外部评价"),
        ("GNPS分子式隔离｜外部（n=5,261）", 1.2545, 0.229, 2.299, EXTERNAL, "外部评价"),
        ("GNPS身份隔离近结构｜外部（n=9,592）", 1.502, 0.589, 2.464, EXTERNAL, "外部评价"),
        ("GNPS分子式隔离近结构｜外部（n=2,944）", 1.970, 0.251, 3.674, EXTERNAL, "外部评价"),
        ("共享模型｜内部固定测试（n=18,333）", 1.0524, None, None, SEALED, "内部固定测试"),
        ("固定谱学后融合｜预留主要测试（n=3,000）", 1.070, 0.240, 1.890, SEALED, "预留独立测试"),
        ("固定谱学后融合｜预留极端近结构（n=496）", -4.230, None, None, NEGATIVE, "预留独立测试"),
        ("候选化学证据重排｜开发（n=1,929）", 3.9399, 2.8191, 5.1921, DEVELOPMENT, "开发集"),
        ("样本生化信息｜开发（n=1,631）", 2.8817, 1.5528, 4.3880, DEVELOPMENT, "开发集"),
    ]
    y = np.arange(len(rows))[::-1]

    # 外部评价区使用极浅底色，形成明确的主视觉区，而不改变数据编码。
    ax.axhspan(y[3] - 0.5, y[0] + 0.5, color=PALE_EXTERNAL, zorder=-2)
    ax.text(-5.25, y[0] + 0.35, "独立GNPS外部评价", ha="left", va="bottom", fontsize=6.5, color=EXTERNAL, fontproperties=CN_BOLD)

    for yy, (label, effect, low, high, color, _) in zip(y, rows):
        ax.barh(yy, effect, left=0, height=0.52, color=color, edgecolor="none", zorder=2)
        if low is not None and high is not None:
            ax.errorbar(
                effect,
                yy,
                xerr=[[effect - low], [high - effect]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.8,
                capsize=2.1,
                capthick=0.7,
                zorder=4,
            )
        ha = "left" if effect >= 0 else "right"
        offset = 0.18 if effect >= 0 else -0.18
        value = f"{effect:+.2f}"
        ax.text(effect + offset, yy, value, ha=ha, va="center", fontsize=6.6, color=INK, fontproperties=CN_BOLD)

    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows], fontproperties=CN)
    ax.set_xlim(-5.6, 6.2)
    ax.set_xticks([-4, -2, 0, 2, 4, 6])
    ax.set_xlabel("首位准确率变化（百分点；95%置信区间）", fontproperties=CN)
    clean(ax)

    ax.text(0.0, -0.16, "未绘制置信区间的条目没有可直接对应的单一累计区间。", transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color=MID, fontproperties=CN)


def draw_risk_benefit(ax: plt.Axes) -> None:
    panel_label(ax, "b")
    ax.set_title("排序改变的收益与风险", loc="left", pad=19, fontproperties=CN_BOLD)
    ax.text(0.0, 1.01, "统一换算为每1,000次查询", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, color=MID, fontproperties=CN)
    ax.text(1.0, 1.01, "左：新增误判　右：纠正误判", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=INK, fontproperties=CN)

    labels = [
        "GNPS身份隔离",
        "GNPS分子式隔离",
        "固定谱学后融合",
        "化学证据重排",
        "样本生化信息",
    ]
    n = np.array([10995, 5261, 3000, 1929, 1631], dtype=float)
    corrected = np.array([328, 156, 89, 93, 50], dtype=float) / n * 1000
    introduced = np.array([196, 90, 57, 17, 3], dtype=float) / n * 1000
    y = np.arange(len(labels))[::-1]

    ax.barh(y, -introduced, height=0.50, color=LIGHT, edgecolor="none", label="新增误判", zorder=2)
    ax.barh(y, corrected, height=0.50, color=EXTERNAL, edgecolor="none", label="纠正误判", zorder=2)
    ax.axvline(0, color=INK, lw=0.8)
    for yy, c, i in zip(y, corrected, introduced):
        ax.text(c + 1.2, yy, f"{c:.1f}", ha="left", va="center", fontsize=6.2)
        ax.text(-i - 1.2, yy, f"{i:.1f}", ha="right", va="center", fontsize=6.2)
    ax.set_yticks(y, labels, fontproperties=CN)
    ax.set_xlim(-24, 55)
    ax.set_xticks([-20, 0, 20, 40])
    ax.set_xlabel("每1,000次查询的数量", fontproperties=CN)
    clean(ax)


def draw_event_control(ax: plt.Axes) -> None:
    panel_label(ax, "c")
    ax.set_title("真实样本中的候选特异事件", loc="left", pad=19, fontproperties=CN_BOLD)
    ax.text(0.0, 1.01, "不读取正确结构，不评价注释准确率", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, color=NEGATIVE, fontproperties=CN_BOLD)

    labels = [
        "真实样本事件",
        "打乱高置信代谢物与样本的对应关系\n（20次最大值）",
        "保持网络度数的随机重连对照\n（最大值）",
    ]
    values = np.array([4996, 4072, 91])
    colors = [EXTERNAL, DEVELOPMENT, LIGHT]
    y = np.arange(3)[::-1]
    ax.barh(y, values, height=0.54, color=colors, edgecolor=INK, linewidth=0.45, zorder=2)
    for yy, value in zip(y, values):
        ax.text(value + 90, yy, f"{value:,}", ha="left", va="center", fontsize=6.6, fontproperties=CN_BOLD)
    ax.set_yticks(y, labels, fontproperties=CN)
    ax.set_xlim(0, 5600)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel("具有候选特异事件的查询数", fontproperties=CN)
    clean(ax)
    ax.text(3650, y[0], "较置换对照最大值高22.69%", ha="center", va="center", fontsize=6.3, color="white", fontproperties=CN_BOLD)
    ax.text(0.99, 0.05, "51,976次检索；2,239个潜在排序机会", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=MID, fontproperties=CN)
    ax.text(0.0, -0.22, "事件分布集中度尚未达到进入后续排序评价的要求。", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=NEGATIVE, fontproperties=CN_BOLD)


def main() -> None:
    configure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 7.45), facecolor="white", constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.19,
        right=0.985,
        bottom=0.115,
        top=0.965,
        hspace=0.58,
        wspace=0.48,
        height_ratios=[1.45, 1.0],
        width_ratios=[1.0, 1.0],
    )
    draw_main_benchmark(fig.add_subplot(grid[0, :]))
    draw_risk_benefit(fig.add_subplot(grid[1, 0]))
    draw_event_control(fig.add_subplot(grid[1, 1]))

    for suffix, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
