#!/usr/bin/env python
"""Render the reader-facing DreaMS-VOLAR manuscript figure suite.

This suite replaces the internal Full23 engineering visualizations with figures
organized around a new reader's questions:

1. What does DreaMS-VOLAR do?
2. Does it beat public spectral-similarity methods on the same candidate axis?
3. Which decisions change, and why?
4. Which evidence families contribute beyond the public baselines?

`Full23 v1.3.2` remains the frozen implementation alias.  GNPS and Enveda are
consumed development resources; CDC/HUMAN are not shown as scored benchmarks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import render_full23_v132_manuscript_suite as legacy  # noqa: E402


ROOT = legacy.ROOT
COMMON = legacy.COMMON
PANELS = legacy.PANELS
PANEL_SHORT = legacy.PANEL_SHORT
METHOD_ID = "full23_v132"
METHOD_NAME = "DreaMS-VOLAR"
IMPLEMENTATION = "Full23 v1.3.2"
EXPANDED_NAME = (
    "Vector, Orthogonal and Local Evidence with an Additive Residual "
    "for MS/MS Candidate Ranking"
)

PUBLIC_METHODS = (
    METHOD_ID,
    "weighted_spectral_entropy",
    "official_dreams",
    "entropy_raw_public",
    "denoising_search_public",
    "spec2vec_gnps_public",
    "ms2deepscore_2x_public",
    "msbert_official",
    "specembedding_official",
)

DISPLAY = dict(legacy.METHOD_NAMES)
DISPLAY[METHOD_ID] = "DreaMS-VOLAR (ours)"

COLORS = {
    METHOD_ID: "#1557B0",
    "weighted_spectral_entropy": "#E68613",
    "official_dreams": "#7A5195",
    "entropy_raw_public": "#D9A441",
    "denoising_search_public": "#B26A5B",
    "spec2vec_gnps_public": "#2E8B57",
    "ms2deepscore_2x_public": "#E4578E",
    "msbert_official": "#4C78A8",
    "specembedding_official": "#7F7F7F",
}

HIGHLIGHT_CURVES = (
    METHOD_ID,
    "weighted_spectral_entropy",
    "official_dreams",
    "spec2vec_gnps_public",
    "ms2deepscore_2x_public",
    "msbert_official",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "deliverables/dreams_volar_publication_figures_20261102",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[str]:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return [f"{stem.name}.{suffix}" for suffix in ("png", "pdf", "svg")]


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.11, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str,
    radius: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.3,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#6b7280") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def draw_spectrum(ax: plt.Axes, origin: tuple[float, float], size: tuple[float, float], color: str) -> None:
    x0, y0 = origin
    width, height = size
    peaks = np.asarray(
        [
            (0.05, 0.18),
            (0.12, 0.44),
            (0.18, 0.24),
            (0.29, 0.72),
            (0.38, 0.31),
            (0.47, 0.91),
            (0.58, 0.40),
            (0.66, 0.62),
            (0.78, 0.28),
            (0.88, 0.52),
            (0.95, 0.22),
        ]
    )
    ax.plot([x0, x0 + width], [y0, y0], color="#475569", lw=0.8)
    for px, py in peaks:
        ax.plot([x0 + px * width, x0 + px * width], [y0, y0 + py * height], color=color, lw=1.8)


def draw_molecule(ax: plt.Axes, center: tuple[float, float], scale: float, color: str) -> None:
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    xy = np.c_[center[0] + scale * np.cos(angles), center[1] + scale * np.sin(angles)]
    ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.4))
    ax.plot([xy[1, 0], xy[1, 0] + 0.7 * scale], [xy[1, 1], xy[1, 1] + 0.55 * scale], color=color, lw=1.2)
    ax.add_patch(Circle((xy[4, 0] - 0.45 * scale, xy[4, 1] - 0.4 * scale), 0.13 * scale, fill=False, edgecolor=color, lw=1.0))


def figure_method_overview(output: Path, dpi: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(13.2, 6.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.98,
        "DreaMS-VOLAR converts complementary spectral and chemical evidence into one candidate decision",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.935,
        EXPANDED_NAME.replace(" for MS/MS Candidate Ranking", ""),
        ha="center",
        color="#475569",
        fontsize=10,
    )

    # Query and candidate set.
    rounded_box(ax, (0.02, 0.19), 0.15, 0.61, "#EFF6FF", "#1557B0")
    ax.text(0.095, 0.755, "Query MS/MS", ha="center", fontweight="bold", color="#0F3D75")
    draw_spectrum(ax, (0.04, 0.54), (0.11, 0.15), "#1557B0")
    ax.text(0.095, 0.49, "precursor mass", ha="center", fontsize=8, color="#475569")
    ax.text(0.095, 0.465, "+ acquisition context", ha="center", fontsize=8, color="#475569")
    ax.text(0.095, 0.405, "10-ppm candidate set", ha="center", fontsize=8.5, fontweight="bold")
    for yy, cc in zip((0.35, 0.29, 0.23), ("#1557B0", "#64748B", "#64748B")):
        draw_molecule(ax, (0.062, yy), 0.018, cc)
        ax.add_patch(FancyBboxPatch((0.087, yy - 0.018), 0.055, 0.036, boxstyle="round,pad=0.004", facecolor="white", edgecolor="#CBD5E1", lw=0.8))

    # Three evidence lanes.
    lane_x, lane_w = 0.22, 0.28
    lanes = (
        (
            0.66,
            "Global vector evidence",
            "condition-robust + chemistry-refined spectral embeddings",
            "#E8F1FB",
            "#2468B4",
        ),
        (
            0.42,
            "Local fragmentation evidence",
            "weighted entropy + peak-pair + learned local matching",
            "#E9F7EF",
            "#27864B",
        ),
        (
            0.18,
            "Candidate-specific chemistry",
            "observed chemical support − matched null support",
            "#F4ECFA",
            "#7A5195",
        ),
    )
    for yy, title, subtitle, face, edge in lanes:
        rounded_box(ax, (lane_x, yy), lane_w, 0.17, face, edge)
        ax.text(lane_x + 0.075, yy + 0.127, title, fontweight="bold", color=edge, fontsize=10)
        ax.text(lane_x + 0.075, yy + 0.093, subtitle, color="#334155", fontsize=7.8)
    # Vector glyph.
    rng = np.random.default_rng(20261102)
    for center, color in (((0.255, 0.715), "#2468B4"), ((0.292, 0.755), "#66A3D2")):
        pts = rng.normal(center, (0.012, 0.018), size=(10, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.85)
    # Local mirror glyph.
    draw_spectrum(ax, (0.238, 0.495), (0.055, 0.07), "#27864B")
    ax.plot([0.238, 0.293], [0.485, 0.485], color="#475569", lw=0.7)
    for xx, hh in ((0.245, 0.035), (0.258, 0.055), (0.276, 0.027), (0.287, 0.047)):
        ax.plot([xx, xx], [0.485, 0.485 - hh], color="#65A976", lw=1.3)
    # Chemistry real/null glyph.
    ax.scatter([0.255], [0.252], s=65, color="#7A5195", zorder=3)
    ax.scatter([0.282, 0.292, 0.302], [0.235, 0.247, 0.228], s=25, facecolor="white", edgecolor="#A98AC1", zorder=3)
    ax.plot([0.255, 0.302], [0.252, 0.228], color="#A98AC1", lw=1.0, ls="--")

    arrow(ax, (0.17, 0.50), (0.22, 0.745))
    arrow(ax, (0.17, 0.50), (0.22, 0.505))
    arrow(ax, (0.17, 0.50), (0.22, 0.265))

    # Fusion and residual.
    rounded_box(ax, (0.56, 0.31), 0.19, 0.38, "#FFF7E6", "#D97706")
    ax.text(0.655, 0.635, "Evidence integration", ha="center", fontweight="bold", color="#9A4D00", fontsize=11)
    ax.text(0.655, 0.585, "learned additive parent", ha="center", fontsize=8.5)
    ax.text(0.655, 0.55, "+", ha="center", fontsize=15, fontweight="bold", color="#D97706")
    ax.text(0.655, 0.515, "query-centred", ha="center", fontsize=8.5)
    ax.text(0.655, 0.486, "full-embedding residual", ha="center", fontsize=8.5)
    ax.text(0.655, 0.42, r"$S(q,c)=S_{add}(q,c)+0.9\,r(q,c)$", ha="center", fontsize=10.5, color="#7C3F00")
    ax.text(0.655, 0.36, "missing evidence stays unavailable", ha="center", fontsize=7.5, color="#64748B")
    for yy in (0.745, 0.505, 0.265):
        arrow(ax, (0.50, yy), (0.56, 0.50))

    # Ranked result.
    rounded_box(ax, (0.81, 0.19), 0.17, 0.61, "#F8FAFC", "#334155")
    ax.text(0.895, 0.755, "Ranked identity hypotheses", ha="center", fontweight="bold", color="#1E293B")
    cards = ((0.64, 0.93, "#1557B0", "1"), (0.49, 0.76, "#64748B", "2"), (0.34, 0.61, "#64748B", "3"))
    for yy, score, color, rank in cards:
        face = "#E8F1FB" if rank == "1" else "white"
        rounded_box(ax, (0.83, yy - 0.055), 0.13, 0.105, face, color, radius=0.014)
        ax.text(0.842, yy + 0.022, rank, color=color, fontweight="bold", fontsize=11)
        draw_molecule(ax, (0.88, yy), 0.021, color)
        ax.text(0.944, yy, f"{score:.2f}", ha="right", va="center", color=color, fontweight="bold")
    ax.text(0.895, 0.245, "Top-1 identity", ha="center", fontweight="bold", color="#1557B0")
    ax.text(0.895, 0.215, "with an auditable evidence record", ha="center", fontsize=7.5, color="#475569")
    arrow(ax, (0.75, 0.50), (0.81, 0.50))

    ax.text(0.02, 0.84, "a", fontsize=13, fontweight="bold")
    ax.text(0.20, 0.84, "b", fontsize=13, fontweight="bold")
    ax.text(0.54, 0.72, "c", fontsize=13, fontweight="bold")
    ax.text(0.79, 0.84, "d", fontsize=13, fontweight="bold")
    ax.text(
        0.5,
        0.06,
        f"Public method name: {METHOD_NAME} v1.0  ·  frozen implementation: {IMPLEMENTATION}",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    return save(fig, output / "fig01_dreams_volar_method_overview", dpi)


def common_frames() -> dict[str, pd.DataFrame]:
    return legacy.common_frames()


def curve(panel: str, method: str, kind: str) -> pd.DataFrame:
    return legacy.load_curve(legacy.curve_path(panel, method, kind), maximum=4500)


def draw_lollipop(ax: plt.Axes, frame: pd.DataFrame, panel: str) -> None:
    data = frame.loc[list(PUBLIC_METHODS)].sort_values("recall@1", ascending=True)
    yy = np.arange(len(data))
    xmin = max(70.0, 100 * data["recall@1"].min() - 1.0)
    for y, (method, row) in enumerate(data.iterrows()):
        value = 100 * float(row["recall@1"])
        ax.hlines(y, xmin, value, color="#D5DAE1", lw=1.4)
        ax.scatter(value, y, s=80 if method == METHOD_ID else 42, color=COLORS[method], edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(value + 0.16, y, f"{value:.2f}", va="center", fontsize=7.2, fontweight="bold" if method == METHOD_ID else "normal")
    ax.set_yticks(yy, [DISPLAY[m] for m in data.index], fontsize=7.4)
    ax.set_xlim(xmin, 91.4)
    ax.set_xlabel("Recall@1 (%)")
    ax.grid(axis="x", alpha=0.18)
    ax.set_title(f"Public-method identity ranking\n{PANEL_SHORT[panel]} (n={int(frame.iloc[0]['N_Q']):,})")
    delta = 100 * float(frame.loc[METHOD_ID, "recall@1"] - frame.loc["official_dreams", "recall@1"])
    ax.text(
        0.98,
        0.93,
        f"+{delta:.2f} pp vs DreaMS",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#0F3D75",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#E8F1FB", "edgecolor": "none"},
    )


def draw_topk(ax: plt.Axes, frame: pd.DataFrame, panel: str) -> None:
    ks = (1, 2, 3, 5, 10, 20)
    for method in PUBLIC_METHODS:
        values = [100 * float(frame.loc[method, f"recall@{k}"]) for k in ks]
        ax.plot(
            ks,
            values,
            marker="o",
            ms=3.0,
            lw=2.7 if method == METHOD_ID else 1.15,
            color=COLORS[method],
            alpha=1 if method == METHOD_ID else 0.82,
            label=DISPLAY[method],
        )
    ax.set_xticks(ks)
    ax.set_ylim(74, 100.5)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (%)")
    ax.set_title(f"Rank recovery\n{PANEL_SHORT[panel]}")
    ax.grid(alpha=0.18)


def draw_roc(ax: plt.Axes, frame: pd.DataFrame, panel: str) -> None:
    for method in frame.index:
        roc = curve(panel, method, "roc")
        ax.plot(roc["false_positive_rate"], roc["true_positive_rate"], color="#C7CCD3", lw=0.65, alpha=0.48, zorder=1)
    for method in HIGHLIGHT_CURVES:
        roc = curve(panel, method, "roc")
        label = f"{DISPLAY[method]} ({frame.loc[method, 'micro_candidate_auroc']:.3f})"
        ax.plot(
            roc["false_positive_rate"],
            roc["true_positive_rate"],
            color=COLORS[method],
            lw=2.5 if method == METHOD_ID else 1.3,
            ls="--" if method == "ms2deepscore_2x_public" else "-",
            label=label,
            zorder=3,
        )
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"Candidate discrimination\n{PANEL_SHORT[panel]} (legend: AUROC)")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=6.2, loc="lower right")


def figure_public_benchmark(output: Path, dpi: int, frames: dict[str, pd.DataFrame]) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0), constrained_layout=True)
    for row, panel in enumerate(PANELS):
        draw_lollipop(axes[row, 0], frames[panel], panel)
        draw_topk(axes[row, 1], frames[panel], panel)
        draw_roc(axes[row, 2], frames[panel], panel)
    for ax, letter in zip(axes.ravel(), "abcdef"):
        panel_label(ax, letter)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=5, fontsize=7.3)
    fig.suptitle(
        "DreaMS-VOLAR outperforms public spectral-similarity methods on a shared candidate axis",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.065,
        "GNPS common-domain development comparison. MS2DeepScore 2.x is dashed because its native objective is structural similarity, not exact identity.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    return save(fig, output / "fig02_public_method_benchmark", dpi)


def draw_pr(ax: plt.Axes, frame: pd.DataFrame, panel: str) -> None:
    for method in frame.index:
        pr = curve(panel, method, "pr")
        ax.plot(pr["recall"], pr["precision"], color="#C7CCD3", lw=0.65, alpha=0.45, zorder=1)
    for method in HIGHLIGHT_CURVES:
        pr = curve(panel, method, "pr")
        label = f"{DISPLAY[method]} ({frame.loc[method, 'micro_candidate_average_precision']:.3f})"
        ax.plot(
            pr["recall"],
            pr["precision"],
            color=COLORS[method],
            lw=2.5 if method == METHOD_ID else 1.3,
            ls="--" if method == "ms2deepscore_2x_public" else "-",
            label=label,
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{PANEL_SHORT[panel]} precision-recall (legend: AP/AUPRC)")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=6.4, loc="lower left")
    delta = 100 * float(
        frame.loc[METHOD_ID, "micro_candidate_average_precision"]
        - frame.loc["weighted_spectral_entropy", "micro_candidate_average_precision"]
    )
    ax.text(
        0.97,
        0.93,
        f"+{delta:.2f} pp AP vs WSE",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#0F3D75",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#E8F1FB", "edgecolor": "none"},
    )


def query_metrics(panel: str) -> pd.DataFrame:
    directory = COMMON / ("identity" if panel == "identity_disjoint" else "formula")
    return pd.read_csv(directory / "query_metrics.csv.gz", compression="gzip")


def flow_patch(
    ax: plt.Axes,
    left: tuple[float, float],
    right: tuple[float, float],
    color: str,
    alpha: float,
) -> None:
    x0, x1 = 0.18, 0.82
    control_left, control_right = 0.43, 0.57
    l0, l1 = left
    r0, r1 = right
    path = MplPath(
        [
            (x0, l0),
            (control_left, l0),
            (control_right, r0),
            (x1, r0),
            (x1, r1),
            (control_right, r1),
            (control_left, l1),
            (x0, l1),
            (x0, l0),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ],
    )
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha))


def draw_decision_flow(ax: plt.Axes, panel: str) -> dict[str, int]:
    query = query_metrics(panel)
    subset = query[query["method"].isin((METHOD_ID, "weighted_spectral_entropy"))]
    ranks = subset.pivot(index="query_index", columns="method", values="rank")
    wse_correct = ranks["weighted_spectral_entropy"].to_numpy() <= 1
    volar_correct = ranks[METHOD_ID].to_numpy() <= 1
    retained = int(np.sum(wse_correct & volar_correct))
    corrected = int(np.sum(~wse_correct & volar_correct))
    introduced = int(np.sum(wse_correct & ~volar_correct))
    unresolved = int(np.sum(~wse_correct & ~volar_correct))
    total = int(len(ranks))

    left_wrong_top = (corrected + unresolved) / total
    right_wrong_top = (introduced + unresolved) / total
    left_segments = {
        "unresolved": (0.0, unresolved / total),
        "corrected": (unresolved / total, left_wrong_top),
        "introduced": (left_wrong_top, left_wrong_top + introduced / total),
        "retained": (left_wrong_top + introduced / total, 1.0),
    }
    right_segments = {
        "unresolved": (0.0, unresolved / total),
        "introduced": (unresolved / total, right_wrong_top),
        "corrected": (right_wrong_top, right_wrong_top + corrected / total),
        "retained": (right_wrong_top + corrected / total, 1.0),
    }
    palette = {
        "retained": ("#86B6E2", 0.64),
        "corrected": ("#27A35A", 0.82),
        "introduced": ("#D64545", 0.82),
        "unresolved": ("#C7CDD4", 0.60),
    }
    for state in ("unresolved", "retained", "introduced", "corrected"):
        flow_patch(ax, left_segments[state], right_segments[state], *palette[state])

    for x, wrong_top, title in (
        (0.18, left_wrong_top, "WSE"),
        (0.82, right_wrong_top, "DreaMS-VOLAR"),
    ):
        ax.plot([x, x], [0, 1], color="#243447", lw=8, solid_capstyle="butt", zorder=5)
        ax.plot([x - 0.015, x + 0.015], [wrong_top, wrong_top], color="white", lw=1.2, zorder=6)
        side = -0.03 if x > 0.5 else 0.03
        align = "right" if x > 0.5 else "left"
        ax.text(x, 1.055, title, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x + side, (1 + wrong_top) / 2, "correct", ha=align, va="center", fontsize=7.2, color="#1557B0", fontweight="bold")
        ax.text(x + side, wrong_top / 2, "wrong", ha=align, va="center", fontsize=7.2, color="#64748B")

    ax.text(0.50, 0.95, f"retained correct  {retained:,}", ha="center", va="top", fontsize=7.4, color="#285F91")
    ax.text(0.50, left_segments["corrected"][0] + 0.006, f"corrected  +{corrected}", ha="center", va="bottom", fontsize=8, color="#178043", fontweight="bold")
    ax.text(0.50, right_segments["introduced"][1] - 0.006, f"introduced  {introduced}", ha="center", va="top", fontsize=8, color="#B92D2D", fontweight="bold")
    ax.text(0.50, 0.035, f"still wrong  {unresolved:,}", ha="center", va="bottom", fontsize=7.4, color="#64748B")
    ax.text(0.50, 1.105, f"net gain: +{corrected - introduced} correct Top-1 decisions", ha="center", fontsize=8.2, color="#0F3D75", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.14)
    ax.set_title(f"Top-1 decision flow · {PANEL_SHORT[panel]}")
    ax.axis("off")
    return {"corrected": corrected, "introduced": introduced, "queries": total}


def figure_precision_decisions(output: Path, dpi: int, frames: dict[str, pd.DataFrame]) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), constrained_layout=True)
    exchanges: list[dict[str, object]] = []
    for column, panel in enumerate(PANELS):
        draw_pr(axes[0, column], frames[panel], panel)
        counts = draw_decision_flow(axes[1, column], panel)
        exchanges.append({"panel": panel, **counts})
    for ax, letter in zip(axes.ravel(), "abcd"):
        panel_label(ax, letter)
    fig.suptitle(
        "DreaMS-VOLAR improves candidate precision and changes specific Top-1 decisions",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(0.5, -0.015, "Flow width equals the exact fraction of shared queries.", ha="center", fontsize=8, color="#475569")
    pd.DataFrame(exchanges).to_csv(output / "wse_decision_exchange.csv", index=False)
    return save(fig, output / "fig03_precision_and_decision_flow", dpi)


def ablation_records() -> pd.DataFrame:
    reports = {
        "Enveda identity": legacy.load_json(legacy.ENVEDA_ABLATION),
        "Enveda formula": legacy.load_json(legacy.ENVEDA_ABLATION),
        "GNPS identity": legacy.load_json(legacy.GNPS_ABLATION),
        "GNPS formula": legacy.load_json(legacy.GNPS_ABLATION),
    }
    panels = {
        "Enveda identity": "identity_disjoint",
        "Enveda formula": "formula_disjoint",
        "GNPS identity": "identity_disjoint",
        "GNPS formula": "formula_disjoint",
    }
    variants = {
        "All self-trained evidence\nvs official DreaMS + WSE": "official_dreams_wse_additive",
        "Representation family\nStage-1 / Phase-A / FullMSG": "without_chemaware_representation_group",
        "Candidate chemistry\nChemAware V2": "without_chemaware_v2",
        "Local reranking\nP2b + MSG": "without_p2b_msg",
    }
    rows = []
    for dataset, report in reports.items():
        panel = panels[dataset]
        metrics = legacy._ablation_metrics(report, panel)
        full = metrics["full_23_additive"]["recall@1"]
        for label, variant in variants.items():
            rows.append(
                {
                    "dataset_panel": dataset,
                    "evidence_family": label,
                    "delta_pp": 100 * (full - metrics[variant]["recall@1"]),
                }
            )
    return pd.DataFrame(rows)


def draw_bubble_ablation(ax: plt.Axes, records: pd.DataFrame) -> None:
    columns = ("Enveda identity", "Enveda formula", "GNPS identity", "GNPS formula")
    rows = tuple(records["evidence_family"].drop_duplicates())
    for yi, row in enumerate(rows):
        for xi, column in enumerate(columns):
            value = float(records[(records["evidence_family"] == row) & (records["dataset_panel"] == column)]["delta_pp"].iloc[0])
            color = "#1557B0" if value >= 0 else "#D64545"
            size = 220 + 470 * min(abs(value) / 2.8, 1.0)
            ax.scatter(xi, yi, s=size, color=color, alpha=0.86, edgecolor="white", linewidth=1.0)
            ax.text(xi, yi, f"{value:+.2f}", ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, len(columns) - 0.45)
    ax.set_title("Strict-refit evidence contribution (pp Recall@1)")
    ax.grid(alpha=0.13)


def comparator_exchange(panel: str, references: tuple[str, ...]) -> pd.DataFrame:
    query = query_metrics(panel)
    subset = query[query["method"].isin((METHOD_ID,) + references)]
    ranks = subset.pivot(index="query_index", columns="method", values="rank")
    rows = []
    for reference in references:
        corrected = int(((ranks[reference] > 1) & (ranks[METHOD_ID] <= 1)).sum())
        introduced = int(((ranks[reference] <= 1) & (ranks[METHOD_ID] > 1)).sum())
        rows.append({"reference": reference, "corrected": corrected, "introduced": introduced, "net": corrected - introduced})
    return pd.DataFrame(rows)


def draw_butterfly(ax: plt.Axes, panel: str, exchange: pd.DataFrame) -> None:
    yy = np.arange(len(exchange))[::-1]
    ax.barh(yy, exchange["corrected"], color="#27A35A", alpha=0.9, label="Corrected")
    ax.barh(yy, -exchange["introduced"], color="#D64545", alpha=0.9, label="Introduced")
    ax.axvline(0, color="#4B5563", lw=0.8)
    ax.set_yticks(yy, [DISPLAY[m] for m in exchange["reference"]], fontsize=7.8)
    limit = max(exchange["corrected"].max(), exchange["introduced"].max()) * 1.16
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("queries lost  ←  decision exchange  →  queries corrected")
    ax.set_title(f"Top-1 exchange · {PANEL_SHORT[panel]}")
    for y, row in zip(yy, exchange.itertuples()):
        ax.text(row.corrected + 0.015 * limit, y, f"net +{row.net}", va="center", fontsize=7.3, fontweight="bold")
    ax.legend(fontsize=7.2, loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2)
    ax.grid(axis="x", alpha=0.15)


def figure_evidence_exchange(output: Path, dpi: int) -> list[str]:
    references = (
        "weighted_spectral_entropy",
        "official_dreams",
        "spec2vec_gnps_public",
        "ms2deepscore_2x_public",
        "msbert_official",
    )
    records = ablation_records()
    identity = comparator_exchange("identity_disjoint", references)
    formula = comparator_exchange("formula_disjoint", references)
    fig = plt.figure(figsize=(15.0, 7.6), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=(1.05, 1.15))
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    draw_bubble_ablation(ax_a, records)
    draw_butterfly(ax_b, "identity_disjoint", identity)
    draw_butterfly(ax_c, "formula_disjoint", formula)
    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, letter)
    fig.suptitle(
        "Complementary evidence contributes beyond public baselines",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Ablations are Full23-v1 strict refits on consumed Enveda/GNPS development resources; exchanges use the shared GNPS common domain.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    records.to_csv(output / "strict_refit_evidence_contribution.csv", index=False)
    pd.concat(
        [identity.assign(panel="identity_disjoint"), formula.assign(panel="formula_disjoint")],
        ignore_index=True,
    ).to_csv(output / "public_method_error_exchange.csv", index=False)
    return save(fig, output / "fig04_evidence_contribution_and_error_exchange", dpi)


def figure_rank_profile(output: Path, dpi: int, frames: dict[str, pd.DataFrame]) -> list[str]:
    metrics = (
        "recall@1",
        "recall@3",
        "recall@5",
        "mrr",
        "macro_query_auroc",
        "micro_candidate_auroc",
        "micro_candidate_average_precision",
        "near_recall@1",
    )
    labels = ("R@1", "R@3", "R@5", "MRR", "Macro\nAUROC", "Micro\nAUROC", "Micro\nAUPRC", "Near\nR@1")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    rows: list[dict[str, object]] = []
    for ax, panel, letter in zip(axes, PANELS, "ab"):
        frame = frames[panel].loc[list(PUBLIC_METHODS), list(metrics)]
        ranks = frame.rank(axis=0, ascending=False, method="min")
        for method in PUBLIC_METHODS:
            values = ranks.loc[method].to_numpy(float)
            ax.plot(
                np.arange(len(metrics)),
                values,
                marker="o",
                ms=4.2 if method == METHOD_ID else 2.8,
                lw=2.8 if method == METHOD_ID else 1.0,
                color=COLORS[method],
                alpha=1 if method == METHOD_ID else 0.72,
                label=DISPLAY[method],
            )
            for metric, rank in zip(metrics, values):
                rows.append({"panel": panel, "method": method, "metric": metric, "rank": rank})
        ax.set_xticks(np.arange(len(metrics)), labels)
        ax.set_ylim(len(PUBLIC_METHODS) + 0.4, 0.6)
        ax.set_ylabel("Rank among public methods (1 = best)")
        ax.set_title(PANEL_SHORT[panel])
        ax.grid(alpha=0.18)
        panel_label(ax, letter)
    handles, labels_out = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=7.4)
    fig.suptitle("Multi-metric performance profile across public methods", fontsize=14, fontweight="bold")
    pd.DataFrame(rows).to_csv(output / "public_method_metric_ranks.csv", index=False)
    return save(fig, output / "figS01_public_method_rank_profile", dpi)


def write_captions(output: Path) -> None:
    text = f"""# DreaMS-VOLAR manuscript figure captions

## Figure 1 | DreaMS-VOLAR integrates complementary evidence before candidate ranking

DreaMS-VOLAR (Vector, Orthogonal and Local Evidence with an Additive Residual)
starts from a query spectrum and a precursor-compatible candidate set. Four
condition-robust representation streams provide global vector evidence; WSE,
P2b-on-Noise and Noise-MSG provide local fragmentation evidence; and ChemAware
V2 supplies candidate-specific real-versus-matched-null chemical evidence. A
learned additive parent is complemented by a query-centred full-embedding
residual. The public DreaMS-VOLAR v1.0 implementation is frozen internally as
{IMPLEMENTATION}.

## Figure 2 | Public-method benchmark on a shared candidate axis

Recall@1 ladders, Top-k curves and candidate-level ROC curves compare
DreaMS-VOLAR with weighted spectral entropy, official fine-tuned DreaMS,
entropy and denoising search, Spec2Vec, MS2DeepScore 2.x, MSBERT and
SpecEmbedding. All methods use the same GNPS candidate axis: 10,983
identity-disjoint and 5,253 formula-disjoint queries. Thin grey ROC curves show
all 18 evaluated scoring streams. MS2DeepScore 2.x is an auxiliary comparator
because its native learning objective is structural similarity rather than
exact identity.

## Figure 3 | Precision-recall and decision changes

Candidate-level precision-recall curves are paired with query-level decision
flows relative to WSE. Flow width is the exact fraction of shared queries.
Green paths are WSE errors corrected by DreaMS-VOLAR, red paths are newly
introduced errors, blue paths retain a correct Top-1 decision, and grey paths
remain unresolved.

## Figure 4 | Evidence contribution and paired error exchange

Bubble size and labels show the Full23-v1 strict-refit Recall@1 advantage over
each evidence-family ablation on consumed Enveda and GNPS development panels.
The butterfly plots show DreaMS-VOLAR Top-1 errors corrected (right) and correct
decisions lost (left) relative to public comparison methods on the shared GNPS
domain. Internal module streams are intentionally separated from external/public
method comparisons.

## Figure S1 | Multi-metric public-method rank profile

Within-panel ranks across Recall@1/3/5, MRR, macro- and micro-AUROC,
micro-AUPRC and near-subset Recall@1. Rank 1 denotes the best result among the
nine reader-facing public methods.

Scientific boundary: GNPS and Enveda are consumed development resources. These
figures support same-axis method development comparisons, not an unopened
external SOTA claim.
"""
    (output / "CAPTIONS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = arguments()
    if args.dpi < 150:
        raise ValueError("dpi must be at least 150")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()
    frames = common_frames()
    generated: list[str] = []
    generated += figure_method_overview(output, args.dpi)
    generated += figure_public_benchmark(output, args.dpi, frames)
    generated += figure_precision_decisions(output, args.dpi, frames)
    generated += figure_evidence_exchange(output, args.dpi)
    generated += figure_rank_profile(output, args.dpi, frames)
    write_captions(output)

    metrics = []
    for panel, frame in frames.items():
        for method in PUBLIC_METHODS:
            row = frame.loc[method]
            for metric, value in row.items():
                metrics.append({"panel": panel, "method": method, "metric": metric, "value": value})
    pd.DataFrame(metrics).to_csv(output / "public_method_metrics_long.csv", index=False)

    manifest = {
        "status": "DREAMS_VOLAR_PUBLICATION_FIGURE_SUITE_COMPLETE",
        "public_method_name": METHOD_NAME,
        "expanded_name": EXPANDED_NAME,
        "public_version": "1.0",
        "frozen_implementation_alias": IMPLEMENTATION,
        "figures": generated,
        "same_axis_domain": {
            "identity_queries": int(frames["identity_disjoint"].iloc[0]["N_Q"]),
            "formula_queries": int(frames["formula_disjoint"].iloc[0]["N_Q"]),
            "reader_facing_public_methods": len(PUBLIC_METHODS),
            "all_scoring_streams_in_background_curves": len(frames["identity_disjoint"]),
        },
        "evidence_role": "consumed GNPS development and consumed Enveda meta-development",
        "external_confirmation": "pending; no CDC/HUMAN model performance is shown",
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        """# DreaMS-VOLAR reader-facing publication figures

This suite replaces the internal Full23 engineering plots. PNG files are
300-dpi previews; PDF and SVG are vector masters. Public methods occupy the
main benchmark figure, while internal modules appear only in the evidence
ablation. `Full23 v1.3.2` is retained solely as the frozen implementation alias.

GNPS and Enveda are consumed development resources. No CDC/HUMAN score is shown.
""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
