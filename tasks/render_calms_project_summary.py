#!/usr/bin/env python
"""Render the CALMS project summary figure from frozen GNPS ledgers.

The figure has one complete method diagram and two result panels on the same
identity-disjoint candidate axis.  It deliberately does not mix identity- and
formula-disjoint results in one visual comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

import render_dreams_volar_publication_suite as source  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
METHOD_ID = "full23_v132"
METHOD_NAME = "CALMS"
EXPANDED_NAME = "Context-Aware Learning for Metabolite Annotation from Mass Spectra"
PANEL = "identity_disjoint"

PUBLIC_METHODS = (
    METHOD_ID,
    "weighted_spectral_entropy",
    "entropy_raw_public",
    "denoising_search_public",
    "spec2vec_gnps_public",
    "official_dreams",
    "msbert_official",
    "ms2deepscore_2x_public",
    "specembedding_official",
)

DISPLAY = dict(source.DISPLAY)
DISPLAY[METHOD_ID] = "CALMS (ours)"

COLORS = dict(source.COLORS)
COLORS[METHOD_ID] = "#1557B0"

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
        default=ROOT / "deliverables/calms_project_summary_20261103",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
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


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str,
    radius: float = 0.018,
    linewidth: float = 1.6,
    linestyle: str = "-",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#374151",
    linestyle: str = "-",
    linewidth: float = 1.6,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def draw_spectrum(
    ax: plt.Axes,
    origin: tuple[float, float],
    size: tuple[float, float],
    color: str,
    flip: bool = False,
) -> None:
    x0, y0 = origin
    width, height = size
    peaks = np.asarray(
        [
            (0.06, 0.20),
            (0.16, 0.48),
            (0.25, 0.31),
            (0.37, 0.78),
            (0.49, 0.27),
            (0.58, 0.93),
            (0.69, 0.44),
            (0.78, 0.66),
            (0.91, 0.35),
        ],
        dtype=float,
    )
    ax.plot([x0, x0 + width], [y0, y0], color="#475569", lw=0.9)
    direction = -1.0 if flip else 1.0
    for px, py in peaks:
        ax.plot(
            [x0 + px * width, x0 + px * width],
            [y0, y0 + direction * py * height],
            color=color,
            lw=1.8,
        )


def draw_molecule(ax: plt.Axes, center: tuple[float, float], scale: float, color: str) -> None:
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    xy = np.c_[center[0] + scale * np.cos(angles), center[1] + scale * np.sin(angles)]
    ax.add_patch(Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.7))
    ax.plot(
        [xy[1, 0], xy[1, 0] + 0.75 * scale],
        [xy[1, 1], xy[1, 1] + 0.55 * scale],
        color=color,
        lw=1.5,
    )


def draw_vector_glyph(ax: plt.Axes, x: float, y: float, width: float, color: str) -> None:
    cells = 7
    cell_w = width / cells
    for i in range(cells):
        shade = 0.18 + 0.07 * ((i * 3) % 5)
        ax.add_patch(
            FancyBboxPatch(
                (x + i * cell_w, y),
                cell_w * 0.88,
                0.050,
                boxstyle="round,pad=0.002",
                facecolor=color,
                alpha=shade,
                edgecolor=color,
                linewidth=0.8,
            )
        )


def draw_architecture(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.018, 1.01, "a", fontsize=16, fontweight="bold", va="top")

    # Input: query spectrum and mass-compatible structures.
    rounded_box(ax, (0.012, 0.10), 0.175, 0.80, "#F8FAFC", "#64748B", radius=0.020)
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.80),
            0.175,
            0.10,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            facecolor="#E5E7EB",
            edgecolor="none",
        )
    )
    ax.text(0.0995, 0.847, "Query MS/MS spectrum", ha="center", va="center", fontsize=11.8, fontweight="bold")
    draw_spectrum(ax, (0.035, 0.575), (0.125, 0.150), "#111827")
    ax.text(0.0995, 0.542, r"$m/z$", ha="center", va="center", fontsize=9.5, style="italic")
    ax.plot([0.026, 0.173], [0.505, 0.505], color="#CBD5E1", lw=1.0)
    ax.text(0.0995, 0.468, "Mass-compatible candidates", ha="center", va="center", fontsize=10.5, fontweight="bold")
    for rank, yy, color in (("C1", 0.378, "#1557B0"), ("C2", 0.282, "#475569"), ("C3", 0.186, "#475569")):
        draw_molecule(ax, (0.064, yy), 0.029, color)
        ax.text(0.132, yy, rank, ha="center", va="center", fontsize=10.5, color=color, fontweight="bold")

    # Evidence lanes, using buses to avoid crossing arrows.
    bus_left_x = 0.222
    bus_right_x = 0.594
    lane_x = 0.258
    lane_w = 0.300
    lane_h = 0.202
    lane_y = (0.690, 0.405, 0.120)
    lane_centers = tuple(y + lane_h / 2 for y in lane_y)
    ax.plot([bus_left_x, bus_left_x], [lane_centers[-1], lane_centers[0]], color="#374151", lw=1.8)
    ax.plot([bus_right_x, bus_right_x], [lane_centers[-1], lane_centers[0]], color="#374151", lw=1.8)
    arrow(ax, (0.187, 0.505), (bus_left_x, 0.505))

    lane_specs = (
        (
            lane_y[0],
            "Condition-robust spectral representations",
            "Self-supervised embeddings preserve molecular identity\nacross acquisition conditions",
            "#E8F2FF",
            "#1769AA",
        ),
        (
            lane_y[1],
            "Fragmentation evidence",
            "Global spectral matching and local fragment /\nneutral-loss patterns",
            "#EAF7E8",
            "#388E3C",
        ),
        (
            lane_y[2],
            "Candidate chemistry",
            "Candidate-specific chemical support relative to\nmatched chemical nulls",
            "#F4E8FA",
            "#7B2CBF",
        ),
    )
    for index, (yy, title, subtitle, face, edge) in enumerate(lane_specs):
        rounded_box(ax, (lane_x, yy), lane_w, lane_h, face, edge, radius=0.018)
        ax.text(lane_x + 0.020, yy + 0.158, title, ha="left", va="center", fontsize=11.4, fontweight="bold", color=edge)
        ax.text(lane_x + 0.020, yy + 0.105, subtitle, ha="left", va="center", fontsize=8.4, color="#334155", linespacing=1.25)
        center = lane_centers[index]
        arrow(ax, (bus_left_x, center), (lane_x, center))
        arrow(ax, (lane_x + lane_w, center), (bus_right_x, center))

    draw_vector_glyph(ax, lane_x + 0.020, lane_y[0] + 0.026, 0.150, "#1769AA")
    draw_spectrum(ax, (lane_x + 0.020, lane_y[1] + 0.050), (0.105, 0.047), "#111827")
    draw_spectrum(ax, (lane_x + 0.158, lane_y[1] + 0.050), (0.105, 0.047), "#388E3C", flip=True)
    ax.text(lane_x + 0.143, lane_y[1] + 0.049, "<->", ha="center", va="center", fontsize=11, color="#334155")
    draw_molecule(ax, (lane_x + 0.050, lane_y[2] + 0.045), 0.025, "#7B2CBF")
    draw_vector_glyph(ax, lane_x + 0.095, lane_y[2] + 0.020, 0.120, "#7B2CBF")

    # Optional real-sample context is explicit and separated from the GNPS benchmark.
    rounded_box(
        ax,
        (lane_x + 0.020, 0.018),
        0.260,
        0.055,
        "#FFFDF5",
        "#B7791F",
        radius=0.014,
        linewidth=1.2,
        linestyle="--",
    )
    ax.text(
        lane_x + 0.150,
        0.045,
        "Real-sample context when available (not used in GNPS benchmark)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#8A5A00",
    )
    arrow(ax, (lane_x + 0.150, 0.073), (lane_x + 0.150, lane_y[2]), color="#B7791F", linestyle="--", linewidth=1.1)

    # CALMS annotation model.
    rounded_box(ax, (0.635, 0.235), 0.178, 0.530, "#FFF6DE", "#D97706", radius=0.022, linewidth=1.9)
    ax.add_patch(
        FancyBboxPatch(
            (0.635, 0.665),
            0.178,
            0.100,
            boxstyle="round,pad=0.010,rounding_size=0.022",
            facecolor="#FFD98A",
            edgecolor="none",
        )
    )
    ax.text(0.724, 0.714, "CALMS annotation model", ha="center", va="center", fontsize=12.2, fontweight="bold", color="#713F12")
    rounded_box(ax, (0.653, 0.535), 0.142, 0.092, "#E8F2FF", "#1769AA", radius=0.014, linewidth=1.3)
    ax.text(0.724, 0.581, "Integrated evidence score", ha="center", va="center", fontsize=9.4, fontweight="bold", color="#0F3D75")
    ax.add_patch(Circle((0.724, 0.482), 0.025, facecolor="white", edgecolor="#374151", lw=1.3))
    ax.text(0.724, 0.482, "+", ha="center", va="center", fontsize=16, fontweight="bold")
    rounded_box(ax, (0.653, 0.337), 0.142, 0.092, "#E8F2FF", "#1769AA", radius=0.014, linewidth=1.3)
    ax.text(0.724, 0.383, "Full-embedding correction", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#0F3D75")
    ax.text(0.724, 0.286, r"$S(q,c)=S_{evidence}(q,c)+0.9\,r_{embed}(q,c)$", ha="center", va="center", fontsize=8.8, color="#713F12")
    arrow(ax, (bus_right_x, 0.505), (0.635, 0.505))

    # Output: annotation rather than defining the method as a ranker.
    rounded_box(ax, (0.855, 0.10), 0.132, 0.80, "#F8FAFC", "#64748B", radius=0.020)
    ax.add_patch(
        FancyBboxPatch(
            (0.855, 0.80),
            0.132,
            0.10,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            facecolor="#E5E7EB",
            edgecolor="none",
        )
    )
    ax.text(0.921, 0.847, "Metabolite annotation", ha="center", va="center", fontsize=11.5, fontweight="bold")
    rounded_box(ax, (0.872, 0.575), 0.098, 0.150, "#E8F2FF", "#1557B0", radius=0.013, linewidth=1.4)
    draw_molecule(ax, (0.910, 0.650), 0.032, "#1557B0")
    ax.text(0.958, 0.650, "0.93", ha="right", va="center", fontsize=9.8, fontweight="bold", color="#1557B0")
    ax.text(0.921, 0.535, "supported structure", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#0F3D75")
    ax.text(0.921, 0.486, "Alternative hypotheses", ha="center", va="center", fontsize=8.3, color="#475569")
    for yy, score in ((0.405, "0.76"), (0.305, "0.61")):
        rounded_box(ax, (0.878, yy - 0.040), 0.086, 0.080, "white", "#94A3B8", radius=0.010, linewidth=1.0)
        draw_molecule(ax, (0.906, yy), 0.020, "#64748B")
        ax.text(0.955, yy, score, ha="right", va="center", fontsize=8.2, color="#64748B")
    ax.text(0.921, 0.188, "auditable evidence record", ha="center", va="center", fontsize=7.9, color="#475569")
    arrow(ax, (0.813, 0.505), (0.855, 0.505))


def draw_ladder(ax: plt.Axes, frame) -> None:
    data = frame.loc[list(PUBLIC_METHODS)].sort_values("recall@1", ascending=True)
    yy = np.arange(len(data))
    xmin = 74.2
    for y, (method, row) in enumerate(data.iterrows()):
        value = 100.0 * float(row["recall@1"])
        ax.hlines(y, xmin, value, color="#D7DEE8", lw=1.8, zorder=1)
        ax.scatter(
            value,
            y,
            s=125 if method == METHOD_ID else 60,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.text(
            value + 0.18,
            y,
            f"{value:.2f}",
            va="center",
            fontsize=9.0,
            fontweight="bold" if method == METHOD_ID else "normal",
            color="#0F3D75" if method == METHOD_ID else "#374151",
        )
    ax.set_yticks(yy, [DISPLAY[m] for m in data.index], fontsize=9.0)
    ax.set_xlim(xmin, 91.2)
    ax.set_xlabel("Recall@1 (%)")
    ax.set_title("Top-1 metabolite annotation accuracy\nGNPS identity-disjoint (n=10,983)", pad=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.text(-0.10, 1.08, "b", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    calms = float(frame.loc[METHOD_ID, "recall@1"])
    dreams = float(frame.loc["official_dreams", "recall@1"])
    delta = 100.0 * (calms - dreams)
    error_reduction = 100.0 * (calms - dreams) / (1.0 - dreams)
    ax.text(
        0.48,
        0.91,
        f"+{delta:.2f} pp vs DreaMS\n{error_reduction:.1f}% fewer Top-1 errors",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        color="#0F3D75",
        fontweight="bold",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E8F2FF", "edgecolor": "#A8C7EA", "linewidth": 1.0},
    )


def draw_roc(ax: plt.Axes, frame) -> None:
    for method in HIGHLIGHT_CURVES:
        roc = source.curve(PANEL, method, "roc")
        label = f"{DISPLAY[method]} ({float(frame.loc[method, 'micro_candidate_auroc']):.3f})"
        ax.plot(
            roc["false_positive_rate"],
            roc["true_positive_rate"],
            color=COLORS[method],
            lw=3.2 if method == METHOD_ID else 1.65,
            ls="--" if method == "ms2deepscore_2x_public" else "-",
            alpha=1.0 if method == METHOD_ID else 0.90,
            label=label,
            zorder=4 if method == METHOD_ID else 3,
        )
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=0.9, ls=":", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Candidate-level discrimination\nGNPS identity-disjoint", pad=10, fontweight="bold")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8.0, loc="lower right", handlelength=2.6)
    ax.text(-0.10, 1.08, "c", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    calms_auc = float(frame.loc[METHOD_ID, "micro_candidate_auroc"])
    dreams_auc = float(frame.loc["official_dreams", "micro_candidate_auroc"])
    ax.text(
        0.52,
        0.66,
        f"CALMS AUROC\n{calms_auc:.3f}\n+{calms_auc - dreams_auc:.3f} vs DreaMS",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.0,
        color="#0F3D75",
        fontweight="bold",
        linespacing=1.28,
        bbox={"boxstyle": "round,pad=0.48", "facecolor": "#E8F2FF", "edgecolor": "#A8C7EA", "linewidth": 1.0},
    )


def render(output: Path, dpi: int) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    frames = source.common_frames()
    frame = frames[PANEL]

    fig = plt.figure(figsize=(17.0, 12.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.18, 0.82),
        left=0.055,
        right=0.975,
        bottom=0.075,
        top=0.895,
        hspace=0.24,
        wspace=0.27,
    )
    architecture = fig.add_subplot(grid[0, :])
    ladder = fig.add_subplot(grid[1, 0])
    roc = fig.add_subplot(grid[1, 1])

    draw_architecture(architecture)
    draw_ladder(ladder, frame)
    draw_roc(roc, frame)

    fig.suptitle(
        f"{METHOD_NAME}: {EXPANDED_NAME}",
        x=0.515,
        y=0.972,
        fontsize=23,
        fontweight="bold",
    )
    fig.text(
        0.515,
        0.935,
        "Condition-robust spectral learning, fragmentation chemistry and candidate-specific evidence support metabolite annotation.",
        ha="center",
        fontsize=12.5,
        color="#334155",
    )
    fig.text(
        0.515,
        0.025,
        "GNPS common-domain identity-disjoint development panel; all methods share the same query and candidate axis. "
        "MS2DeepScore 2.x is dashed because its native objective is structural similarity rather than exact annotation.",
        ha="center",
        fontsize=8.8,
        color="#475569",
    )

    stem = output / "calms_project_summary"
    files: list[str] = []
    for suffix in ("png", "pdf", "svg"):
        target = stem.with_suffix(f".{suffix}")
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(target, bbox_inches="tight", **kwargs)
        files.append(target.name)
    plt.close(fig)

    calms_r1 = float(frame.loc[METHOD_ID, "recall@1"])
    dreams_r1 = float(frame.loc["official_dreams", "recall@1"])
    calms_auc = float(frame.loc[METHOD_ID, "micro_candidate_auroc"])
    dreams_auc = float(frame.loc["official_dreams", "micro_candidate_auroc"])
    manifest = {
        "status": "CALMS_PROJECT_SUMMARY_COMPLETE",
        "public_method_name": METHOD_NAME,
        "expanded_name": EXPANDED_NAME,
        "frozen_implementation_alias": "Full23 v1.3.2",
        "panel": PANEL,
        "dataset_role": "consumed development",
        "queries": int(frame.iloc[0]["N_Q"]),
        "calms": {
            "recall_at_1": calms_r1,
            "micro_candidate_auroc": calms_auc,
        },
        "relative_to_official_finetuned_dreams": {
            "recall_at_1_pp": 100.0 * (calms_r1 - dreams_r1),
            "top1_error_reduction_percent": 100.0 * (calms_r1 - dreams_r1) / (1.0 - dreams_r1),
            "micro_candidate_auroc_absolute": calms_auc - dreams_auc,
        },
        "files": files,
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "CAPTION.md").write_text(
        "# CALMS project summary figure\n\n"
        "**a,** CALMS combines condition-robust spectral representations, fragmentation evidence and candidate-specific "
        "chemistry before producing a metabolite annotation with an auditable evidence record. Real-sample context is an "
        "optional extension and was not used in the GNPS benchmark. **b,** Recall@1 on the common-domain GNPS "
        f"identity-disjoint development panel ({int(frame.iloc[0]['N_Q']):,} queries). CALMS reaches "
        f"{100.0 * calms_r1:.2f}% and improves over official fine-tuned DreaMS by "
        f"{100.0 * (calms_r1 - dreams_r1):.2f} percentage points. **c,** Candidate-level ROC curves on the same "
        f"query/candidate axis; CALMS achieves AUROC {calms_auc:.3f}. GNPS is a consumed development resource, not an "
        "unopened external confirmation set.\n",
        encoding="utf-8",
    )
    return files


def main() -> None:
    args = arguments()
    style()
    files = render(args.output, args.dpi)
    print(
        json.dumps(
            {
                "status": "CALMS_PROJECT_SUMMARY_COMPLETE",
                "output": str(args.output),
                "files": files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
