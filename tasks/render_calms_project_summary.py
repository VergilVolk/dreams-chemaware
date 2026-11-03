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
ARCHITECTURE_ARTWORK = ROOT / "assets/calms_architecture_ai_v5.png"
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
DISPLAY["official_dreams"] = "DreaMS"
DISPLAY["weighted_spectral_entropy"] = "WSE"

COLORS = dict(source.COLORS)
COLORS[METHOD_ID] = "#1557B0"
COLORS["specembedding_official"] = "#86B6D1"

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
            "axes.edgecolor": "#A9CDE2",
            "axes.labelcolor": "#214F73",
            "xtick.color": "#365B73",
            "ytick.color": "#365B73",
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
    ax.plot([bus_left_x, bus_left_x], [lane_centers[1], lane_centers[0]], color="#374151", lw=1.8)
    ax.plot([bus_right_x, bus_right_x], [lane_centers[-1], lane_centers[0]], color="#374151", lw=1.8)
    arrow(ax, (0.187, 0.505), (bus_left_x, 0.505))

    lane_specs = (
        (
            lane_y[0],
            "Condition-robust spectral representation",
            "Multi-condition spectra and targeted peak perturbations\nlearn stable molecular identity",
            "#E8F2FF",
            "#1769AA",
        ),
        (
            lane_y[1],
            "Fragmentation-structure consistency",
            "Diagnostic fragments, neutral losses and candidate-specific\nchemical evidence separate related structures",
            "#EAF7E8",
            "#388E3C",
        ),
        (
            lane_y[2],
            "Sample-level metabolic context",
            "Optional real-sample evidence refines ambiguous annotations;\nnot used for the GNPS results below",
            "#F4E8FA",
            "#7B2CBF",
        ),
    )
    for index, (yy, title, subtitle, face, edge) in enumerate(lane_specs):
        rounded_box(
            ax,
            (lane_x, yy),
            lane_w,
            lane_h,
            face,
            edge,
            radius=0.018,
            linestyle="--" if index == 2 else "-",
        )
        ax.text(lane_x + 0.020, yy + 0.158, title, ha="left", va="center", fontsize=11.4, fontweight="bold", color=edge)
        ax.text(lane_x + 0.020, yy + 0.105, subtitle, ha="left", va="center", fontsize=8.4, color="#334155", linespacing=1.25)
        center = lane_centers[index]
        if index < 2:
            arrow(ax, (bus_left_x, center), (lane_x, center))
        arrow(
            ax,
            (lane_x + lane_w, center),
            (bus_right_x, center),
            color=edge if index == 2 else "#374151",
            linestyle="--" if index == 2 else "-",
        )

    draw_vector_glyph(ax, lane_x + 0.020, lane_y[0] + 0.026, 0.150, "#1769AA")
    draw_spectrum(ax, (lane_x + 0.020, lane_y[1] + 0.050), (0.105, 0.047), "#111827")
    draw_spectrum(ax, (lane_x + 0.158, lane_y[1] + 0.050), (0.105, 0.047), "#388E3C", flip=True)
    ax.text(lane_x + 0.143, lane_y[1] + 0.049, "<->", ha="center", va="center", fontsize=11, color="#334155")
    network_nodes = (
        (lane_x + 0.042, lane_y[2] + 0.044),
        (lane_x + 0.102, lane_y[2] + 0.072),
        (lane_x + 0.164, lane_y[2] + 0.036),
        (lane_x + 0.225, lane_y[2] + 0.070),
    )
    for left, right in zip(network_nodes[:-1], network_nodes[1:]):
        ax.plot([left[0], right[0]], [left[1], right[1]], color="#A855F7", lw=1.3, alpha=0.75)
    for index, (xx, yy) in enumerate(network_nodes):
        ax.add_patch(Circle((xx, yy), 0.012 if index in (1, 2) else 0.009, facecolor="#F3E8FF", edgecolor="#7B2CBF", lw=1.3))
    ax.text(lane_x + 0.278, lane_y[2] + 0.040, "when\navailable", ha="right", va="center", fontsize=7.2, color="#6B21A8")
    rounded_box(ax, (0.030, 0.012), 0.145, 0.055, "#FAF5FF", "#7B2CBF", radius=0.012, linewidth=1.2, linestyle="--")
    ax.text(0.1025, 0.039, "Real-sample profiles (optional)", ha="center", va="center", fontsize=7.7, color="#6B21A8", fontweight="bold")
    arrow(ax, (0.175, 0.040), (lane_x, lane_centers[2]), color="#7B2CBF", linestyle="--", linewidth=1.2)

    # CALMS combines the three scientific evidence streams. Internal
    # implementation names stay out of the reader-facing architecture.
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
    ax.text(0.724, 0.714, "CALMS", ha="center", va="center", fontsize=14.0, fontweight="bold", color="#713F12")
    rounded_box(ax, (0.653, 0.500), 0.142, 0.126, "#E8F2FF", "#1769AA", radius=0.014, linewidth=1.3)
    ax.text(0.724, 0.563, "Candidate-specific\nevidence integration", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#0F3D75", linespacing=1.2)
    ax.add_patch(Circle((0.724, 0.452), 0.023, facecolor="white", edgecolor="#374151", lw=1.3))
    ax.text(0.724, 0.452, "↓", ha="center", va="center", fontsize=14, fontweight="bold")
    rounded_box(ax, (0.653, 0.304), 0.142, 0.104, "#FFFDF7", "#D97706", radius=0.014, linewidth=1.3)
    ax.text(0.724, 0.356, "Annotation confidence\nand evidence record", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#713F12", linespacing=1.2)
    ax.text(0.724, 0.270, "structure-aware, context-aware", ha="center", va="center", fontsize=7.7, color="#713F12")
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


def draw_ai_architecture(ax: plt.Axes) -> None:
    """Render the AI-drawn, text-free architecture with deterministic labels.

    Keeping all typography in Matplotlib prevents generated pseudo-text and
    guarantees stable alignment in PNG, PDF and SVG exports.
    """
    if not ARCHITECTURE_ARTWORK.is_file():
        raise FileNotFoundError(ARCHITECTURE_ARTWORK)
    artwork = plt.imread(ARCHITECTURE_ARTWORK)
    # Image-generation exports can contain a few black boundary rows.  Remove
    # only those boundary rows before scaling so they cannot become rules in
    # the publication figure.
    row_brightness = artwork[..., :3].mean(axis=(1, 2))
    content_rows = np.flatnonzero(row_brightness > 0.95)
    if content_rows.size:
        artwork = artwork[content_rows[0] : content_rows[-1] + 1]
    ax.imshow(artwork, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    ax.text(-0.010, 1.015, "a", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    labels = (
        (0.120, 0.858, "LC-MS/MS spectrum\nand candidate structures", "#214F73", 12.6),
        (0.418, 0.906, "Condition-robust\nspectral representation", "#214F73", 12.2),
        (0.418, 0.585, "Fragmentation chemistry-guided\ndiscrimination", "#214F73", 11.8),
        (0.418, 0.258, "Biochemical relationships in\nbiological samples", "#214F73", 11.8),
        (0.681, 0.525, "CALMS", "#214F73", 19.0),
        (0.681, 0.420, "metabolite\nannotation", "#214F73", 12.6),
        (0.875, 0.858, "Annotated candidate\nstructures", "#214F73", 12.6),
    )
    for xx, yy, label, color, fontsize in labels:
        ax.text(
            xx,
            yy,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            linespacing=1.12,
        )


def draw_ladder(ax: plt.Axes, frame) -> None:
    data = frame.loc[list(PUBLIC_METHODS)].sort_values("recall@1", ascending=True)
    yy = np.arange(len(data))
    xmin = 74.2
    for y, (method, row) in enumerate(data.iterrows()):
        value = 100.0 * float(row["recall@1"])
        ax.hlines(y, xmin, value, color="#CFE3EF", lw=1.8, zorder=1)
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
            color="#0F3D75" if method == METHOD_ID else "#365B73",
        )
    ax.set_yticks(yy, [DISPLAY[m] for m in data.index], fontsize=9.0)
    ax.set_xlim(xmin, 91.2)
    ax.set_xlabel("Recall@1 (%)")
    ax.set_title("Top-1 metabolite annotation accuracy\nGNPS identity-disjoint (n=10,983)", pad=10, fontweight="bold")
    ax.grid(axis="x", color="#DCEAF3", alpha=0.85, linewidth=0.8)
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
    ax.plot([0, 1], [0, 1], color="#A9CDE2", lw=0.9, ls=":", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Candidate-level discrimination\nGNPS identity-disjoint", pad=10, fontweight="bold")
    ax.grid(color="#DCEAF3", alpha=0.85, linewidth=0.8)
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

    fig = plt.figure(figsize=(15.2, 10.2))
    architecture = fig.add_axes((0.035, 0.400, 0.930, 0.515))
    ladder = fig.add_axes((0.175, 0.085, 0.285, 0.255))
    roc = fig.add_axes((0.565, 0.085, 0.285, 0.255))

    draw_ai_architecture(architecture)

    draw_ladder(ladder, frame)
    draw_roc(roc, frame)

    fig.suptitle(
        f"{METHOD_NAME}: {EXPANDED_NAME}",
        x=0.500,
        y=0.963,
        fontsize=21,
        fontweight="bold",
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
        "architecture_asset": str(ARCHITECTURE_ARTWORK.relative_to(ROOT)),
        "architecture_prompt_record": "deliverables/calms_project_summary_20261103/ARTWORK_PROMPT.md",
        "architecture_scope": "three reader-facing scientific modules; AI-drawn text-free artwork with deterministic code typography",
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
        "**a,** CALMS combines condition-robust MS/MS representations with fragmentation-structure consistency evidence, "
        "then optionally refines ambiguous annotations using real-sample metabolic context when that context is available. "
        "The sample-context module was not used for the GNPS benchmark. **b,** Recall@1 on the common-domain GNPS "
        f"identity-disjoint development panel ({int(frame.iloc[0]['N_Q']):,} queries). CALMS reaches "
        f"{100.0 * calms_r1:.2f}% and improves over DreaMS by "
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
