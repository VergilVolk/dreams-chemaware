#!/usr/bin/env python3
"""Render the frozen CALMS mechanism/selective-ranking manuscript figure.

Panel A localizes CALMS-versus-WSE Top-1 decision exchanges by quartiles of
the observable WSE Top1--Top2 margin.  Panel B renders the completed fixed-
coverage selective closed-set analysis with formula-cluster intervals.

No model is loaded, refit, selected, or rescored by this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PANELS = ("identity_disjoint", "formula_disjoint")
PANEL_TITLES = {
    "identity_disjoint": "Identity-disjoint",
    "formula_disjoint": "Formula-disjoint",
}
EXPECTED_QUERIES = {"identity_disjoint": 10_995, "formula_disjoint": 5_261}
CALMS_COLOR = "#0072B2"
WSE_COLOR = "#D55E00"
GRID_COLOR = "#D9DEE3"
TEXT_COLOR = "#20262E"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calms-query-dir",
        type=Path,
        default=Path("deliverables/GLM_gnps_article_ladder/run16_full23_v132"),
    )
    parser.add_argument(
        "--wse-query-dir",
        type=Path,
        default=Path(
            "data/validation/GLM_gnps_article_benchmark_s2v26/run15/evaluation"
        ),
    )
    parser.add_argument(
        "--selective-dir",
        type=Path,
        default=Path(
            "data/validation/calms_v132_closed_set_selective_ranking_20261105"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("deliverables/calms_v1_mechanism_selective_20261105"),
    )
    return parser.parse_args()


def read_query_tables(args: argparse.Namespace, panel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    calms_path = (
        args.calms_query_dir / f"queries_{panel}_full23_v132.csv.gz"
    )
    wse_path = (
        args.wse_query_dir
        / f"queries_{panel}_weighted_spectral_entropy.csv.gz"
    )
    for path in (calms_path, wse_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    columns = (
        "query_index",
        "query_row",
        "query_formula",
        "rank",
        "top1_top2_gap",
    )
    calms = pd.read_csv(calms_path, usecols=columns).sort_values(
        "query_index", kind="stable"
    ).reset_index(drop=True)
    wse = pd.read_csv(wse_path, usecols=columns).sort_values(
        "query_index", kind="stable"
    ).reset_index(drop=True)
    if len(calms) != EXPECTED_QUERIES[panel] or len(wse) != EXPECTED_QUERIES[panel]:
        raise RuntimeError(f"{panel}: expected full frozen GNPS query axis")
    for table, method in ((calms, "CALMS"), (wse, "WSE")):
        if table["query_index"].duplicated().any() or table["query_row"].duplicated().any():
            raise RuntimeError(f"{panel}/{method}: duplicate query axis")
        if not np.all(np.isfinite(table[["rank", "top1_top2_gap"]].to_numpy(float))):
            raise RuntimeError(f"{panel}/{method}: non-finite ranking values")
        if np.any(table["rank"].to_numpy(float) < 1) or np.any(
            table["top1_top2_gap"].to_numpy(float) < 0
        ):
            raise RuntimeError(f"{panel}/{method}: invalid ranking values")
    for column in ("query_index", "query_row", "query_formula"):
        if not np.array_equal(calms[column].to_numpy(), wse[column].to_numpy()):
            raise RuntimeError(f"{panel}: CALMS/WSE {column} axis drifted")
    return calms, wse


def exchange_table(calms: pd.DataFrame, wse: pd.DataFrame, panel: str) -> pd.DataFrame:
    quartile = pd.qcut(
        wse["top1_top2_gap"],
        q=4,
        labels=("Q1", "Q2", "Q3", "Q4"),
        duplicates="raise",
    )
    calms_correct = calms["rank"].eq(1).to_numpy()
    wse_correct = wse["rank"].eq(1).to_numpy()
    records = []
    for label in ("Q1", "Q2", "Q3", "Q4"):
        selected = quartile.eq(label).to_numpy()
        n = int(np.sum(selected))
        corrected = int(np.sum(selected & calms_correct & ~wse_correct))
        introduced = int(np.sum(selected & ~calms_correct & wse_correct))
        margins = wse.loc[selected, "top1_top2_gap"].to_numpy(float)
        records.append({
            "panel": panel,
            "quartile": label,
            "queries": n,
            "corrected": corrected,
            "introduced": introduced,
            "corrected_percent": 100.0 * corrected / n,
            "introduced_percent": 100.0 * introduced / n,
            "net_correct": corrected - introduced,
            "margin_min": float(np.min(margins)),
            "margin_max": float(np.max(margins)),
        })
    result = pd.DataFrame.from_records(records)
    observed_net = int(result["net_correct"].sum())
    expected_net = int(np.sum(calms_correct) - np.sum(wse_correct))
    if observed_net != expected_net:
        raise RuntimeError(f"{panel}: decision exchanges do not reconcile")
    return result


def read_selective_points(args: argparse.Namespace) -> pd.DataFrame:
    path = args.selective_dir / "selective_operating_points.csv"
    report = args.selective_dir / "report.json"
    for required in (path, report):
        if not required.is_file():
            raise FileNotFoundError(required)
    table = pd.read_csv(path)
    required_columns = {
        "panel",
        "method",
        "target_coverage",
        "selective_recall_at_1",
        "formula_cluster_ci_low",
        "formula_cluster_ci_high",
    }
    if not required_columns.issubset(table.columns):
        raise RuntimeError("selective operating-point table is incomplete")
    table = table.loc[
        table["panel"].isin(PANELS)
        & table["method"].isin(("calms_v132", "weighted_spectral_entropy"))
    ].copy()
    expected_rows = len(PANELS) * 2 * 7
    if len(table) != expected_rows or table.duplicated(
        ["panel", "method", "target_coverage"]
    ).any():
        raise RuntimeError("selective operating-point axis drifted")
    for column in (
        "target_coverage",
        "selective_recall_at_1",
        "formula_cluster_ci_low",
        "formula_cluster_ci_high",
    ):
        if not np.all(np.isfinite(table[column].to_numpy(float))):
            raise RuntimeError(f"selective table contains non-finite {column}")
    return table


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8B949E")
    axis.tick_params(colors=TEXT_COLOR, labelsize=9)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def plot_exchanges(axis: plt.Axes, table: pd.DataFrame, title: str) -> None:
    x = np.arange(4, dtype=float)
    width = 0.32
    corrected = table["corrected_percent"].to_numpy(float)
    introduced = -table["introduced_percent"].to_numpy(float)
    corrected_bars = axis.bar(
        x - width / 2,
        corrected,
        width,
        color=CALMS_COLOR,
        label="WSE error corrected",
        zorder=3,
    )
    introduced_bars = axis.bar(
        x + width / 2,
        introduced,
        width,
        color=WSE_COLOR,
        label="WSE correct introduced as error",
        zorder=3,
    )
    axis.axhline(0, color="#59636E", linewidth=0.9)
    for bar, count in zip(corrected_bars, table["corrected"]):
        if count:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=CALMS_COLOR,
                fontweight="bold",
            )
    for bar, count in zip(introduced_bars, table["introduced"]):
        if count:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.25,
                f"{int(count)}",
                ha="center",
                va="top",
                fontsize=8.5,
                color=WSE_COLOR,
                fontweight="bold",
            )
    q1 = table.iloc[0]
    axis.text(
        0,
        max(corrected[0] + 1.35, 9.6),
        f"net +{int(q1['net_correct'])} correct",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TEXT_COLOR,
    )
    axis.text(
        2,
        0.55,
        "No Top-1 exchanges in Q2--Q4",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#59636E",
    )
    axis.set_xticks(x, ("Q1\nlowest", "Q2", "Q3", "Q4\nhighest"))
    axis.set_ylim(-5.4, 11.5)
    axis.set_yticks((-4, 0, 4, 8))
    axis.set_title(f"{title} (n={int(table['queries'].sum()):,})", fontsize=11, pad=9)
    axis.set_xlabel("WSE Top1--Top2 margin quartile", fontsize=9.5)
    style_axis(axis)


def plot_selective(axis: plt.Axes, table: pd.DataFrame, title: str) -> None:
    for method, label, color in (
        ("calms_v132", "CALMS", CALMS_COLOR),
        ("weighted_spectral_entropy", "WSE", WSE_COLOR),
    ):
        block = table.loc[table["method"].eq(method)].sort_values("target_coverage")
        x = 100.0 * block["target_coverage"].to_numpy(float)
        y = 100.0 * block["selective_recall_at_1"].to_numpy(float)
        low = 100.0 * block["formula_cluster_ci_low"].to_numpy(float)
        high = 100.0 * block["formula_cluster_ci_high"].to_numpy(float)
        axis.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
        axis.plot(
            x,
            y,
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=label,
            zorder=4,
        )
    axis.set_xlim(48, 101)
    axis.set_ylim(85.5, 100.5)
    axis.set_xticks((50, 60, 70, 80, 90, 100))
    axis.set_yticks((86, 90, 94, 98, 100))
    axis.set_title(f"{title} (n={EXPECTED_QUERIES[table['panel'].iloc[0]]:,})", fontsize=11, pad=9)
    axis.set_xlabel("Retained query coverage (%)", fontsize=9.5)
    style_axis(axis)


def main() -> None:
    args = arguments()
    exchange = {}
    for panel in PANELS:
        calms, wse = read_query_tables(args, panel)
        exchange[panel] = exchange_table(calms, wse, panel)
    selective = read_selective_points(args)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), constrained_layout=False)
    figure.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.17, wspace=0.20, hspace=0.42)

    for column, panel in enumerate(PANELS):
        plot_exchanges(axes[0, column], exchange[panel], PANEL_TITLES[panel])
        plot_selective(
            axes[1, column],
            selective.loc[selective["panel"].eq(panel)],
            PANEL_TITLES[panel],
        )
    axes[0, 0].set_ylabel("Decision exchanges (% of quartile)", fontsize=9.5)
    axes[1, 0].set_ylabel("Selective Recall@1 (%)", fontsize=9.5)
    axes[0, 0].text(-0.17, 1.13, "A", transform=axes[0, 0].transAxes, fontsize=17, fontweight="bold")
    axes[1, 0].text(-0.17, 1.13, "B", transform=axes[1, 0].transAxes, fontsize=17, fontweight="bold")

    exchange_legend = [
        plt.Rectangle((0, 0), 1, 1, color=CALMS_COLOR, label="WSE error corrected by CALMS"),
        plt.Rectangle((0, 0), 1, 1, color=WSE_COLOR, label="WSE correct introduced as error"),
    ]
    axes[0, 0].legend(
        handles=exchange_legend,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=False,
        fontsize=8.2,
    )
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color=CALMS_COLOR, marker="o", label="CALMS"),
            Line2D([0], [0], color=WSE_COLOR, marker="o", label="WSE"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(
        "CALMS interventions localize to low-margin WSE decisions while preserving selective accuracy",
        fontsize=13.2,
        fontweight="bold",
        y=0.965,
    )
    caption = (
        "A, Top-1 exchanges from WSE to frozen CALMS v1.0 (internal Full23 v1.3.2), stratified "
        "by WSE margin quartile; "
        "bar labels are query counts and bar heights are percentages within quartile. B, closed-set "
        "selective Recall@1 (shading: 95% formula-cluster bootstrap intervals). GNPS Gold/Silver is "
        "consumed development evidence and every query contains its true candidate; these curves do "
        "not evaluate no-match/open-set rejection, false-match FDR, or a deployable acceptance threshold."
    )
    figure.text(
        0.09,
        0.055,
        caption,
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#3F4852",
        wrap=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "calms_mechanism_selective"
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    (args.output_dir / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    pd.concat(exchange.values(), ignore_index=True).to_csv(
        args.output_dir / "panel_a_exchange_values.tsv", sep="\t", index=False
    )
    print(f"CALMS_MECHANISM_SELECTIVE_FIGURE_COMPLETE: {args.output_dir}")


if __name__ == "__main__":
    main()
