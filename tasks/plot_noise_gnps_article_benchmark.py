#!/usr/bin/env python
"""Render the Noise/P2b GNPS article figures from a completed evaluation.

Inputs are exactly the artifacts of ``evaluate_noise_gnps_article_benchmark.py``:
``report.json``, ``curves_<panel>.csv.gz`` and ``queries_<panel>_<method>.csv.gz``.
Outputs: a six-panel main figure (PDF+PNG), a formula-panel curve supplement,
and a method-by-panel summary CSV.  No numbers are recomputed; every plotted
value is copied from the evaluation artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PANELS = ("identity_disjoint", "formula_disjoint")
PANEL_LABEL = {"identity_disjoint": "Identity-disjoint", "formula_disjoint": "Formula-disjoint"}
RECALL_CUTOFFS = (1, 2, 3, 5, 10, 20)

STRATA = {
    "official_dreams": ("embedding", "Official DreaMS"),
    "noise_v1": ("embedding", "Noise V1 (ours)"),
    "cosine_greedy": ("classical", "Greedy cosine"),
    "modified_cosine": ("classical", "Modified cosine"),
    "weighted_spectral_entropy": ("classical", "Weighted spectral entropy"),
    "p2b_sqrt_cosine": ("channel", "P2b sqrt-cosine channel"),
    "p2b_unweighted_entropy": ("channel", "P2b entropy channel"),
    "neutral_loss_sqrt_cosine": ("channel", "Neutral-loss sqrt cosine"),
    "p2b_official_frozen": ("reranker", "Frozen P2b + official"),
    "p2b_noise_v1_frozen": ("reranker", "Frozen P2b + Noise V1"),
}
STRATUM_NAME = {
    "classical": "classical similarity (no training)",
    "channel": "single evidence channel (diagnostic)",
    "embedding": "shared-structure embedding",
    "reranker": "frozen candidate reranker (extra evidence)",
}
STRATUM_COLOR = {
    "classical": "#8c8c8c",
    "channel": "#c7a35a",
    "embedding": "#1f77b4",
    "reranker": "#d62728",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def recall_at_k(table: pd.DataFrame) -> dict[int, float]:
    rank = table["rank"].to_numpy(np.int64)
    return {k: float(np.mean(rank <= k)) for k in RECALL_CUTOFFS}


def method_order(report: dict, panel: str) -> list[str]:
    return [
        row["method"]
        for row in report["panels"][panel]["rankings"]["recall_at_1"]
    ]


def draw_recall_bars(axis, report: dict, methods: list[str]) -> None:
    width = 0.38
    positions = np.arange(len(methods))
    for offset, panel, hatch in ((-width / 2, "identity_disjoint", ""), (width / 2, "formula_disjoint", "//")):
        values = [
            report["panels"][panel]["absolute"][method]["retrieval"]["recall@1"]
            for method in methods
        ]
        colors = [STRATUM_COLOR[STRATA[method][0]] for method in methods]
        bars = axis.bar(
            positions + offset, values, width, color=colors, hatch=hatch,
            edgecolor="black", linewidth=0.4,
            label=PANEL_LABEL[panel],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2, value + 0.004, f"{100*value:.1f}",
                ha="center", va="bottom", fontsize=6,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([STRATA[method][1] for method in methods], rotation=32, ha="right", fontsize=7)
    axis.set_ylabel("Recall@1")
    axis.set_ylim(0, 1.06)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for color in STRATUM_COLOR.values()
    ]
    axis.legend(handles=handles + [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black"),
    ], labels=[
        *STRATUM_NAME.values(), "identity (solid) / formula (hatched)",
    ], fontsize=5.5, loc="lower right", ncol=2)


def draw_delta_forest(axis, report: dict, methods: list[str]) -> None:
    baseline = report["baseline_method"]
    positions = np.arange(len(methods))
    for offset, panel, marker in ((-0.12, "identity_disjoint", "o"), (0.12, "formula_disjoint", "^")):
        for position, method in zip(positions, methods):
            if method == baseline:
                continue
            block = report["panels"][panel]["vs_official_dreams"][method]
            delta = block["formula_cluster_paired_ci"]["recall@1"]
            axis.errorbar(
                position + offset, delta["delta_pp"],
                yerr=[[delta["delta_pp"] - delta["ci_low_pp"]], [delta["ci_high_pp"] - delta["delta_pp"]]],
                fmt=marker, markersize=4, linewidth=1, capsize=2.5,
                color=STRATUM_COLOR[STRATA[method][0]],
                ecolor="#444444",
            )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels([STRATA[method][1] for method in methods], rotation=32, ha="right", fontsize=7)
    axis.set_ylabel(r"$\Delta$Recall@1 vs official DreaMS (pp)")
    handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color="black", label=PANEL_LABEL[panel])
        for marker, panel in (("o", "identity_disjoint"), ("^", "formula_disjoint"))
    ]
    axis.legend(handles=handles, fontsize=7)


def draw_recall_k(axis, evaluation: Path, report: dict, methods: list[str], panel: str) -> None:
    for method in methods:
        table = pd.read_csv(evaluation / f"queries_{panel}_{method}.csv.gz")
        values = recall_at_k(table)
        axis.plot(
            RECALL_CUTOFFS, [values[k] for k in RECALL_CUTOFFS],
            marker="o", markersize=3, linewidth=1.2,
            color=STRATUM_COLOR[STRATA[method][0]],
            label=STRATA[method][1],
        )
    axis.set_xscale("log")
    axis.set_xticks(RECALL_CUTOFFS)
    axis.set_xticklabels([str(k) for k in RECALL_CUTOFFS])
    axis.set_xlabel("k")
    axis.set_ylabel("Recall@k")
    axis.set_ylim(0.6, 1.01)
    axis.set_title(f"{PANEL_LABEL[panel]} panel", fontsize=9)
    axis.legend(fontsize=5.5, loc="lower right")


def draw_curves(axis, evaluation: Path, report: dict, methods: list[str], panel: str, curve: str) -> None:
    frame = pd.read_csv(evaluation / f"curves_{panel}.csv.gz")
    frame = frame[frame["curve"] == curve].sort_values(["method", "x"])
    key = "auroc" if curve == "roc" else "auprc"
    for method in methods:
        block = frame[frame["method"] == method]
        value = report["panels"][panel]["absolute"][method]["gnps_10ppm_pooled_pairwise"][key]
        axis.plot(
            block["x"], block["y"], linewidth=1.1,
            color=STRATUM_COLOR[STRATA[method][0]],
            label=f"{STRATA[method][1]}: {value:.3f}",
        )
    if curve == "roc":
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_xlim(-0.005, 1.005)
        axis.set_ylim(0, 1.02)
    else:
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1.005)
        axis.set_ylim(0, 1.05)
    axis.legend(fontsize=5.5, loc="lower left")


def draw_risk_scatter(axis, report: dict, methods: list[str]) -> None:
    baseline = report["baseline_method"]
    maximum = 1.0
    for method in methods:
        if method == baseline:
            continue
        for offset, panel, marker in ((-4, "identity_disjoint", "o"), (4, "formula_disjoint", "^")):
            block = report["panels"][panel]["vs_official_dreams"][method]
            corrected = float(block["corrected"])
            introduced = float(block["introduced"])
            maximum = max(maximum, corrected, introduced)
            axis.scatter(
                introduced, corrected, marker=marker, s=28,
                color=STRATUM_COLOR[STRATA[method][0]], edgecolor="black", linewidth=0.4,
            )
            if method in ("noise_v1", "p2b_noise_v1_frozen"):
                axis.annotate(
                    STRATA[method][1], (introduced, corrected),
                    textcoords="offset points", xytext=(6, offset), fontsize=6,
                )
    span = np.linspace(0, maximum * 1.05, 64)
    axis.plot(span, span, linestyle="--", color="#888888", linewidth=0.9, label="corrected = introduced")
    axis.plot(span, 2 * span, linestyle=":", color="black", linewidth=1.0, label=r"risk-net $\lambda=2$ boundary")
    axis.set_xlabel("Introduced vs official")
    axis.set_ylabel("Corrected vs official")
    axis.set_xlim(-maximum * 0.02, maximum * 1.05)
    axis.set_ylim(-maximum * 0.02, maximum * 1.05)
    handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color="black", label=PANEL_LABEL[panel])
        for marker, panel in (("o", "identity_disjoint"), ("^", "formula_disjoint"))
    ]
    axis.legend(handles=handles, fontsize=7, loc="upper left")


def summary_table(report: dict) -> pd.DataFrame:
    rows = []
    for panel in PANELS:
        for method, block in report["panels"][panel]["absolute"].items():
            paired = report["panels"][panel]["vs_official_dreams"].get(method, {})
            delta = paired.get("formula_cluster_paired_ci", {}).get("recall@1", {})
            stratum, label = STRATA.get(method, ("unknown", method))
            rows.append({
                "panel": panel,
                "method": method,
                "label": label,
                "stratum": stratum,
                "recall_at_1": block["retrieval"]["recall@1"],
                "recall_at_10": block["retrieval"]["recall@10"],
                "mrr": block["retrieval"]["mrr"],
                "near_recall_at_1": block["near_subset"]["recall@1"],
                "pooled_pairwise_auroc": block["gnps_10ppm_pooled_pairwise"]["auroc"],
                "pooled_pairwise_auprc": block["gnps_10ppm_pooled_pairwise"]["auprc"],
                "delta_recall_at_1_pp": delta.get("delta_pp"),
                "delta_recall_at_1_ci_low_pp": delta.get("ci_low_pp"),
                "delta_recall_at_1_ci_high_pp": delta.get("ci_high_pp"),
                "corrected": paired.get("corrected"),
                "introduced": paired.get("introduced"),
                "risk_net_lambda2": paired.get("risk_net_lambda2"),
            })
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.evaluation / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "noise_gnps_article_benchmark_complete":
        raise RuntimeError("evaluation report is not a completed article benchmark")
    methods = method_order(report, "identity_disjoint")
    missing = [method for method in methods if method not in STRATA]
    if missing:
        raise RuntimeError(f"unknown benchmark method(s) need figure metadata: {missing}")
    args.output.mkdir(parents=True)

    figure = plt.figure(figsize=(14.5, 17.5))
    grid = figure.add_gridspec(3, 2, hspace=0.34, wspace=0.22)
    draw_recall_bars(figure.add_subplot(grid[0, 0]), report, methods)
    draw_delta_forest(figure.add_subplot(grid[0, 1]), report, methods)
    draw_recall_k(figure.add_subplot(grid[1, 0]), args.evaluation, report, methods, "identity_disjoint")
    draw_curves(
        figure.add_subplot(grid[1, 1]), args.evaluation, report, methods,
        "identity_disjoint", "roc",
    )
    draw_curves(
        figure.add_subplot(grid[2, 0]), args.evaluation, report, methods,
        "identity_disjoint", "precision_recall",
    )
    draw_risk_scatter(figure.add_subplot(grid[2, 1]), report, methods)
    for axis, tag in zip(figure.axes, "abcdef"):
        axis.set_title(f"{tag}", loc="left", fontweight="bold", fontsize=11)
    figure.suptitle("Noise / P2b on the frozen GNPS Gold/Silver 10 ppm benchmark", fontsize=13)
    figure.savefig(args.output / "main_benchmark_figure.pdf", dpi=args.dpi)
    figure.savefig(args.output / "main_benchmark_figure.png", dpi=args.dpi)
    plt.close(figure)

    supplement = plt.figure(figsize=(14.5, 5.2))
    grid = supplement.add_gridspec(1, 3, hspace=0.3, wspace=0.24)
    draw_recall_k(supplement.add_subplot(grid[0, 0]), args.evaluation, report, methods, "formula_disjoint")
    draw_curves(
        supplement.add_subplot(grid[0, 1]), args.evaluation, report, methods,
        "formula_disjoint", "roc",
    )
    draw_curves(
        supplement.add_subplot(grid[0, 2]), args.evaluation, report, methods,
        "formula_disjoint", "precision_recall",
    )
    supplement.suptitle("Formula-disjoint panel curves", fontsize=12)
    supplement.savefig(args.output / "formula_panel_curves.pdf", dpi=args.dpi)
    supplement.savefig(args.output / "formula_panel_curves.png", dpi=args.dpi)
    plt.close(supplement)

    summary_table(report).to_csv(args.output / "summary_table.csv", index=False)
    print(json.dumps({
        "status": "noise_gnps_article_figures_complete",
        "methods": len(methods),
        "outputs": sorted(path.name for path in args.output.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()
