#!/usr/bin/env python
"""Plot ladder, top-k, ROC, and PR figures from a mega evaluation directory.

The exact curve CSVs remain the metric source of truth. For rendering only,
this script streams gzip CSVs and uses two-zone deterministic subsampling.
Half of the display budget is reserved for the low-FPR ROC / low-recall PR
front, so operational front resolution is not erased by whole-curve thinning.
Each figure is emitted as both PNG and vector PDF.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOP_K = (1, 2, 3, 5, 10, 20)
LADDER_METRICS = (
    *(f"recall@{cutoff}" for cutoff in TOP_K),
    "mrr",
    "mean_rank",
    "median_rank",
    "macro_query_auroc",
    "macro_query_average_precision",
    "micro_candidate_auroc",
    "micro_candidate_average_precision",
)
LOWER_IS_BETTER = {"mean_rank", "median_rank"}
DISPLAY_NAMES = {
    **{f"recall@{cutoff}": f"R@{cutoff}" for cutoff in TOP_K},
    "mrr": "MRR",
    "mean_rank": "Mean rank",
    "median_rank": "Median rank",
    "macro_query_auroc": "Macro AUROC",
    "macro_query_average_precision": "Macro AP\n(hist. AUPRC)",
    "micro_candidate_auroc": "Micro AUROC",
    "micro_candidate_average_precision": "Micro AP\n(hist. AUPRC)",
}


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New figure directory (defaults to EVALUATION_DIR/figures).",
    )
    parser.add_argument(
        "--max-curve-points",
        type=int,
        default=25_000,
        help="Maximum points loaded per method for display only.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--roc-front-fpr", type=float, default=0.01,
        help="Reserve half of display points for FPR at or below this value.",
    )
    parser.add_argument(
        "--pr-front-recall", type=float, default=0.10,
        help="Reserve half of display points for recall at or below this value.",
    )
    parser.add_argument(
        "--title",
        default="Large-scale labelled identity benchmark",
        help="Figure title prefix.",
    )
    return parser.parse_args(argv)


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"summary table is empty: {path}")
    output: list[dict[str, object]] = []
    for row in rows:
        parsed: dict[str, object] = {"method": row["method"]}
        for key, value in row.items():
            if key == "method":
                continue
            parsed[key] = float(value) if value not in (None, "") else math.nan
        output.append(parsed)
    return output


def _competition_ranks(values: np.ndarray, lower_is_better: bool) -> np.ndarray:
    order = np.argsort(values if lower_is_better else -values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = 1.0 + 0.5 * (left + right - 1)
        left = right
    return ranks


def _metric_annotation(metric: str, value: float) -> str:
    if metric in LOWER_IS_BETTER:
        return f"{value:.2f}"
    return f"{100 * value:.1f}%"


def plot_ladder(
    rows: list[dict[str, object]], title: str, output: Path, dpi: int
) -> list[str]:
    methods = [str(row["method"]) for row in rows]
    values = np.asarray(
        [[float(row[metric]) for metric in LADDER_METRICS] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("ladder metrics contain missing/non-finite values")
    ranks = np.column_stack(
        [
            _competition_ranks(values[:, column], metric in LOWER_IS_BETTER)
            for column, metric in enumerate(LADDER_METRICS)
        ]
    )
    # The ladder preset is frozen to the primary endpoint, Recall@1. Other
    # metrics colour cells but never silently change the row ordering.
    display_order = np.asarray(
        sorted(range(len(methods)), key=lambda index: (-values[index, 0], methods[index])),
        dtype=np.int64,
    )
    values = values[display_order]
    ranks = ranks[display_order]
    methods = [methods[index] for index in display_order]

    width = max(15.0, 1.18 * len(LADDER_METRICS))
    height = max(3.2, 0.62 * len(methods) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(ranks, aspect="auto", cmap="viridis_r", vmin=1, vmax=max(1, len(methods)))
    for row in range(len(methods)):
        for column, metric in enumerate(LADDER_METRICS):
            color = "white" if ranks[row, column] > max(1.5, len(methods) * 0.55) else "black"
            ax.text(
                column,
                row,
                _metric_annotation(metric, values[row, column]),
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )
    ax.set_xticks(range(len(LADDER_METRICS)), [DISPLAY_NAMES[key] for key in LADDER_METRICS])
    ax.set_yticks(range(len(methods)), methods)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(
        f"{title}\nRecall@1 ladder (rows preset by R@1; cells show within-metric rank)"
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.015)
    colorbar.set_label("Rank (1 = best)")
    colorbar.set_ticks(range(1, len(methods) + 1))
    _save_both(fig, output / "ladder", dpi)
    return methods


def plot_topk(rows: list[dict[str, object]], title: str, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
    for row in rows:
        values = [100 * float(row[f"recall@{cutoff}"]) for cutoff in TOP_K]
        ax.plot(TOP_K, values, marker="o", linewidth=2.0, markersize=4.5, label=str(row["method"]))
    ax.set_xticks(TOP_K)
    ax.set_xlim(min(TOP_K), max(TOP_K))
    ax.set_ylim(0, 101)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (%)")
    ax.set_title(f"{title}\nTop-k retrieval")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=max(1, math.ceil(len(rows) / 10)))
    _save_both(fig, output / "topk", dpi)


def stream_curve(
    path: Path,
    x_field: str,
    y_field: str,
    total_points: int,
    maximum_points: int,
    front_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    def open_text():
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", newline="")
        return path.open("r", encoding="utf-8", newline="")

    observed = 0
    front_points = 0
    previous_x = -math.inf
    with open_text() as handle:
        for row in csv.DictReader(handle):
            current_x = float(row[x_field])
            if current_x < previous_x:
                raise RuntimeError(f"curve x-axis is not monotone in {path}")
            previous_x = current_x
            observed += 1
            if current_x <= front_limit:
                front_points += 1
    if observed != total_points:
        raise RuntimeError(
            f"curve point count mismatch for {path}: report={total_points}, CSV={observed}"
        )
    if observed <= maximum_points:
        x: list[float] = []
        y: list[float] = []
        with open_text() as handle:
            for row in csv.DictReader(handle):
                x.append(float(row[x_field]))
                y.append(float(row[y_field]))
        if not x:
            raise RuntimeError(f"curve CSV contains no points: {path}")
        return np.asarray(x), np.asarray(y)
    remainder_points = observed - front_points
    if front_points == 0:
        front_budget, remainder_budget = 0, maximum_points
    elif remainder_points == 0:
        front_budget, remainder_budget = maximum_points, 0
    else:
        front_budget = min(front_points, maximum_points // 2)
        remainder_budget = min(remainder_points, maximum_points - front_budget)
        unused = maximum_points - front_budget - remainder_budget
        add_front = min(unused, front_points - front_budget)
        front_budget += add_front
        unused -= add_front
        remainder_budget += min(unused, remainder_points - remainder_budget)

    def state(count: int, budget: int) -> dict[str, object]:
        buckets = max(1, budget // 4) if count > budget else 0
        return {
            "count": count,
            "budget": budget,
            "bucket_size": max(1, math.ceil(count / buckets)) if buckets else 0,
            "local": 0,
            "bucket": -1,
            "first": None,
            "last": None,
            "minimum": None,
            "maximum": None,
        }

    front_state = state(front_points, front_budget)
    remainder_state = state(remainder_points, remainder_budget)
    selected: list[tuple[int, float, float]] = []

    def flush(current: dict[str, object]) -> None:
        points = [
            current["first"], current["minimum"], current["maximum"], current["last"]
        ]
        unique = {
            point[0]: point for point in points if point is not None  # type: ignore[index]
        }
        selected.extend(unique[index] for index in sorted(unique))
        current["first"] = current["last"] = None
        current["minimum"] = current["maximum"] = None

    def add(current: dict[str, object], point: tuple[int, float, float]) -> None:
        local = int(current["local"])
        count = int(current["count"])
        budget = int(current["budget"])
        if count <= budget:
            selected.append(point)
            current["local"] = local + 1
            return
        bucket = local // int(current["bucket_size"])
        if bucket != int(current["bucket"]):
            if current["first"] is not None:
                flush(current)
            current["bucket"] = bucket
            current["first"] = point
            current["minimum"] = point
            current["maximum"] = point
        current["last"] = point
        if point[2] < current["minimum"][2]:  # type: ignore[index]
            current["minimum"] = point
        if point[2] > current["maximum"][2]:  # type: ignore[index]
            current["maximum"] = point
        current["local"] = local + 1

    with open_text() as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            x_value = float(row[x_field])
            point = (index, x_value, float(row[y_field]))
            add(front_state if x_value <= front_limit else remainder_state, point)
    if front_state["first"] is not None:
        flush(front_state)
    if remainder_state["first"] is not None:
        flush(remainder_state)
    selected = [
        value for _, value in sorted({point[0]: point for point in selected}.items())
    ]
    if len(selected) > maximum_points:
        raise AssertionError("front-preserving sampler exceeded its display budget")
    if not selected:
        raise RuntimeError(f"curve CSV contains no points: {path}")
    return (
        np.asarray([point[1] for point in selected]),
        np.asarray([point[2] for point in selected]),
    )


def plot_curves(
    evaluation: Path,
    report: dict[str, object],
    title: str,
    output: Path,
    maximum_points: int,
    roc_front_fpr: float,
    pr_front_recall: float,
    dpi: int,
) -> None:
    methods = report.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise RuntimeError("report.json contains no method results")
    fig_roc, ax_roc = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    fig_pr, ax_pr = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    prevalence: float | None = None
    for name, value in methods.items():
        if not isinstance(value, dict):
            raise RuntimeError(f"malformed method result: {name}")
        micro = value["micro_candidate"]
        curves = value["curves"]
        points = int(micro["curve_points"])
        roc_x, roc_y = stream_curve(
            evaluation / curves["roc_csv_gz"],
            "false_positive_rate",
            "true_positive_rate",
            points,
            maximum_points,
            roc_front_fpr,
        )
        pr_x, pr_y = stream_curve(
            evaluation / curves["pr_csv_gz"],
            "recall",
            "precision",
            points,
            maximum_points,
            pr_front_recall,
        )
        ax_roc.plot(roc_x, roc_y, linewidth=1.8, label=f"{name} ({float(micro['auroc']):.4f})")
        ax_pr.plot(
            pr_x, pr_y, linewidth=1.8,
            label=f"{name} ({float(micro['average_precision']):.4f})",
        )
        current_prevalence = (
            float(micro["positive_molecule_edges"]) / float(micro["molecule_edges"])
        )
        prevalence = current_prevalence if prevalence is None else prevalence
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1, label="Chance")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    ax_roc.set_title(f"{title}\nMicro candidate ROC (legend: AUROC)")
    ax_roc.grid(True, alpha=0.22)
    ax_roc.legend(frameon=False, fontsize=8)
    assert prevalence is not None
    ax_pr.axhline(prevalence, linestyle="--", color="0.55", linewidth=1, label=f"Prevalence ({prevalence:.4g})")
    ax_pr.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax_pr.set_title(f"{title}\nMicro precision-recall (legend: AP / historical AUPRC)")
    ax_pr.grid(True, alpha=0.22)
    ax_pr.legend(frameon=False, fontsize=8)
    _save_both(fig_roc, output / "roc", dpi)
    _save_both(fig_pr, output / "pr", dpi)


def _save_both(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(args: argparse.Namespace) -> dict[str, object]:
    evaluation = args.evaluation_dir.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation / "figures"
    )
    if not evaluation.is_dir():
        raise FileNotFoundError(evaluation)
    if output.exists():
        raise FileExistsError(output)
    if args.max_curve_points < 8 or args.dpi < 72:
        raise ValueError("max-curve-points must be >=8 and dpi must be >=72")
    if not (0 < args.roc_front_fpr < 1) or not (0 < args.pr_front_recall < 1):
        raise ValueError("front thresholds must lie strictly between zero and one")
    with (evaluation / "report.json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    rows = load_summary(evaluation / "summary_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        ladder_order = plot_ladder(rows, args.title, staging, args.dpi)
        plot_topk(rows, args.title, staging, args.dpi)
        plot_curves(
            evaluation,
            report,
            args.title,
            staging,
            args.max_curve_points,
            args.roc_front_fpr,
            args.pr_front_recall,
            args.dpi,
        )
        plot_report = {
            "status": "complete",
            "evaluation_dir": str(evaluation),
            "methods": [row["method"] for row in rows],
            "ladder_preset": "recall@1_descending_then_method_name",
            "ladder_order": ladder_order,
            "curve_display_max_points_per_method": args.max_curve_points,
            "front_preserving_sampling": {
                "roc_fpr_at_or_below": args.roc_front_fpr,
                "pr_recall_at_or_below": args.pr_front_recall,
                "display_budget_fraction": 0.5,
            },
            "exact_curve_csvs_preserved": True,
            "figures": [
                f"{name}.{extension}"
                for name in ("ladder", "topk", "roc", "pr")
                for extension in ("png", "pdf")
            ],
        }
        with (staging / "plot_report.json").open("w", encoding="utf-8") as handle:
            json.dump(plot_report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        staging.replace(output)
        print(json.dumps(plot_report, indent=2, ensure_ascii=False), flush=True)
        return plot_report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    plot(arguments(argv))


if __name__ == "__main__":
    main()
