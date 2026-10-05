#!/usr/bin/env python
"""Merge unified GNPS benchmark evaluations into the ChemAware paper figure.

Consumes one or more ``evaluation`` directories produced by
``evaluate_noise_gnps_article_benchmark.py`` (each containing ``report.json``
plus ``curves_<panel>.csv.gz``), merges every method across jobs, and emits:

- ``summary_table.csv``: method x panel x key metrics with paired deltas and
  formula-cluster confidence intervals versus the frozen baseline;
- ``main_benchmark_figure.png/pdf``: a) grouped Recall@1 bars on both panels,
  b) delta-versus-baseline forest with CI, c) Recall@k curves,
  d) pooled ROC with AUROC annotation, e) precision-recall curves,
  f) corrected/introduced scatter with the lambda=2 risk line;
- ``panel_supplement_figure.png/pdf`` per panel.

The same method appearing in several evaluations must agree exactly in its
absolute Recall@1 -- this is the CA-1 cross-pipeline consistency gate from the
ChemAware preregistration and it fails closed.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PANELS = ("identity_disjoint", "formula_disjoint")
PANEL_LABELS = {
    "identity_disjoint": "Identity-disjoint (10,995)",
    "formula_disjoint": "Formula-disjoint (5,261)",
}
STRATUM_COLORS = {
    "S1 classical": "#8c8c8c",
    "S1b P2b channel": "#b3b3b3",
    "S2 DreaMS family": "#1f77b4",
    "S2 ChemAware": "#d62728",
    "S4 public neural (utility only)": "#ff9820",
    "unlabeled": "#7f7f7f",
}
DEFAULT_STRATUM = {
    "official_dreams": "S2 DreaMS family",
    "noise_v1": "S2 DreaMS family",
    "chemaware_stage1": "S2 ChemAware",
    "chemaware_phaseA": "S2 ChemAware",
    "spec2vec_gnps_2019": "S4 public neural (utility only)",
    "ms2deepscore_dual_2024": "S4 public neural (utility only)",
}
CLASSICAL = (
    "cosine_greedy", "modified_cosine", "weighted_spectral_entropy",
    "neutral_loss_sqrt_cosine",
)
P2B_CHANNEL = (
    "p2b_sqrt_cosine", "p2b_unweighted_entropy", "p2b_official_frozen",
    "p2b_noise_v1_frozen",
)


def stratum_of(method: str) -> str:
    if method in DEFAULT_STRATUM:
        return DEFAULT_STRATUM[method]
    if method in CLASSICAL:
        return "S1 classical"
    if method in P2B_CHANNEL:
        return "S1b P2b channel"
    return "unlabeled"


def load_evaluations(paths: list[Path]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for path in paths:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        reports[str(path.resolve())] = report
    return reports


def consistency_gate(reports: dict[str, dict]) -> dict[str, dict[str, dict[str, float]]]:
    """CA-1: identical methods across evaluations must reproduce exactly."""
    merged: dict[str, dict[str, dict[str, object]]] = {}
    for source, report in reports.items():
        for panel, block in report["panels"].items():
            for method, metrics in block["absolute"].items():
                value = float(metrics["retrieval"]["recall@1"])
                slot = merged.setdefault(method, {}).setdefault(
                    panel, {"recall@1": value, "sources": []},
                )
                if abs(float(slot["recall@1"]) - value) > 1e-12:
                    raise RuntimeError(
                        "CA-1 cross-pipeline consistency violation: "
                        f"{method}/{panel} recall@1 {slot['recall@1']} vs "
                        f"{value} ({source})"
                    )
                slot["sources"].append(source)
    return merged


def build_rows(
    reports: dict[str, dict], baseline: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for report in reports.values():
        for panel, block in report["panels"].items():
            for method, metrics in block["absolute"].items():
                if (method, panel) in seen:
                    continue
                seen.add((method, panel))
                retrieval = metrics["retrieval"]
                paired = block.get("vs_official_dreams", {}).get(method)
                row: dict[str, object] = {
                    "method": method,
                    "stratum": stratum_of(method),
                    "panel": panel,
                    "recall_at_1": retrieval["recall@1"],
                    "recall_at_5": retrieval["recall@5"],
                    "recall_at_10": retrieval["recall@10"],
                    "mrr": retrieval["mrr"],
                    "macro_query_auroc": retrieval["macro_query_auroc"],
                    "pooled_pairwise_auroc":
                        metrics["gnps_10ppm_pooled_pairwise"]["auroc"],
                    "pooled_pairwise_auprc":
                        metrics["gnps_10ppm_pooled_pairwise"]["auprc"],
                    "delta_recall_at_1": None,
                    "delta_ci_low": None,
                    "delta_ci_high": None,
                    "corrected": None,
                    "introduced": None,
                }
                if paired is not None:
                    ci = paired["formula_cluster_paired_ci"]["recall@1"]
                    row.update({
                        "delta_recall_at_1": ci["delta"],
                        "delta_ci_low": ci["ci_low"],
                        "delta_ci_high": ci["ci_high"],
                        "corrected": paired.get("corrected"),
                        "introduced": paired.get("introduced"),
                    })
                rows.append(row)
    return rows


def load_curves(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        for panel in PANELS:
            curve_file = path / f"curves_{panel}.csv.gz"
            if curve_file.is_file():
                with gzip.open(curve_file, "rt", encoding="utf-8") as handle:
                    frame = pd.read_csv(handle)
                frame["panel"] = panel
                frames.append(frame)
    if not frames:
        raise RuntimeError("no curves_*.csv.gz found in any evaluation directory")
    return pd.concat(frames, ignore_index=True)


def plot_main(
    rows: pd.DataFrame,
    curves: pd.DataFrame,
    baseline: str,
    output: Path,
) -> None:
    order = (
        rows[rows["panel"] == PANELS[0]]
        .sort_values(["recall_at_1"], ascending=False)["method"]
        .tolist()
    )
    figure, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    # a) grouped Recall@1 bars on both panels.
    axis = axes[0, 0]
    width = 0.38
    positions = np.arange(len(order))
    for offset, panel, label in (
        (-width / 2, PANELS[0], "Identity"), (width / 2, PANELS[1], "Formula"),
    ):
        block = (
            rows[rows["panel"] == panel].set_index("method")
            .reindex(order)["recall_at_1"]
        )
        colors = [STRATUM_COLORS[stratum_of(m)] for m in order]
        axis.bar(
            positions + offset, block.to_numpy(), width=width,
            color=colors, alpha=0.55 if label == "Formula" else 1.0,
            edgecolor="black", linewidth=0.4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
    axis.set_ylabel("Recall@1")
    axis.set_title("a) Recall@1 by panel")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in STRATUM_COLORS.values()
    ]
    axis.legend(
        handles, list(STRATUM_COLORS), fontsize=6, loc="lower right", ncol=2,
    )

    # b) delta versus baseline forest with CI.
    axis = axes[0, 1]
    forest = rows[
        (rows["panel"] == PANELS[0]) & (rows["delta_recall_at_1"].notna())
    ].set_index("method").reindex(order)
    y = np.arange(len(order))
    axis.errorbar(
        forest["delta_recall_at_1"] * 100, y,
        xerr=[
            (forest["delta_recall_at_1"] - forest["delta_ci_low"]) * 100,
            (forest["delta_ci_high"] - forest["delta_recall_at_1"]) * 100,
        ],
        fmt="o", color="#d62728", ecolor="#999999", capsize=2, ms=3,
    )
    axis.axvline(0, color="black", lw=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(order, fontsize=7)
    axis.set_xlabel("ΔRecall@1 vs official (pp), identity panel")
    axis.set_title("b) Paired delta with formula-cluster CI")

    # c) Recall@k curves.
    axis = axes[0, 2]
    for method in order:
        block = (
            rows[(rows["panel"] == PANELS[0]) & (rows["method"] == method)]
            .iloc[0]
        )
        recalls = [block[f"recall_at_{k}"] for k in (1, 5, 10)]
        axis.plot([1, 5, 10], recalls, marker="o", ms=3, lw=1, label=method)
    axis.set_xlabel("k")
    axis.set_ylabel("Recall@k")
    axis.set_title("c) Recall@k, identity panel")
    axis.legend(fontsize=5, ncol=2, loc="lower right")

    # d) pooled ROC.
    axis = axes[1, 0]
    roc = curves[curves["curve"] == "roc"]
    for method in order:
        block = roc[(roc["panel"] == PANELS[0]) & (roc["method"] == method)]
        if not len(block):
            continue
        block = block.sort_values("x")
        auroc = rows[
            (rows["panel"] == PANELS[0]) & (rows["method"] == method)
        ]["pooled_pairwise_auroc"].iloc[0]
        axis.plot(
            block["x"], block["y"], lw=1,
            label=f"{method} ({auroc:.3f})",
        )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("d) Pooled pairwise ROC, identity panel")
    axis.legend(fontsize=5, loc="lower right")

    # e) precision-recall.
    axis = axes[1, 1]
    pr = curves[curves["curve"] == "precision_recall"]
    for method in order:
        block = pr[(pr["panel"] == PANELS[0]) & (pr["method"] == method)]
        if not len(block):
            continue
        block = block.sort_values("x")
        auprc = rows[
            (rows["panel"] == PANELS[0]) & (rows["method"] == method)
        ]["pooled_pairwise_auprc"].iloc[0]
        axis.plot(
            block["x"], block["y"], lw=1,
            label=f"{method} ({auprc:.3f})",
        )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("e) Pooled pairwise PR, identity panel")
    axis.legend(fontsize=5, loc="upper right")

    # f) corrected/introduced scatter with lambda=2 risk line.
    axis = axes[1, 2]
    scatter = rows[
        (rows["panel"] == PANELS[0]) & (rows["corrected"].notna())
    ]
    axis.scatter(
        scatter["introduced"], scatter["corrected"],
        c=[STRATUM_COLORS[stratum_of(m)] for m in scatter["method"]],
        s=28, edgecolor="black", linewidth=0.4,
    )
    limit = float(
        np.nanmax(np.concatenate([scatter["corrected"], scatter["introduced"]])) * 1.15
    ) + 1
    axis.plot([0, limit], [0, 2 * limit], "--", color="black", lw=0.9,
              label="λ=2 break-even")
    for _, row in scatter.iterrows():
        axis.annotate(row["method"], (row["introduced"], row["corrected"]),
                      fontsize=5, xytext=(2, 2), textcoords="offset points")
    axis.set_xlabel("Introduced")
    axis.set_ylabel("Corrected")
    axis.set_title("f) Corrected vs introduced, identity panel")
    axis.legend(fontsize=6, loc="lower right")

    figure.suptitle(
        "Unified GNPS Gold/Silver benchmark — merged methods "
        f"(baseline {baseline}); S4 public neural = utility tier only",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", type=Path, action="append", required=True,
        help="An evaluation directory containing report.json and curves CSVs",
    )
    parser.add_argument("--baseline", default="official_dreams")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reports = load_evaluations(args.evaluation)
    consistency_gate(reports)
    rows = pd.DataFrame(build_rows(reports, args.baseline))
    curves = load_curves(args.evaluation)
    args.output_dir.mkdir(parents=True)
    rows.to_csv(args.output_dir / "summary_table.csv", index=False)
    plot_main(rows, curves, args.baseline, args.output_dir / "main_benchmark_figure")
    print(
        json.dumps(
            {
                "status": "GLM_CHEMAWARE_GNPS_UNIFIED_FIGURE_COMPLETE",
                "evaluations": [str(path) for path in args.evaluation],
                "methods": sorted(rows["method"].unique().tolist()),
                "rows": int(len(rows)),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
