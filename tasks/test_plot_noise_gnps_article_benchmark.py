#!/usr/bin/env python
"""Smoke test: render article figures from a schema-exact synthetic evaluation."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import plot_noise_gnps_article_benchmark as plotting

METHODS = ("official_dreams", "cosine_greedy", "p2b_official_frozen")
PANELS = ("identity_disjoint", "formula_disjoint")


def synthetic_evaluation(root: Path) -> Path:
    evaluation = root / "evaluation"
    evaluation.mkdir()
    rng = np.random.default_rng(20261003)
    panels: dict[str, object] = {}
    for panel_index, panel in enumerate(PANELS):
        absolute: dict[str, object] = {}
        paired: dict[str, object] = {}
        curves: list[dict] = []
        rankings_absolute = []
        for method_index, method in enumerate(METHODS):
            rank = rng.integers(1, 4, 60)
            if method_index:
                rank = np.maximum(1, rank - method_index)
            pd.DataFrame({
                "query_index": np.arange(60),
                "rank": rank,
                "near": rng.random(60) < 0.5,
            }).to_csv(
                evaluation / f"queries_{panel}_{method}.csv.gz",
                index=False, compression="gzip",
            )
            recall1 = float(np.mean(rank == 1))
            absolute[method] = {
                "retrieval": {
                    "recall@1": recall1,
                    "recall@10": float(np.mean(rank <= 10)),
                    "mrr": float(np.mean(1.0 / rank)),
                },
                "near_subset": {"recall@1": recall1 - 0.05},
                "gnps_10ppm_pooled_pairwise": {
                    "auroc": 0.90 + 0.01 * method_index,
                    "auprc": 0.65 + 0.02 * method_index,
                },
            }
            rankings_absolute.append({"method": method, "value": recall1})
            if method != "official_dreams":
                paired[method] = {
                    "corrected": 20 + 5 * method_index,
                    "introduced": 10 + panel_index,
                    "risk_net_lambda2": 5 * method_index - panel_index,
                    "formula_cluster_paired_ci": {"recall@1": {
                        "delta_pp": 0.5 * method_index,
                        "ci_low_pp": 0.5 * method_index - 0.4,
                        "ci_high_pp": 0.5 * method_index + 0.4,
                    }},
                }
            threshold = np.linspace(0, 1, 25)
            curves.extend(
                {"method": method, "curve": "roc", "x": float(x),
                 "y": float(min(1.0, x ** (0.5 - 0.05 * method_index) + 0.2)),
                 "threshold": float(1 - x)}
                for x in threshold
            )
            curves.extend(
                {"method": method, "curve": "precision_recall", "x": float(x),
                 "y": float(0.9 + 0.03 * method_index - 0.3 * x), "threshold": float(1 - x)}
                for x in threshold
            )
        pd.DataFrame(curves).to_csv(
            evaluation / f"curves_{panel}.csv.gz", index=False, compression="gzip",
        )
        panels[panel] = {
            "absolute": absolute,
            "vs_official_dreams": paired,
            "rankings": {
                "recall_at_1": sorted(
                    rankings_absolute, key=lambda row: (-row["value"], row["method"]),
                ),
                "pooled_pairwise_auroc": sorted(
                    [
                        {"method": method, "value": block["gnps_10ppm_pooled_pairwise"]["auroc"]}
                        for method, block in absolute.items()
                    ],
                    key=lambda row: (-row["value"], row["method"]),
                ),
            },
        }
    report = {
        "status": "noise_gnps_article_benchmark_complete",
        "baseline_method": "official_dreams",
        "methods": list(METHODS),
        "panels": panels,
    }
    (evaluation / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return evaluation


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        evaluation = synthetic_evaluation(root)
        output = root / "figures"
        previous = plotting.arguments
        plotting.arguments = lambda: SimpleNamespace(
            evaluation=evaluation, output=output, dpi=80,
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plotting.main()
        finally:
            plotting.arguments = previous
        expected = {
            "main_benchmark_figure.pdf", "main_benchmark_figure.png",
            "formula_panel_curves.pdf", "formula_panel_curves.png",
            "summary_table.csv",
        }
        names = {path.name for path in output.iterdir()}
        assert expected <= names, expected - names
        for name in ("main_benchmark_figure.png", "formula_panel_curves.png"):
            assert (output / name).stat().st_size > 20_000, name
        table = pd.read_csv(output / "summary_table.csv")
        assert len(table) == len(METHODS) * len(PANELS)
        assert set(table["stratum"]) == {"embedding", "classical", "reranker"}
    print("[test_plot_noise_gnps_article_benchmark] PASS")


if __name__ == "__main__":
    main()
