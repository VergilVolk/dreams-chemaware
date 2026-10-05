"""Smoke contracts for the merged ChemAware GNPS benchmark figure."""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import plot_GLM_chemaware_gnps_unified_benchmark as target  # noqa: E402


def retrieval_block(recall_1: float) -> dict[str, object]:
    return {
        "retrieval": {
            "recall@1": recall_1,
            "recall@5": min(recall_1 + 0.05, 1.0),
            "recall@10": min(recall_1 + 0.08, 1.0),
            "mrr": recall_1 + 0.01,
            "macro_query_auroc": 0.90,
        },
        "gnps_10ppm_pooled_pairwise": {"auroc": 0.92, "auprc": 0.70},
    }


def write_evaluation(
    directory: Path, methods: dict[str, float], shift: float = 0.0,
) -> None:
    directory.mkdir(parents=True)
    panels = {}
    for panel in target.PANELS:
        absolute = {m: retrieval_block(r) for m, r in methods.items()}
        paired = {}
        for method, value in methods.items():
            if method == "official_dreams":
                continue
            paired[method] = {
                "corrected": int(100 * value),
                "introduced": int(60 * value),
                "formula_cluster_paired_ci": {
                    "recall@1": {
                        "delta": (value - methods["official_dreams"]),
                        "ci_low": (value - methods["official_dreams"] - 0.004),
                        "ci_high": (value - methods["official_dreams"] + 0.004),
                    },
                },
            }
        panels[panel] = {
            "absolute": absolute,
            "vs_official_dreams": paired,
            "rankings": {"recall_at_1": [], "pooled_pairwise_auroc": []},
        }
        rows = []
        for method in methods:
            for curve, (xs, ys) in {
                "roc": ([0.0, 0.2, 1.0], [0.0, 0.8 + shift, 1.0]),
                "precision_recall": ([0.0, 0.5, 1.0], [1.0, 0.6, 0.2]),
            }.items():
                rows.extend(
                    {"method": method, "curve": curve, "x": x, "y": y}
                    for x, y in zip(xs, ys)
                )
        with gzip.open(directory / f"curves_{panel}.csv.gz", "wt",
                       encoding="utf-8", newline="") as handle:
            handle.write("method,curve,x,y\n")
            for row in rows:
                handle.write(
                    f"{row['method']},{row['curve']},{row['x']},{row['y']}\n"
                )
    (directory / "report.json").write_text(
        json.dumps(
            {
                "status": "noise_gnps_article_benchmark_complete",
                "baseline_method": "official_dreams",
                "methods": sorted(methods),
                "panels": panels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_merge_and_figure_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as base:
        root = Path(base)
        job_a = {"official_dreams": 0.85, "chemaware_stage1": 0.86}
        job_b = {"official_dreams": 0.85, "chemaware_phaseA": 0.855,
                 "spec2vec_gnps_2019": 0.80}
        write_evaluation(root / "eval_a", job_a, shift=0.0)
        write_evaluation(root / "eval_b", job_b, shift=0.01)
        out = root / "figures"
        sys.argv = [
            "plot",
            "--evaluation", str(root / "eval_a"),
            "--evaluation", str(root / "eval_b"),
            "--output-dir", str(out),
        ]
        target.main()
        assert (out / "summary_table.csv").is_file()
        assert (out / "main_benchmark_figure.png").is_file()
        assert (out / "main_benchmark_figure.pdf").is_file()
        import pandas as pd

        table = pd.read_csv(out / "summary_table.csv")
        merged_methods = set(table["method"])
        assert merged_methods == {
            "official_dreams", "chemaware_stage1", "chemaware_phaseA",
            "spec2vec_gnps_2019",
        }
        assert set(table["stratum"]) <= set(target.STRATUM_COLORS)


def test_ca1_consistency_gate_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as base:
        root = Path(base)
        write_evaluation(root / "eval_a", {"official_dreams": 0.85})
        write_evaluation(root / "eval_b", {"official_dreams": 0.86})
        reports = target.load_evaluations([root / "eval_a", root / "eval_b"])
        with pytest.raises(RuntimeError, match="CA-1"):
            target.consistency_gate(reports)
