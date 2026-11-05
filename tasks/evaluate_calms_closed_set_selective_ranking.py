#!/usr/bin/env python3
"""Evaluate CALMS confidence for selective *closed-set* candidate ranking.

Every query in this analysis has exactly one true molecule in its fixed
candidate block.  The Top-1--Top-2 score gap is used only to order queries
from higher to lower confidence.  Consequently, the resulting risk--coverage
curve measures selective closed-set ranking; it does not evaluate no-match
detection, open-set false acceptance, or a calibrated match probability.

The frozen CALMS v1.3.2 scores have already been materialized and evaluated.
This script consumes their per-query table and never refits or rescales CALMS.
If matching WSE tables are available, it also reports CALMS-minus-WSE
functional differences at each method's own high-confidence prefix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PANELS = ("identity_disjoint", "formula_disjoint")
EXPECTED_PANEL_QUERIES = {"identity_disjoint": 10_995, "formula_disjoint": 5_261}
FIXED_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)
REQUIRED_COLUMNS = (
    "query_index",
    "query_row",
    "query_formula",
    "rank",
    "top1_top2_gap",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calms-identity",
        type=Path,
        default=Path(
            "deliverables/GLM_gnps_article_ladder/run16_full23_v132/"
            "queries_identity_disjoint_full23_v132.csv.gz"
        ),
    )
    parser.add_argument(
        "--calms-formula",
        type=Path,
        default=Path(
            "deliverables/GLM_gnps_article_ladder/run16_full23_v132/"
            "queries_formula_disjoint_full23_v132.csv.gz"
        ),
    )
    parser.add_argument(
        "--wse-identity",
        type=Path,
        default=Path(
            "data/validation/GLM_gnps_article_benchmark_s2v26/run15/evaluation/"
            "queries_identity_disjoint_weighted_spectral_entropy.csv.gz"
        ),
    )
    parser.add_argument(
        "--wse-formula",
        type=Path,
        default=Path(
            "data/validation/GLM_gnps_article_benchmark_s2v26/run15/evaluation/"
            "queries_formula_disjoint_weighted_spectral_entropy.csv.gz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261105)
    return parser.parse_args()


def load_query_table(path: Path, method: str, panel: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise RuntimeError(f"{method}/{panel}: missing columns {missing}")
    table = table.loc[:, REQUIRED_COLUMNS].copy()
    if table.empty:
        raise RuntimeError(f"{method}/{panel}: empty query table")
    if table["query_index"].duplicated().any():
        raise RuntimeError(f"{method}/{panel}: duplicate query_index")
    if table["query_row"].duplicated().any():
        raise RuntimeError(f"{method}/{panel}: duplicate query_row")

    for column in ("query_index", "query_row", "rank", "top1_top2_gap"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if not np.all(np.isfinite(table[column].to_numpy(dtype=np.float64))):
            raise RuntimeError(f"{method}/{panel}: non-finite {column}")
    for column in ("query_index", "query_row", "rank"):
        values = table[column].to_numpy(dtype=np.float64)
        if not np.all(values == np.floor(values)):
            raise RuntimeError(f"{method}/{panel}: non-integral {column}")
        table[column] = values.astype(np.int64)
    if np.any(table["query_index"].to_numpy() < 0):
        raise RuntimeError(f"{method}/{panel}: negative query_index")
    if np.any(table["rank"].to_numpy() < 1):
        raise RuntimeError(f"{method}/{panel}: rank must be positive")
    if np.any(table["top1_top2_gap"].to_numpy(dtype=np.float64) < 0):
        raise RuntimeError(f"{method}/{panel}: negative Top1--Top2 gap")

    if table["query_formula"].isna().any():
        raise RuntimeError(f"{method}/{panel}: missing query formula")
    table["query_formula"] = table["query_formula"].astype(str).str.strip()
    if table["query_formula"].eq("").any():
        raise RuntimeError(f"{method}/{panel}: empty query formula")
    table = table.sort_values("query_index", kind="stable").reset_index(drop=True)
    expected = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(table["query_index"].to_numpy(), expected):
        raise RuntimeError(f"{method}/{panel}: query_index must be contiguous 0..n-1")
    return table


def assert_same_axis(candidate: pd.DataFrame, comparator: pd.DataFrame, panel: str) -> None:
    for column in ("query_index", "query_row", "query_formula"):
        if not np.array_equal(candidate[column].to_numpy(), comparator[column].to_numpy()):
            raise RuntimeError(f"{panel}: CALMS/WSE {column} axis mismatch")


def confidence_order(table: pd.DataFrame) -> np.ndarray:
    """Return a deterministic, truth-blind descending confidence order."""
    gap = table["top1_top2_gap"].to_numpy(dtype=np.float64)
    row = table["query_row"].to_numpy(dtype=np.int64)
    return np.lexsort((row, -gap)).astype(np.int64)


def optimal_aurc(correct: np.ndarray) -> float:
    n = len(correct)
    n_correct = int(np.sum(correct))
    prefix = np.arange(1, n + 1, dtype=np.float64)
    ideal_errors = np.maximum(0, prefix - n_correct)
    return float(np.mean(ideal_errors / prefix))


def global_metrics(table: pd.DataFrame) -> dict[str, float | int]:
    correct = table["rank"].to_numpy(dtype=np.int64) == 1
    if np.all(correct) or not np.any(correct):
        raise RuntimeError("correctness AUROC requires both correct and incorrect queries")
    gap = table["top1_top2_gap"].to_numpy(dtype=np.float64)
    order = confidence_order(table)
    prefix = np.arange(1, len(table) + 1, dtype=np.float64)
    risk = 1.0 - np.cumsum(correct[order]) / prefix
    aurc = float(np.mean(risk))
    ideal = optimal_aurc(correct)
    return {
        "queries": int(len(table)),
        "correct_queries": int(np.sum(correct)),
        "incorrect_queries": int(np.sum(~correct)),
        "full_coverage_recall_at_1": float(np.mean(correct)),
        "correctness_auroc_from_margin": float(roc_auc_score(correct, gap)),
        "error_auprc_from_negative_margin": float(
            average_precision_score(~correct, -gap)
        ),
        "error_prevalence": float(np.mean(~correct)),
        "aurc": aurc,
        "optimal_aurc_at_observed_error_rate": ideal,
        "excess_aurc": aurc - ideal,
    }


def _retained_count(n: int, coverage: float) -> int:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")
    return n if coverage == 1 else max(1, int(math.floor(coverage * n)))


def fixed_coverage_points(
    table: pd.DataFrame,
    method: str,
    panel: str,
    coverages: Iterable[float] = FIXED_COVERAGES,
) -> list[dict[str, float | int | str | bool]]:
    correct = table["rank"].to_numpy(dtype=np.int64) == 1
    gap = table["top1_top2_gap"].to_numpy(dtype=np.float64)
    order = confidence_order(table)
    rows: list[dict[str, float | int | str | bool]] = []
    for coverage in coverages:
        retained = _retained_count(len(table), float(coverage))
        chosen = order[:retained]
        accuracy = float(np.mean(correct[chosen]))
        threshold = float(gap[order[retained - 1]])
        tie_size = int(np.sum(gap == threshold))
        tie_retained = int(np.sum(gap[chosen] == threshold))
        rows.append({
            "panel": panel,
            "method": method,
            "target_coverage": float(coverage),
            "retained_queries": int(retained),
            "achieved_coverage": float(retained / len(table)),
            "selective_recall_at_1": accuracy,
            "selective_risk": 1.0 - accuracy,
            "correct_yield_all_queries": float(np.sum(correct[chosen]) / len(table)),
            "accepted_errors_all_queries": float(np.sum(~correct[chosen]) / len(table)),
            "gap_threshold": threshold,
            "boundary_tie_size": tie_size,
            "boundary_tie_retained": tie_retained,
            "boundary_tie_split": bool(0 < tie_retained < tie_size),
            "boundary_tie_split": bool(0 < tie_retained < tie_size),
        })
    return rows


def _weighted_prefix_accuracy(
    correct: np.ndarray,
    order: np.ndarray,
    query_weights: np.ndarray,
    coverage: float,
) -> float:
    weights = query_weights[order].astype(np.float64, copy=False)
    total = int(np.sum(weights))
    if total <= 0:
        raise RuntimeError("bootstrap replicate contains no queries")
    retained = _retained_count(total, coverage)
    cumulative_weight = np.cumsum(weights)
    index = int(np.searchsorted(cumulative_weight, retained, side="left"))
    previous_weight = float(cumulative_weight[index - 1]) if index else 0.0
    previous_correct = (
        float(np.sum(weights[:index] * correct[order[:index]])) if index else 0.0
    )
    final_take = float(retained - previous_weight)
    selected_correct = previous_correct + final_take * float(correct[order[index]])
    return selected_correct / retained


def formula_cluster_bootstrap(
    tables: dict[str, pd.DataFrame],
    coverages: tuple[float, ...],
    resamples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], int]:
    if resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    reference = next(iter(tables.values()))
    formulas = reference["query_formula"].to_numpy(dtype=str)
    unique_formula, inverse = np.unique(formulas, return_inverse=True)
    methods = tuple(tables)
    correct = {
        method: table["rank"].to_numpy(dtype=np.int64) == 1
        for method, table in tables.items()
    }
    order = {method: confidence_order(table) for method, table in tables.items()}
    values = {
        method: np.empty((resamples, len(coverages)), dtype=np.float64)
        for method in methods
    }
    rng = np.random.default_rng(seed)
    for replicate in range(resamples):
        sampled = rng.integers(0, len(unique_formula), size=len(unique_formula))
        cluster_weight = np.bincount(sampled, minlength=len(unique_formula))
        query_weight = cluster_weight[inverse]
        for method in methods:
            for coverage_index, coverage in enumerate(coverages):
                values[method][replicate, coverage_index] = _weighted_prefix_accuracy(
                    correct[method], order[method], query_weight, coverage
                )
    return values, int(len(unique_formula))


def add_bootstrap_intervals(
    points: list[dict],
    bootstraps: dict[str, np.ndarray],
    coverages: tuple[float, ...],
) -> None:
    index = {
        (row["method"], float(row["target_coverage"])): row for row in points
    }
    for method, matrix in bootstraps.items():
        for coverage_index, coverage in enumerate(coverages):
            low, high = np.quantile(matrix[:, coverage_index], (0.025, 0.975))
            row = index[(method, coverage)]
            row["formula_cluster_ci_low"] = float(low)
            row["formula_cluster_ci_high"] = float(high)


def paired_functional_deltas(
    panel: str,
    point_rows: list[dict],
    bootstraps: dict[str, np.ndarray],
    coverages: tuple[float, ...],
    candidate: str = "calms_v132",
    comparator: str = "weighted_spectral_entropy",
) -> list[dict[str, float | int | str]]:
    if candidate not in bootstraps or comparator not in bootstraps:
        return []
    points = {
        (row["method"], float(row["target_coverage"])): row for row in point_rows
    }
    rows = []
    delta = bootstraps[candidate] - bootstraps[comparator]
    for coverage_index, coverage in enumerate(coverages):
        low, high = np.quantile(delta[:, coverage_index], (0.025, 0.975))
        point_delta = (
            float(points[(candidate, coverage)]["selective_recall_at_1"])
            - float(points[(comparator, coverage)]["selective_recall_at_1"])
        )
        rows.append({
            "panel": panel,
            "candidate": candidate,
            "comparator": comparator,
            "target_coverage": coverage,
            "candidate_minus_comparator": point_delta,
            "candidate_minus_comparator_pp": 100.0 * point_delta,
            "formula_cluster_ci_low": float(low),
            "formula_cluster_ci_high": float(high),
            "formula_cluster_ci_low_pp": 100.0 * float(low),
            "formula_cluster_ci_high_pp": 100.0 * float(high),
        })
    return rows


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")

    calms_paths = {
        "identity_disjoint": args.calms_identity,
        "formula_disjoint": args.calms_formula,
    }
    wse_paths = {
        "identity_disjoint": args.wse_identity,
        "formula_disjoint": args.wse_formula,
    }
    wse_exists = {panel: path.is_file() for panel, path in wse_paths.items()}
    if len(set(wse_exists.values())) != 1:
        raise RuntimeError("WSE comparator is incomplete: both panel tables are required")
    include_wse = all(wse_exists.values())

    all_points: list[dict] = []
    all_deltas: list[dict] = []
    all_global: list[dict] = []
    panel_reports = {}
    coverages = tuple(float(value) for value in FIXED_COVERAGES)
    for panel_index, panel in enumerate(PANELS):
        tables = {
            "calms_v132": load_query_table(calms_paths[panel], "calms_v132", panel)
        }
        if len(tables["calms_v132"]) != EXPECTED_PANEL_QUERIES[panel]:
            raise RuntimeError(
                f"{panel}: expected {EXPECTED_PANEL_QUERIES[panel]} frozen GNPS "
                f"queries, observed {len(tables['calms_v132'])}"
            )
        if include_wse:
            tables["weighted_spectral_entropy"] = load_query_table(
                wse_paths[panel], "weighted_spectral_entropy", panel
            )
            assert_same_axis(
                tables["calms_v132"], tables["weighted_spectral_entropy"], panel
            )

        points = []
        for method, table in tables.items():
            metric = global_metrics(table)
            all_global.append({"panel": panel, "method": method, **metric})
            points.extend(fixed_coverage_points(table, method, panel, coverages))
        bootstrap, clusters = formula_cluster_bootstrap(
            tables,
            coverages,
            args.bootstrap_resamples,
            args.seed + panel_index,
        )
        add_bootstrap_intervals(points, bootstrap, coverages)
        deltas = paired_functional_deltas(panel, points, bootstrap, coverages)
        all_points.extend(points)
        all_deltas.extend(deltas)
        panel_reports[panel] = {
            "queries": int(len(tables["calms_v132"])),
            "formula_clusters": clusters,
            "methods": list(tables),
        }

    args.output_dir.mkdir(parents=True)
    pd.DataFrame(all_global).to_csv(args.output_dir / "global_metrics.csv", index=False)
    pd.DataFrame(all_points).to_csv(
        args.output_dir / "selective_operating_points.csv", index=False
    )
    if all_deltas:
        pd.DataFrame(all_deltas).to_csv(
            args.output_dir / "paired_functional_deltas.csv", index=False
        )

    report = {
        "status": "CALMS_CLOSED_SET_SELECTIVE_RANKING_COMPLETE",
        "method": "frozen CALMS v1.3.2",
        "dataset": {
            "name": "GNPS Gold/Silver",
            "role": "consumed development; used in CALMS residual-amplitude selection",
            "truth_status": "consumed",
            "panels": panel_reports,
        },
        "confidence": "per-query CALMS Top1--Top2 candidate-score gap",
        "input_tables": {
            "calms": {panel: str(path) for panel, path in calms_paths.items()},
            "weighted_spectral_entropy": (
                {panel: str(path) for panel, path in wse_paths.items()}
                if include_wse
                else None
            ),
        },
        "fixed_coverages": list(coverages),
        "bootstrap": {
            "unit": "query formula cluster",
            "resamples": int(args.bootstrap_resamples),
            "seed": int(args.seed),
            "paired_delta_semantics": (
                "The same formula-cluster draw is used for both methods, while each "
                "method selects its own highest-confidence prefix at matched coverage. "
                "This is a paired functional delta, not a comparison on an identical "
                "accepted-query subset."
            ),
        },
        "closed_set_boundary": {
            "true_candidate_present_for_every_query": True,
            "open_set_or_no_match_evaluated": False,
            "false_match_acceptance_rate_evaluated": False,
            "calibrated_match_probability": False,
            "input_missing_abstention_evaluated": False,
            "interpretation": (
                "The curve measures accuracy among retained closed-set Top-1 guesses. "
                "It cannot support a no-match, open-set, FDR, or deployment-threshold claim."
            ),
        },
        "selection_boundary": (
            "GNPS is consumed development data. Any threshold chosen from these curves "
            "is a development operating point and requires independent calibration before "
            "deployment or confirmation."
        ),
        "artifacts": {
            "global_metrics": "global_metrics.csv",
            "selective_operating_points": "selective_operating_points.csv",
            "paired_functional_deltas": (
                "paired_functional_deltas.csv" if all_deltas else None
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
