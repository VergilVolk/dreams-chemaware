#!/usr/bin/env python
"""Read-only drift audit for sequential Full23 parents versus released v1.

The parent-query-relative trainer stores, on the exact outer-validation
candidate axis, both the fold-specific sequential parent and the released
Full23-v1 score.  This script reads that frozen ledger and reports how much
rebuilding the parent changed Enveda OOF decisions.  It never refits a model
and never writes into the run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnps_pair_score_cache import sha256_file
from train_full23_centered_residual_from_frozen_ledger import PARENT
from train_enveda_unified_candidate_integrator import PANELS


EXPECTED_STATUS = "FULL23_PARENT_QUERY_RELATIVE_MGDA_RESIDUAL_DEVELOPMENT_COMPLETE"
SEQUENTIAL_PARENT = "full23_v1_sequential_outer_parent"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument(
        "--prediction-file", type=Path,
        help="Defaults to RUN_OUTPUT/candidate_predictions_outer_oof.npz.",
    )
    parser.add_argument(
        "--run-report", type=Path, help="Defaults to RUN_OUTPUT/report.json.",
    )
    return parser.parse_args()


def _strict_rank_one(scores: np.ndarray, labels: np.ndarray) -> bool:
    positive = np.flatnonzero(labels > 0)
    if len(positive) != 1:
        raise RuntimeError("query does not have exactly one positive")
    return bool(np.sum(scores >= scores[int(positive[0])]) == 1)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.sum(left_centered ** 2) * np.sum(right_centered ** 2))
    )
    if denominator == 0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def _drift_summary(delta: np.ndarray) -> dict:
    values = np.asarray(delta, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "mean_signed": float(np.mean(values)),
        "mean_absolute": float(np.mean(absolute)),
        "root_mean_square": float(np.sqrt(np.mean(values ** 2))),
        "median_absolute": float(np.quantile(absolute, 0.50)),
        "q95_absolute": float(np.quantile(absolute, 0.95)),
        "q99_absolute": float(np.quantile(absolute, 0.99)),
        "maximum_absolute": float(np.max(absolute)),
    }


def _panel_report(body: np.lib.npyio.NpzFile, panel_name: str) -> dict:
    prefix = f"enveda__{panel_name}__"
    required = (
        "query_ptr", "labels", "query_keys", "formula", "near",
        SEQUENTIAL_PARENT, PARENT,
    )
    missing = [prefix + key for key in required if prefix + key not in body.files]
    if missing:
        raise RuntimeError(f"parent OOF ledger misses fields: {missing}")
    ptr = np.asarray(body[prefix + "query_ptr"], dtype=np.int64)
    labels = np.asarray(body[prefix + "labels"], dtype=np.int8)
    query_keys = np.asarray(body[prefix + "query_keys"]).astype("U")
    formulas = np.asarray(body[prefix + "formula"]).astype("U")
    sequential = np.asarray(body[prefix + SEQUENTIAL_PARENT], dtype=np.float64)
    released = np.asarray(body[prefix + PARENT], dtype=np.float64)
    if (
        ptr.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(labels)
        or np.any(np.diff(ptr) < 2) or len(query_keys) != len(ptr) - 1
        or len(formulas) != len(ptr) - 1
        or sequential.shape != labels.shape or released.shape != labels.shape
        or not np.all(np.isfinite(sequential)) or not np.all(np.isfinite(released))
    ):
        raise RuntimeError(f"invalid Enveda parent OOF axis: {panel_name}")

    sequential_ok = []
    released_ok = []
    deterministic_agreement = []
    top_set_agreement = []
    sequential_ties = []
    released_ties = []
    sequential_margin = []
    released_margin = []
    centered_delta = np.empty_like(sequential)
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        local_labels = labels[left:right]
        local_sequential = sequential[left:right]
        local_released = released[left:right]
        sequential_ok.append(_strict_rank_one(local_sequential, local_labels))
        released_ok.append(_strict_rank_one(local_released, local_labels))
        deterministic_agreement.append(
            int(np.argmax(local_sequential)) == int(np.argmax(local_released))
        )
        sequential_top = np.flatnonzero(local_sequential == np.max(local_sequential))
        released_top = np.flatnonzero(local_released == np.max(local_released))
        top_set_agreement.append(np.array_equal(sequential_top, released_top))
        sequential_ties.append(len(sequential_top) > 1)
        released_ties.append(len(released_top) > 1)
        sequential_margin.append(float(
            np.partition(local_sequential, -2)[-1]
            - np.partition(local_sequential, -2)[-2]
        ))
        released_margin.append(float(
            np.partition(local_released, -2)[-1]
            - np.partition(local_released, -2)[-2]
        ))
        centered_delta[left:right] = (
            local_sequential - np.mean(local_sequential)
            - local_released + np.mean(local_released)
        )

    sequential_ok = np.asarray(sequential_ok, dtype=bool)
    released_ok = np.asarray(released_ok, dtype=bool)
    margin_delta = np.asarray(sequential_margin) - np.asarray(released_margin)
    return {
        "queries": int(len(ptr) - 1),
        "candidate_rows": int(len(labels)),
        "unique_query_keys": int(len(np.unique(query_keys))),
        "formula_clusters": int(len(np.unique(formulas))),
        "sequential_parent_recall_at_1": float(np.mean(sequential_ok)),
        "released_full23_v1_recall_at_1": float(np.mean(released_ok)),
        "recall_at_1_delta": float(np.mean(sequential_ok) - np.mean(released_ok)),
        "corrected": int(np.sum(sequential_ok & ~released_ok)),
        "introduced": int(np.sum(~sequential_ok & released_ok)),
        "deterministic_top1_agreement": float(np.mean(deterministic_agreement)),
        "exact_top_tie_set_agreement": float(np.mean(top_set_agreement)),
        "sequential_parent_tied_top_queries": int(np.sum(sequential_ties)),
        "released_full23_v1_tied_top_queries": int(np.sum(released_ties)),
        "candidate_score_pearson": _pearson(sequential, released),
        "raw_candidate_score_drift": _drift_summary(sequential - released),
        "query_centered_candidate_score_drift": _drift_summary(centered_delta),
        "top1_top2_margin_drift": _drift_summary(margin_delta),
    }


def main() -> None:
    args = arguments()
    prediction_path = args.prediction_file or (
        args.run_output / "candidate_predictions_outer_oof.npz"
    )
    report_path = args.run_report or (args.run_output / "report.json")
    if not prediction_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("parent-qrel output is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != EXPECTED_STATUS:
        raise RuntimeError(f"unexpected parent-qrel status: {report.get('status')!r}")
    declared = report.get("artifacts", {}).get("predictions_sha256")
    observed = sha256_file(prediction_path)
    if declared != observed:
        raise RuntimeError("parent-qrel prediction hash disagrees with report")
    with np.load(prediction_path, allow_pickle=False) as body:
        panels = {name: _panel_report(body, name) for name in PANELS}
    output = {
        "status": "FULL23_SEQUENTIAL_PARENT_DRIFT_READ_ONLY_COMPLETE",
        "run_status": report["status"],
        "prediction_file": str(prediction_path),
        "prediction_sha256": observed,
        "comparison": f"{SEQUENTIAL_PARENT}_vs_{PARENT}",
        "scope": "Enveda outer-validation OOF panels only",
        "panels": panels,
        "interpretation_boundary": (
            "The sequential parent is refitted without each outer-validation formula, "
            "whereas released Full23 v1 is the all-Enveda development refit. Drift is "
            "therefore expected and must not be attributed to the qrel residual."
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
