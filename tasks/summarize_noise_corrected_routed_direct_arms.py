"""Paired comparison of routed-direct and equal-budget clean-control arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import CandidateGraph, sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--minimum-causal-queries", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def formula_cluster_ci(
    formulas: np.ndarray, values: np.ndarray, repeats: int, seed: int,
) -> dict[str, float]:
    if repeats < 1 or not len(values):
        raise ValueError("formula-cluster CI requires observations and resamples")
    grouped = pd.DataFrame({
        "formula": np.asarray(formulas, dtype=str),
        "value": np.asarray(values, dtype=float),
    }).groupby("formula", sort=True).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=float)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), len(grouped))
        boot[index] = 100.0 * sums[take].sum() / counts[take].sum()
    return {
        "delta_pp": float(100.0 * np.mean(values)),
        "ci_low_pp": float(np.quantile(boot, 0.025)),
        "ci_high_pp": float(np.quantile(boot, 0.975)),
        "formula_clusters": int(len(grouped)),
    }


def summarize(
    routed_dir: Path, control_dir: Path, graph_dir: Path, minimum: int,
    bootstrap_resamples: int, seed: int,
) -> dict[str, object]:
    routed_decision = routed_dir / "decision.json"
    control_decision = control_dir / "decision.json"
    routed_table = routed_dir / "development_per_query.csv.gz"
    control_table = control_dir / "development_per_query.csv.gz"
    for path in (routed_decision, control_decision, routed_table, control_table):
        if not path.is_file():
            raise FileNotFoundError(path)
    routed = json.loads(routed_decision.read_text(encoding="utf-8"))
    control = json.loads(control_decision.read_text(encoding="utf-8"))
    if (
        routed.get("status") != "noise_corrected_routed_direct_complete"
        or control.get("status") != "noise_corrected_routed_direct_complete"
        or routed.get("arm") != "routed_direct"
        or control.get("arm") != "clean_control"
        or routed.get("corrective_objective_mode") != control.get("corrective_objective_mode")
    ):
        raise RuntimeError("routed/control arm status differs")
    for key in (
        "candidate_graph_sha256", "routed_ledger_report_sha256",
        "training_actions_sha256", "action_spectra_sha256",
        "initial_student_checkpoint_sha256", "official_checkpoint_sha256",
    ):
        if routed["provenance"].get(key) != control["provenance"].get(key):
            raise RuntimeError(f"routed/control provenance differs: {key}")
    graph_path = graph_dir / "candidate_graph.npz"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if sha256_file(graph_path) != routed["provenance"]["candidate_graph_sha256"]:
        raise RuntimeError("paired-control graph differs from training graph")
    graph = CandidateGraph(graph_path)
    calibration_keys = (
        "microbatches", "query_observations", "effective_corrective_scale",
        "effective_global_gradient_scale", "effective_margin_transfer_multiplier",
    )
    for key in calibration_keys:
        if not np.isclose(
            float(routed["gradient_calibration"][key]),
            float(control["gradient_calibration"][key]), rtol=0, atol=1e-12,
        ):
            raise RuntimeError(f"routed/control calibration differs: {key}")
    left = pd.read_csv(routed_table)
    right = pd.read_csv(control_table)
    keys = ["panel", "query_index"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise RuntimeError("per-query arm table contains duplicates")
    paired = left.merge(right, on=keys, suffixes=("_routed", "_control"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("routed/control per-query panels are not identical")
    query_index = paired.query_index.to_numpy(np.int64)
    if np.any((query_index < 0) | (query_index >= graph.n_queries)):
        raise RuntimeError("paired-control query index is outside the frozen graph")
    paired["query_formula"] = graph.query_formula[query_index]
    paired["near"] = graph.query_has_near[query_index]
    panel_reports: dict[str, object] = {}
    for panel, block in paired.groupby("panel", sort=True):
        routed_correct = block.rank_final_routed.eq(1)
        control_correct = block.rank_final_control.eq(1)
        recall_delta = routed_correct.astype(float) - control_correct.astype(float)
        formula_ci = formula_cluster_ci(
            block.query_formula.to_numpy(str), recall_delta.to_numpy(float),
            bootstrap_resamples, seed + len(panel_reports),
        )
        near = block.loc[block.near].copy()
        near_formula_ci = (
            formula_cluster_ci(
                near.query_formula.to_numpy(str),
                (near.rank_final_routed.eq(1).astype(float)
                 - near.rank_final_control.eq(1).astype(float)).to_numpy(float),
                bootstrap_resamples, seed + 10_000 + len(panel_reports),
            )
            if len(near) else None
        )
        panel_reports[str(panel)] = {
            "queries": int(len(block)),
            "routed_minus_control_recall1_pp": float(100 * np.mean(recall_delta)),
            "routed_minus_control_mrr_pp": float(100 * np.mean(
                1.0 / block.rank_final_routed - 1.0 / block.rank_final_control
            )),
            "routed_minus_control_mean_margin": float(np.mean(
                block.margin_final_routed - block.margin_final_control
            )),
            "rank_wins": int(np.sum(block.rank_final_routed < block.rank_final_control)),
            "rank_losses": int(np.sum(block.rank_final_routed > block.rank_final_control)),
            "routed_only_top1": int(np.sum(routed_correct & ~control_correct)),
            "control_only_top1": int(np.sum(control_correct & ~routed_correct)),
            "formula_cluster_delta_recall1": formula_ci,
            "near_formula_cluster_delta_recall1": near_formula_ci,
        }
    primary = panel_reports.get("train_corrective", {"queries": 0})
    enough = int(primary["queries"]) >= minimum
    causal_positive = bool(
        enough
        and float(primary["routed_minus_control_recall1_pp"]) > 0
        and int(primary["routed_only_top1"]) > int(primary["control_only_top1"])
    )
    return {
        "status": "noise_corrected_routed_direct_arm_summary_complete",
        "corrective_objective_mode": routed["corrective_objective_mode"],
        "panel_comparisons": panel_reports,
        "signal_transmission": routed["signal_transmission"],
        "minimum_causal_queries": minimum,
        "causal_query_gate_sufficient": enough,
        "routed_beats_equal_budget_control": causal_positive,
        "paired_formula_ci_strict_positive": bool(
            enough
            and float(primary["formula_cluster_delta_recall1"]["ci_low_pp"]) > 0
        ),
        "pass_to_larger_local_gate": causal_positive,
        "claim_limit": (
            "Insufficient small development panel; wiring evidence only."
            if not enough else
            "Paired development arm comparison; not formal held performance."
        ),
        "provenance": {
            "routed_decision_sha256": sha256_file(routed_decision),
            "control_decision_sha256": sha256_file(control_decision),
            "routed_table_sha256": sha256_file(routed_table),
            "control_table_sha256": sha256_file(control_table),
            "candidate_graph_sha256": sha256_file(graph_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "paired_table": paired,
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    report = summarize(
        args.routed_dir, args.control_dir, args.graph_dir,
        args.minimum_causal_queries, args.bootstrap_resamples, args.seed,
    )
    paired = report.pop("paired_table")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        paired.to_csv(staging / "paired_queries.csv.gz", index=False, compression="gzip")
        report["provenance"]["paired_queries_sha256"] = sha256_file(staging / "paired_queries.csv.gz")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
