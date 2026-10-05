"""Paired summary for direct candidate-boundary (E4-DEB) fine-tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from summarize_noise_final_e4a_causal_attribution import paired_summary


EXPECTED = {
    "clean_duplicate": ("clean_duplicate", False),
    "matched_random": ("matched_random", False),
    "candidate_boundary": ("paired_target", True),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument(
        "--nist-report", type=Path, default=None,
        help="Optional legacy NIST20 callback report; never a promotion gate.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_one(root: Path, label: str, arm: str, boundary: bool) -> tuple[Path, dict, pd.DataFrame]:
    matches = list((root / label).rglob("decision.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one completed {label} arm; found {len(matches)}")
    run = matches[0].parent
    report = json.loads(matches[0].read_text(encoding="utf-8"))
    config = report.get("configuration", {})
    contracts = report.get("contracts", {})
    if (
        report.get("formal") is not True
        or config.get("pmt_arm") != arm
        or bool(config.get("candidate_boundary_loss", False)) is not boundary
        or contracts.get("pmt_all_corrective_actions_exposed_before_recycling") is not True
        or contracts.get("pmt_harmful_target_weight_exact_zero") is not True
        or contracts.get("pmt_P_actions_used") is not False
        or config.get("pmt_manifest_corrective_query_scope") != "errors"
    ):
        raise RuntimeError(f"invalid E4-DEB arm contract: {label}")
    if boundary and (
        contracts.get("candidate_boundary_matrix_preserved") is not True
        or contracts.get("candidate_boundary_scalar_teacher_used") is not False
        or contracts.get("candidate_boundary_control_gradient_stopped") is not True
        or contracts.get("candidate_boundary_live_topk_refresh") is not True
        or contracts.get("candidate_boundary_formula_stratified_gradient_calibration") is not True
    ):
        raise RuntimeError("candidate-boundary treatment lacks a required contract")
    ledger = pd.read_csv(run / "held_per_query.csv.gz").sort_values(
        "query_index", kind="stable",
    ).reset_index(drop=True)
    return run, report, ledger


def metric_delta(treatment: dict, control: dict) -> dict[str, float]:
    return {
        key: float(treatment[key] - control[key])
        for key in treatment
        if isinstance(treatment[key], float) and key in control
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E4-DEB summary: {args.output_dir}")
    runs: dict[str, Path] = {}
    reports: dict[str, dict] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for label, (arm, boundary) in EXPECTED.items():
        runs[label], reports[label], ledgers[label] = load_one(
            args.arms_root, label, arm, boundary,
        )
    reference = ledgers["clean_duplicate"]
    invariant = (
        "query_index", "query_formula", "has_near", "baseline_rank",
        "initialization_rank", "baseline_top_molecule_local",
        "initialization_top_molecule_local", "baseline_full_margin",
        "initialization_full_margin",
    )
    for label, ledger in ledgers.items():
        for column in invariant:
            left, right = reference[column], ledger[column]
            if left.dtype.kind == "f":
                same = np.allclose(left, right, rtol=0, atol=2e-6)
            else:
                same = left.equals(right)
            if not same:
                raise RuntimeError(f"E4-DEB invariant {column} drifted in {label}")
    histories = {label: reports[label].get("history", []) for label in EXPECTED}
    lengths = {len(value) for value in histories.values()}
    if lengths != {4}:
        raise RuntimeError(f"E4-DEB histories are incomplete: {lengths}")
    for epoch in range(4):
        schedules = {
            histories[label][epoch].get("action_exposure_schedule_sha256")
            for label in EXPECTED
        }
        if len(schedules) != 1:
            raise RuntimeError(f"E4-DEB action schedule drifted at epoch {epoch + 1}")

    treatment = "candidate_boundary"
    comparisons = {
        f"{treatment}_vs_{control}": paired_summary(
            ledgers[control], ledgers[treatment], args.bootstrap_resamples,
            args.seed + 100 * index,
        )
        for index, control in enumerate(("clean_duplicate", "matched_random"), start=1)
    }
    complete_metrics = {
        label: reports[label]["held_clean"]["complete_candidate_metrics"]["student"]
        for label in EXPECTED
    }
    metric_deltas = {
        f"{treatment}_vs_{control}": metric_delta(
            complete_metrics[treatment], complete_metrics[control],
        )
        for control in ("clean_duplicate", "matched_random")
    }
    required_nonnegative = (
        "mrr", "recall2", "recall3", "recall5", "recall10", "recall20",
        "macro_query_auc", "micro_candidate_auc", "macro_query_auprc",
        "micro_candidate_auprc",
        "mean_positive_vs_best_negative_margin", "mean_top1_top2_gap",
        "selective_one_minus_aurc",
        "coverage_at_empirical_top1_error_1pct",
        "coverage_at_empirical_top1_error_5pct",
        "coverage_at_empirical_top1_error_10pct",
    )
    paired_gates = {}
    for control in ("clean_duplicate", "matched_random"):
        comparison = comparisons[f"{treatment}_vs_{control}"]
        deltas = metric_deltas[f"{treatment}_vs_{control}"]
        paired_gates[f"top1_formula_ci_positive_vs_{control}"] = bool(
            comparison["formula_cluster_top1_ci"]["ci_low"] > 0
        )
        paired_gates[f"corrected_gt_introduced_vs_{control}"] = bool(
            comparison["corrected"] > comparison["introduced"]
        )
        paired_gates[f"risk_net_positive_vs_{control}"] = bool(
            comparison["risk_net_lambda2"] > 0
        )
        paired_gates[f"near_nonnegative_vs_{control}"] = bool(
            comparison["delta_near_recall1"] >= 0
        )
        paired_gates[f"all_secondary_metrics_nonnegative_vs_{control}"] = bool(
            all(deltas[key] >= 0 for key in required_nonnegative)
        )
    nist = None
    if args.nist_report is not None:
        if not args.nist_report.is_file():
            raise FileNotFoundError(args.nist_report)
        nist = json.loads(args.nist_report.read_text(encoding="utf-8"))
    final_gates = paired_gates
    report = {
        "status": "noise_final_e4_deb_summary_complete",
        "formal": True,
        "scientific_question": (
            "Does a non-scalar, candidate-boundary target-minus-control objective add "
            "clean shared-embedding retrieval value beyond matched random noise and "
            "clean continuation from the same mature E4 initialization?"
        ),
        "arms": {
            label: {
                "run": str(runs[label]),
                "decision_sha256": sha256_file(runs[label] / "decision.json"),
                "checkpoint_sha256": sha256_file(runs[label] / "final_shared_encoder.pt"),
                "held": reports[label]["held_clean"],
            }
            for label in EXPECTED
        },
        "paired_top1_comparisons": comparisons,
        "complete_candidate_metrics": complete_metrics,
        "complete_candidate_metric_deltas": metric_deltas,
        "external_nist20_supplementary": nist,
        "gates": final_gates,
        "internal_candidate_graph_pass": bool(all(paired_gates.values())),
        "pass_to_multifold": bool(all(final_gates.values())),
        "decision": (
            "replicate candidate-boundary treatment across frozen folds and seeds"
            if all(final_gates.values()) else (
                "do not scale; locate the failed paired or secondary-metric gate"
            )
        ),
        "contracts": {
            "one_shared_clean_spectrum_encoder": True,
            "frozen_best_fold0_clean_duplicate_initialization": True,
            "same_action_membership_and_epoch_schedule": True,
            "candidate_edges_not_scalar_collapsed": True,
            "target_control_difference_preserved_per_candidate": True,
            "matched_control_not_trained_as_positive": True,
            "harmful_candidate_edges_zero_corrective_weight": True,
            "target_action_exposure_restricted_to_current_geometry_errors": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "claim_limit": (
            "One development formula fold and seed. The 3-5 pp action oracle is an "
            "upper bound, so this experiment tests transfer efficiency but cannot "
            "guarantee complete oracle transfer. Optional NIST20 callback results are "
            "supplementary and never substitute for or gate this task's frozen graph."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e4_deb_summary_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        combined = reference[[
            "query_index", "query_formula", "has_near", "baseline_rank",
            "initialization_rank",
        ]].copy()
        for label in EXPECTED:
            combined[f"{label}_rank"] = ledgers[label]["final_rank"].to_numpy(np.int16)
            combined[f"{label}_margin"] = ledgers[label]["final_full_margin"].to_numpy(np.float32)
            combined[f"{label}_top_molecule_local"] = ledgers[label][
                "final_top_molecule_local"
            ].to_numpy(np.int32)
        for control in ("clean_duplicate", "matched_random"):
            combined[f"candidate_boundary_corrected_vs_{control}"] = (
                (combined[f"{control}_rank"] != 1)
                & (combined["candidate_boundary_rank"] == 1)
            )
            combined[f"candidate_boundary_introduced_vs_{control}"] = (
                (combined[f"{control}_rank"] == 1)
                & (combined["candidate_boundary_rank"] != 1)
            )
            combined[f"candidate_boundary_switch_vs_{control}"] = (
                combined[f"candidate_boundary_top_molecule_local"]
                != combined[f"{control}_top_molecule_local"]
            )
        combined.to_csv(staging / "paired_per_query.csv.gz", index=False, compression="gzip")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
