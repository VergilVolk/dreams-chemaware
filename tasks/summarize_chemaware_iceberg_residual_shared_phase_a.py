"""Paired causal summary for the four ICEBERG residual shared-encoder arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_direct_action_core import formula_bootstrap  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402


ARMS = ("clean_duplicate", "correct_alpha050", "structure_alpha050", "peak_alpha050")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def pairwise_metrics(
    treatment: dict[str, np.ndarray],
    comparator: dict[str, np.ndarray],
    seed: int,
    draws: int,
) -> dict:
    treatment_rank = treatment["new_rank"].astype(np.int64)
    comparator_rank = comparator["new_rank"].astype(np.int64)
    count = treatment["candidate_count"].astype(np.int64)
    formula = treatment["formula"].astype(str)
    denominator = np.maximum(count - 1, 1)
    delta_hit1 = (treatment_rank == 1).astype(float) - (
        comparator_rank == 1
    ).astype(float)
    result = {
        "delta_recall1": float(np.mean(delta_hit1)),
        "delta_mrr": float(np.mean(1 / treatment_rank - 1 / comparator_rank)),
        "delta_macro_auc": float(np.mean((comparator_rank - treatment_rank) / denominator)),
        "delta_micro_auc": float(np.sum(comparator_rank - treatment_rank) / np.sum(denominator)),
        "treatment_wins_top1": int(np.sum((treatment_rank == 1) & (comparator_rank != 1))),
        "comparator_wins_top1": int(np.sum((treatment_rank != 1) & (comparator_rank == 1))),
        "formula_bootstrap_delta_recall1": formula_bootstrap(
            delta_hit1, formula, seed, draws
        ),
    }
    for k in (5, 10, 20, 50):
        result[f"delta_recall{k}"] = float(
            np.mean(treatment_rank <= k) - np.mean(comparator_rank <= k)
        )
    return result


def main() -> None:
    args = arguments()
    if args.bootstrap_draws < 10_000:
        raise ValueError("bootstrap draws were weakened below 10,000")
    output = args.root / "phase_a_summary.json"
    if output.exists():
        raise FileExistsError(output)
    payload = {}
    reports = {}
    for arm in ARMS:
        directory = args.root / arm
        report_path = directory / "report.json"
        per_query_path = directory / "inner_per_query.npz"
        checkpoint_path = directory / "final_shared_encoder.pt"
        if not all(path.is_file() for path in (report_path, per_query_path, checkpoint_path)):
            raise FileNotFoundError(f"incomplete matched arm: {arm}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_ARM_COMPLETE"
            or report.get("causal_chemistry_status") != "NOT_EVALUATED_SINGLE_ARM"
            or report.get("release_eligible") is not False
            or report.get("fixed_final_epoch_no_arm_specific_selection") is not True
        ):
            raise RuntimeError(f"arm report violates the Phase-A contract: {arm}")
        with np.load(per_query_path, allow_pickle=True) as loaded:
            payload[arm] = {key: np.asarray(loaded[key]) for key in loaded.files}
        reports[arm] = report

    reference = payload[ARMS[0]]
    for arm in ARMS[1:]:
        for key in ("query", "formula", "old_rank", "old_margin", "candidate_count"):
            if not np.array_equal(reference[key], payload[arm][key]):
                raise RuntimeError(f"matched arm changed the evaluation cohort: {arm}/{key}")
    provenance = reports[ARMS[0]]["preflight"]["provenance"]
    for arm in ARMS[1:]:
        if reports[arm]["preflight"]["provenance"] != provenance:
            raise RuntimeError(f"matched arm provenance differs: {arm}")

    correct = payload["correct_alpha050"]
    comparator_names = ("clean_duplicate", "structure_alpha050", "peak_alpha050")
    comparisons = {
        name: pairwise_metrics(
            correct, payload[name], args.seed + 101 * index, args.bootstrap_draws
        )
        for index, name in enumerate(comparator_names)
    }
    correct_report = reports["correct_alpha050"]
    correct_absolute = correct_report["final_inner"]
    all_ci_positive = all(
        value["formula_bootstrap_delta_recall1"]["formula_cluster_bootstrap_95ci"][0] > 0
        for value in comparisons.values()
    )
    all_primary_metrics_positive = all(
        correct_absolute[key] > 0
        for key in ("delta_recall1", "delta_mrr", "delta_macro_auc", "delta_micro_auc")
    )
    all_recall_k_nonnegative = all(
        correct_absolute[f"delta_recall{k}"] >= 0 for k in (5, 10, 20, 50)
    )
    causal_pass = bool(
        all(value["delta_recall1"] > 0 for value in comparisons.values())
        and all_ci_positive
        and all_primary_metrics_positive
        and all_recall_k_nonnegative
        and correct_report["preservation"] >= 0.995
        and correct_report["mean_clip_fraction"] < 0.9
    )
    summary = {
        "status": (
            "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_PHASE_A_DEVELOPMENT_CAUSAL_PASS"
            if causal_pass
            else "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_PHASE_A_DEVELOPMENT_CAUSAL_FAIL"
        ),
        "development_causal_chemistry_pass": causal_pass,
        "goal_3pp_absolute_recall1_met": bool(correct_absolute["delta_recall1"] >= 0.03),
        "goal_4pp_absolute_recall1_met": bool(correct_absolute["delta_recall1"] >= 0.04),
        "method_held_development_inner_fold3_consumed": True,
        "globally_untouched_confirmation_claimed": False,
        "reserve_fold4_consumed": False,
        "release_eligible": False,
        "correct_absolute_vs_official": correct_absolute,
        "correct_vs_matched_comparators": comparisons,
        "arm_summaries": {
            arm: {
                "prior_arm": reports[arm]["arm"],
                "alpha": reports[arm]["alpha"],
                "final_inner": reports[arm]["final_inner"],
                "formula_bootstrap_vs_official": reports[arm]["formula_bootstrap"],
                "preservation": reports[arm]["preservation"],
                "mean_clip_fraction": reports[arm]["mean_clip_fraction"],
                "checkpoint_sha256": sha256_file(args.root / arm / "final_shared_encoder.pt"),
            }
            for arm in ARMS
        },
        "decision_contract": {
            "correct_recall1_strictly_above_clean_and_both_chemical_controls": True,
            "each_paired_formula_cluster_ci_lower_bound_above_zero": True,
            "correct_absolute_recall1_mrr_macro_auc_micro_auc_positive": True,
            "correct_absolute_recall5_10_20_50_nonnegative": True,
            "mean_embedding_preservation_at_least_0.995": True,
            "gradient_clipping_not_saturated": True,
            "single_development_fold_never_release_eligible": True,
        },
        "provenance": provenance,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
