"""Causal summary for frozen qualified-action effect transfer into clean DreaMS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemaware_direct_action_core import formula_bootstrap


ARMS = ("clean_duplicate", "correct_synthetic", "candidate_swapped", "peak_permuted")
RANK_METRICS = ("recall1", "recall5", "recall10", "recall20", "recall50")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("primary", "full"), required=True)
    parser.add_argument("--seed", type=int, default=20260935)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def load_arm(root: Path, arm: str) -> tuple[dict, pd.DataFrame, dict]:
    directory = root / arm
    required = (
        directory / "COMPLETE.json", directory / "report.json",
        directory / "broad_inner_per_query.csv.gz", directory / "final_shared_encoder.pt",
        directory / "action_target_audit.json",
    )
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"arm {arm} is incomplete")
    complete = json.loads(required[0].read_text(encoding="utf-8"))
    report = json.loads(required[1].read_text(encoding="utf-8"))
    target = json.loads(required[4].read_text(encoding="utf-8"))
    rows = pd.read_csv(required[2]).sort_values("query_index").reset_index(drop=True)
    expected_columns = {
        "query_index", "formula", "identity", "initial_rank", "final_rank",
        "initial_margin", "final_margin", "candidate_count",
    }
    if (
        complete.get("status") != "CHEMAWARE_DIRECT_ARM_COMPLETE"
        or complete.get("arm") != arm
        or complete.get("report_status") != report.get("status")
        or report.get("status") not in ("PASS", "FAIL")
        or report.get("optimization", {}).get("training_objective")
        != "direct_action_delta_transfer"
        or rows.empty or not expected_columns.issubset(rows.columns)
    ):
        raise RuntimeError(f"arm {arm} violates the action-delta transfer contract")
    contract = report.get("preflight", {}).get("contracts", {})
    if not (
        contract.get("qualified_action_effect_target") is True
        and contract.get("trainable_action_view_forward") is False
        and contract.get("chemical_reference_gradient") is False
        and contract.get("candidate_centred_action_effect") is True
        and contract.get("formula_equal_action_mass") is True
        and contract.get("unsupported_action_role_weight_zero") is True
        and contract.get("outer_fold_evaluated") is False
    ):
        raise RuntimeError(f"arm {arm} has an invalid chemical-transfer contract")
    provenance = report.get("preflight", {}).get("provenance", {})
    required_hashes = (
        "graph_sha256", "evaluation_manifest_sha256", "official_checkpoint_sha256",
        "architecture_checkpoint_sha256", "teacher_report_sha256",
        "teacher_predictions_sha256",
    )
    if any(
        not isinstance(provenance.get(key), str)
        or len(provenance[key]) != 64
        or not set(provenance[key]).issubset(set("0123456789abcdef"))
        for key in required_hashes
    ):
        raise RuntimeError(f"arm {arm} lacks complete immutable data provenance")
    if (
        target.get("rank_replay_mismatches") != 0
        or float(target.get("maximum_margin_replay_error", np.inf)) > 5e-4
        or target.get("action_view_trainable_forward") is not False
        or target.get("chemical_reference_gradient") is not False
    ):
        raise RuntimeError(f"arm {arm} failed frozen action-target integrity")
    if arm == "clean_duplicate" and int(target.get("zero_target_actions", -1)) != int(
        target.get("qualified_actions", -2)
    ):
        raise RuntimeError("clean duplicate did not produce exact zero chemical targets")
    if arm == "clean_duplicate" and (
        int(target.get("frozen_action_corrected", -1)) != 0
        or int(target.get("frozen_action_introduced", -1)) != 0
        or abs(float(target.get("frozen_action_mean_margin_gain", np.inf))) > 1e-7
    ):
        raise RuntimeError("clean duplicate changed the frozen candidate decision")
    if arm == "correct_synthetic" and (
        int(target.get("frozen_action_corrected", 0)) < 1
        or float(target.get("frozen_action_mean_margin_gain", -np.inf)) <= 0
    ):
        raise RuntimeError("correct arm lacks a positive frozen qualified-action effect")
    optimizer_signal = report.get("chemical_optimizer_signal", {})
    if not (
        optimizer_signal.get("gate_passed") is True
        and optimizer_signal.get("all_losses_nonincreasing") is True
        and optimizer_signal.get("all_group_geometries_valid") is True
        and int(optimizer_signal.get("observations", 0))
        == int(optimizer_signal.get("expected_observations", -1))
        and int(optimizer_signal.get("observations", 0)) > 0
    ):
        raise RuntimeError(f"arm {arm} failed its realized chemical-optimizer audit")
    fit = report.get("chemical_target_fit", {})
    initial_fit = fit.get("initial", {})
    final_fit = fit.get("final", {})
    fit_values = (
        initial_fit.get("formula_weighted_huber"),
        final_fit.get("formula_weighted_huber"),
        fit.get("formula_weighted_huber_reduction"),
    )
    if any(value is None or not np.isfinite(float(value)) for value in fit_values):
        raise RuntimeError(f"arm {arm} lacks a finite complete-panel target-fit audit")
    if arm != "clean_duplicate" and float(fit_values[2]) <= 0:
        raise RuntimeError(f"arm {arm} did not reduce its complete chemical target loss")
    numeric = rows[[
        "initial_rank", "final_rank", "initial_margin", "final_margin", "candidate_count",
    ]].to_numpy()
    if not np.all(np.isfinite(numeric)) or np.any(rows["candidate_count"] < 2):
        raise RuntimeError(f"arm {arm} contains invalid per-query outcomes")
    return report, rows, target


def row_metrics(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    old = rows["initial_rank"].to_numpy(dtype=np.int64)
    new = rows["final_rank"].to_numpy(dtype=np.int64)
    count = rows["candidate_count"].to_numpy(dtype=np.int64)
    output = {
        f"delta_{name}": (new <= int(name.removeprefix("recall"))).astype(float)
        - (old <= int(name.removeprefix("recall"))).astype(float)
        for name in RANK_METRICS
    }
    output["delta_mrr"] = 1.0 / new - 1.0 / old
    output["delta_macro_auc"] = (old - new) / (count - 1)
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_draws < 10_000:
        raise ValueError("formula-cluster causal summary requires 10,000 draws")
    required_arms = ARMS[:2] if args.stage == "primary" else ARMS
    loaded = {arm: load_arm(args.root, arm) for arm in required_arms}
    identity = loaded[required_arms[0]][1][["query_index", "formula", "identity"]]
    for arm in required_arms[1:]:
        if not identity.equals(loaded[arm][1][["query_index", "formula", "identity"]]):
            raise RuntimeError("matched arms do not evaluate identical held queries")
    fingerprints = {}
    metrics = {}
    arm_summary = {}
    for arm, (report, rows, target) in loaded.items():
        optimization = report["optimization"]
        fingerprints[arm] = {
            key: optimization.get(key) for key in (
                "training_objective", "seed", "fold_seed", "inner_fold", "outer_fold",
                "epochs", "batch_queries",
                "max_action_identities", "max_safety_identities",
                "max_eval_identities", "unfreeze_blocks",
                "backbone_lr", "head_lr", "weight_decay", "temperature",
                "lambda_clean_rank", "lambda_action_rank", "safety_stream_weight",
                "lambda_margin_floor", "lambda_preserve", "margin_floor_slack", "grad_clip",
                "action_delta_alpha", "action_delta_huber",
                "base_chemical_operator_split", "separate_optimizer_moments",
                "chemical_weight_decay",
            )
        }
        fingerprints[arm]["action_bank"] = {
            key: report.get("preflight", {}).get("action_bank", {}).get(key)
            for key in (
                "bank_sha256", "selected_setting", "mode", "strength", "top_k",
                "action_generation_seed",
                "eligible_training_actions", "corrective_rank_actions",
                "corrective_margin_actions", "margin_role_dose_calibration",
                "formula_equal_training_mass",
            )
        }
        fingerprints[arm]["data_provenance"] = {
            key: report.get("preflight", {}).get("provenance", {}).get(key)
            for key in (
                "graph_sha256", "evaluation_manifest_sha256",
                "official_checkpoint_sha256", "architecture_checkpoint_sha256",
                "teacher_report_sha256",
                "teacher_predictions_sha256",
            )
        }
        fingerprints[arm]["qualified_action_targets"] = target.get("qualified_actions")
        metrics[arm] = row_metrics(rows)
        arm_summary[arm] = {
            "status": report.get("status"),
            "held_metrics": report["final"]["broad_inner"],
            "formula_bootstrap_recall1": formula_bootstrap(
                metrics[arm]["delta_recall1"], identity["formula"].astype(str).to_numpy(),
                args.seed + ARMS.index(arm), args.bootstrap_draws,
            ),
            "mean_clip_fraction": float(
                report.get("gradient_clipping", {}).get("maximum_stream_fraction", np.inf)
            ),
            "chemical_target_fit": report["chemical_target_fit"],
            "report_sha256": sha256_file(args.root / arm / "report.json"),
        }
    if len({json.dumps(value, sort_keys=True) for value in fingerprints.values()}) != 1:
        raise RuntimeError("action-delta arms do not share an identical optimization schedule")

    controls = ("clean_duplicate",) if args.stage == "primary" else (
        "clean_duplicate", "candidate_swapped", "peak_permuted",
    )
    formula = identity["formula"].astype(str).to_numpy()
    contrasts = {}
    for index, control in enumerate(controls):
        metric_delta = {
            name: float(np.mean(metrics["correct_synthetic"][name] - metrics[control][name]))
            for name in metrics["correct_synthetic"]
        }
        metric_delta["delta_micro_auc"] = (
            loaded["correct_synthetic"][0]["final"]["broad_inner"]["delta_micro_auc"]
            - loaded[control][0]["final"]["broad_inner"]["delta_micro_auc"]
        )
        contrasts[control] = {
            "metric_delta": metric_delta,
            "formula_bootstrap_recall1": formula_bootstrap(
                metrics["correct_synthetic"]["delta_recall1"]
                - metrics[control]["delta_recall1"],
                formula, args.seed + 101 + index, args.bootstrap_draws,
            ),
        }
    correct = arm_summary["correct_synthetic"]["held_metrics"]
    gates = {
        "correct_absolute_recall1_formula_ci_positive": (
            arm_summary["correct_synthetic"]["formula_bootstrap_recall1"]
            ["formula_cluster_bootstrap_95ci"][0] > 0
        ),
        **{
            f"correct_minus_{control}_recall1_formula_ci_positive":
            value["formula_bootstrap_recall1"]["formula_cluster_bootstrap_95ci"][0] > 0
            for control, value in contrasts.items()
        },
        **{
            f"correct_minus_{control}_{metric}_nonnegative":
            value["metric_delta"][f"delta_{metric}"] >= 0
            for control, value in contrasts.items()
            for metric in (
                "recall5", "recall10", "recall20", "recall50",
                "mrr", "macro_auc", "micro_auc",
            )
        },
        "correct_mrr_positive": correct["delta_mrr"] > 0,
        "correct_macro_auc_positive": correct["delta_macro_auc"] > 0,
        "correct_micro_auc_positive": correct["delta_micro_auc"] > 0,
        **{
            f"correct_recall{k}_nonnegative": correct[f"delta_recall{k}"] >= 0
            for k in (5, 10, 20, 50)
        },
        "correct_preservation": correct["preservation_mean"] >= 0.995,
        "correct_clipping_not_saturated": arm_summary["correct_synthetic"]["mean_clip_fraction"] < 0.9,
    }
    passed = bool(all(gates.values()))
    status = (
        "CHEMAWARE_ACTION_DELTA_STAGE1_PASS" if args.stage == "primary" and passed
        else "CHEMAWARE_ACTION_DELTA_STAGE1_FAIL" if args.stage == "primary"
        else "CHEMAWARE_ACTION_DELTA_CAUSAL_PASS" if passed
        else "CHEMAWARE_ACTION_DELTA_CAUSAL_FAIL"
    )
    report = {
        "status": status,
        "stage": args.stage,
        "pass_to_matched_controls": bool(args.stage == "primary" and passed),
        "pass_to_additional_seed": bool(args.stage == "full" and passed),
        "release_eligible": False,
        "goal_3pp_absolute_recall1_met": correct["delta_recall1"] >= 0.03,
        "goal_4pp_absolute_recall1_met": correct["delta_recall1"] >= 0.04,
        "goal_5pp_absolute_recall1_met": correct["delta_recall1"] >= 0.05,
        "action_source": "frozen bank-selected observed-peak conflict attenuation",
        "injection": "frozen official action-score delta inherited by live clean query against frozen references",
        "arms": arm_summary,
        "correct_vs_controls": contrasts,
        "gates": gates,
        "optimization_fingerprints": fingerprints,
        "scope": {
            "formula_clustered": True,
            "outer_fold_evaluated": False,
            "action_setting_rediscovered": False,
            "action_view_forwarded_through_trainable_encoder": False,
            "teacher_derived_counterfactual_effect": True,
            "single_seed_development_only": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
