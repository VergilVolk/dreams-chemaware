"""Summarize the paired E4-A clean/random/targeted attribution experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import json_dump, sha256_file
from train_noise_final_r2_shared_encoder import formula_bootstrap_delta, formula_bootstrap_mean


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("clean_duplicate", "matched_random", "targeted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution",
    )
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution_summary",
    )
    return parser.parse_args()


def locate_run(root: Path, suffix: str, arm: str, seed: int, fold: int) -> Path:
    matches = list(root.glob(
        f"*_{suffix}_causal_{arm}/seed_{seed}/fold_{fold}"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {arm} run, found {len(matches)}: {matches}")
    return matches[0]


def paired_summary(
    reference: pd.DataFrame, treatment: pd.DataFrame, resamples: int, seed: int,
) -> dict[str, object]:
    ref_rank = reference["final_rank"].to_numpy(np.int16)
    new_rank = treatment["final_rank"].to_numpy(np.int16)
    formulas = treatment["query_formula"].astype(str).to_numpy()
    near = treatment["has_near"].astype(bool).to_numpy()
    ref_correct = ref_rank == 1
    new_correct = new_rank == 1
    ref_top = reference["final_top_molecule_local"].to_numpy(np.int32)
    new_top = treatment["final_top_molecule_local"].to_numpy(np.int32)
    margin_delta = (
        treatment["final_full_margin"].to_numpy(np.float64)
        - reference["final_full_margin"].to_numpy(np.float64)
    )
    mrr_delta = 1.0 / new_rank.astype(np.float64) - 1.0 / ref_rank.astype(np.float64)
    return {
        "queries": int(len(new_rank)),
        "delta_recall1": float(np.mean(new_correct) - np.mean(ref_correct)),
        "corrected": int(np.sum(~ref_correct & new_correct)),
        "introduced": int(np.sum(ref_correct & ~new_correct)),
        "risk_net_lambda2": int(
            np.sum(~ref_correct & new_correct) - 2 * np.sum(ref_correct & ~new_correct)
        ),
        "delta_mrr": float(np.mean(mrr_delta)),
        "delta_near_recall1": float(
            np.mean(new_correct[near]) - np.mean(ref_correct[near])
        ) if np.any(near) else float("nan"),
        "formula_cluster_top1_ci": formula_bootstrap_delta(
            ref_rank, new_rank, formulas, resamples, seed,
        ),
        "formula_cluster_mrr_ci": formula_bootstrap_mean(
            mrr_delta, formulas, resamples, seed + 1,
        ),
        "mean_full_margin_delta": float(np.mean(margin_delta)),
        "formula_cluster_margin_ci": formula_bootstrap_mean(
            margin_delta, formulas, resamples, seed + 2,
        ),
        "top_molecule_changed": int(np.sum(ref_top != new_top)),
        "wrong_to_different_wrong": int(np.sum(
            ~ref_correct & ~new_correct & (ref_top != new_top)
        )),
        "protected_correct": int(np.sum(ref_correct & new_correct)),
        "persistent_wrong": int(np.sum(~ref_correct & ~new_correct)),
    }


def main() -> None:
    args = arguments()
    if args.outer_fold != 0:
        raise ValueError("the first causal attribution gate is frozen to development fold 0")
    output = args.output_dir / args.run_suffix / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite causal attribution summary: {output}")

    runs: dict[str, Path] = {}
    decisions: dict[str, dict] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        run = locate_run(args.run_root, args.run_suffix, arm, args.seed, args.outer_fold)
        decision_path = run / "decision.json"
        ledger_path = run / "held_per_query.csv.gz"
        checkpoint_path = run / "final_shared_encoder.pt"
        if not decision_path.is_file() or not ledger_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"incomplete causal arm: {run}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        configuration = decision.get("configuration", {})
        contracts = decision.get("contracts", {})
        if (
            decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
            or not decision.get("formal")
            or configuration.get("causal_arm") != arm
            or contracts.get("causal_attribution_arm") != arm
            or contracts.get("matched_control_selection_uses_outcome") is not False
            or contracts.get("P2b") != "forbidden"
            or contracts.get("P3_consumed") is not False
        ):
            raise RuntimeError(f"invalid causal arm contract: {arm}")
        ledger = pd.read_csv(ledger_path).sort_values("query_index", kind="stable").reset_index(drop=True)
        required_columns = {
            "query_index", "query_formula", "has_near", "baseline_rank",
            "initialization_rank", "final_rank", "baseline_top_molecule_local",
            "initialization_top_molecule_local", "final_top_molecule_local",
            "baseline_full_margin", "initialization_full_margin", "final_full_margin",
        }
        if required_columns - set(ledger.columns):
            raise RuntimeError(f"{arm} ledger lacks columns: {sorted(required_columns - set(ledger.columns))}")
        runs[arm] = run
        decisions[arm] = decision
        ledgers[arm] = ledger

    invariant_configuration = [
        "graph", "r0_dir", "data", "embedding_cache", "official_checkpoint",
        "architecture_checkpoint", "policy", "action_selection", "action_scope",
        "outer_fold", "formula_fold_seed", "seed", "epochs", "batch_actions",
        "views_per_identity", "error_views_per_identity", "positive_spectra",
        "negative_molecules", "unfreeze_blocks", "head_lr", "backbone_lr",
        "weight_decay", "rank_margin", "temperature", "lambda_clean_rank",
        "lambda_aug_rank", "lambda_consistency", "direct_transfer_mode",
        "rank_reference_mode", "lambda_margin_floor", "lambda_preserve",
        "margin_floor_slack", "safety_ratio", "safety_stream_weight", "grad_clip",
        "amp",
    ]
    reference_config = decisions["clean_duplicate"]["configuration"]
    for arm in ARMS[1:]:
        observed = decisions[arm]["configuration"]
        drift = {
            key: (reference_config.get(key), observed.get(key))
            for key in invariant_configuration
            if reference_config.get(key) != observed.get(key)
        }
        if drift:
            raise RuntimeError(f"non-arm configuration drift for {arm}: {drift}")

    invariant_provenance = (
        "r0_report_sha256", "r0_actions_sha256", "graph_sha256",
        "official_checkpoint_sha256", "script_sha256",
    )
    reference_provenance = decisions["clean_duplicate"]["provenance"]
    for arm in ARMS[1:]:
        observed = decisions[arm]["provenance"]
        if any(observed.get(key) != reference_provenance.get(key) for key in invariant_provenance):
            raise RuntimeError(f"input provenance drift for {arm}")

    reference_history = decisions["clean_duplicate"].get("history", [])
    if len(reference_history) != 4:
        raise RuntimeError("clean-duplicate arm does not contain four epoch records")
    schedule_keys = (
        "action_sampling_schedule_sha256", "safety_sampling_schedule_sha256",
    )
    for arm in ARMS[1:]:
        history = decisions[arm].get("history", [])
        if len(history) != len(reference_history):
            raise RuntimeError(f"epoch count drift for {arm}")
        for expected_epoch, observed_epoch in zip(reference_history, history):
            if any(expected_epoch.get(key) != observed_epoch.get(key) for key in schedule_keys):
                raise RuntimeError(f"batch schedule drift for {arm}")

    invariant_columns = [
        "query_index", "query_formula", "has_near", "baseline_rank",
        "initialization_rank", "baseline_top_molecule_local",
        "initialization_top_molecule_local",
    ]
    reference_ledger = ledgers["clean_duplicate"]
    for arm in ARMS[1:]:
        if not ledgers[arm][invariant_columns].equals(reference_ledger[invariant_columns]):
            raise RuntimeError(f"held-query or initialization drift for {arm}")
        for column in ("baseline_full_margin", "initialization_full_margin"):
            if not np.allclose(
                ledgers[arm][column].to_numpy(np.float64),
                reference_ledger[column].to_numpy(np.float64),
                rtol=0.0, atol=2e-6,
            ):
                raise RuntimeError(f"{column} drift for {arm}")

    comparisons = {
        "targeted_vs_matched_random": paired_summary(
            ledgers["matched_random"], ledgers["targeted"],
            args.bootstrap_resamples, args.seed + 100,
        ),
        "targeted_vs_clean_duplicate": paired_summary(
            ledgers["clean_duplicate"], ledgers["targeted"],
            args.bootstrap_resamples, args.seed + 200,
        ),
        "matched_random_vs_clean_duplicate": paired_summary(
            ledgers["clean_duplicate"], ledgers["matched_random"],
            args.bootstrap_resamples, args.seed + 300,
        ),
    }
    primary = comparisons["targeted_vs_matched_random"]
    secondary = comparisons["targeted_vs_clean_duplicate"]
    gates = {
        "all_three_arms_complete_and_formal": True,
        "all_non_arm_configuration_identical": True,
        "all_input_provenance_identical": True,
        "all_epoch_batch_schedules_identical": True,
        "held_queries_and_initialization_identical": True,
        "targeted_beats_matched_random_formula_ci": bool(
            primary["formula_cluster_top1_ci"]["ci_low"] > 0
        ),
        "targeted_beats_clean_duplicate_formula_ci": bool(
            secondary["formula_cluster_top1_ci"]["ci_low"] > 0
        ),
        "targeted_vs_random_corrected_gt_introduced": bool(
            primary["corrected"] > primary["introduced"]
        ),
        "targeted_vs_random_risk_net_positive": bool(primary["risk_net_lambda2"] > 0),
        "targeted_vs_random_near_nonnegative": bool(primary["delta_near_recall1"] >= 0),
        "targeted_vs_random_mrr_nonnegative": bool(primary["delta_mrr"] >= 0),
    }
    report = {
        "status": "noise_final_e4a_causal_attribution_complete",
        "formal": True,
        "seed": args.seed,
        "outer_formula_fold": args.outer_fold,
        "arms": {
            arm: {
                "run": str(runs[arm]),
                "decision_sha256": sha256_file(runs[arm] / "decision.json"),
                "checkpoint_sha256": sha256_file(runs[arm] / "final_shared_encoder.pt"),
                "held": decisions[arm]["held_clean"],
                "causal_action_audit": decisions[arm]["causal_action_audit"],
            } for arm in ARMS
        },
        "comparisons": comparisons,
        "gates": gates,
        "pass_to_action_learnability": bool(all(gates.values())),
        "decision": (
            "proceed to clean-input action-learnability audit"
            if all(gates.values())
            else "do not scale or tune; targeted noise has not beaten its paired controls"
        ),
        "contracts": {
            "one_shared_clean_spectrum_encoder_per_arm": True,
            "only_action_view_differs_between_arms": True,
            "matched_controls_frozen_before_training": True,
            "outcomes_used_for_control_assignment": False,
            "full_candidate_graph_used_for_primary_clean_evaluation": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "claim_limit": (
            "One development formula fold. This is the causal attribution gate for the "
            "existing E4-A recipe, not multifold confirmation or sealed P3 performance."
        ),
    }
    # Publish the paired result atomically.  A preempted summarizer may leave a
    # PID-scoped temporary directory, but never a directory that looks like a
    # completed result to a recovery submission.
    temporary_output = output.with_name(f"{output.name}.tmp_{os.getpid()}")
    temporary_output.mkdir(parents=True, exist_ok=False)
    combined = reference_ledger[[
        "query_index", "query_formula", "has_near", "baseline_rank",
        "baseline_top_molecule_local", "baseline_full_margin",
    ]].copy()
    for arm in ARMS:
        combined[f"{arm}_rank"] = ledgers[arm]["final_rank"].to_numpy(np.int16)
        combined[f"{arm}_top_molecule_local"] = ledgers[arm][
            "final_top_molecule_local"
        ].to_numpy(np.int32)
        combined[f"{arm}_full_margin"] = ledgers[arm]["final_full_margin"].to_numpy(np.float32)
    combined.to_csv(
        temporary_output / "paired_per_query.csv.gz", index=False, compression="gzip"
    )
    json_dump(temporary_output / "report.json", report)
    temporary_output.rename(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
