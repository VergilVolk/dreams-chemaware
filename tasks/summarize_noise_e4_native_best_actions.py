"""Compare targeted and shuffled best-action arms under the historical E4 kernel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from summarize_noise_corrected_direct_v3_arms import formula_cluster_ci
from summarize_noise_corrected_direct_v3_canary import _complete_metric_delta


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def _load(path: Path, arm: str) -> tuple[dict[str, object], pd.DataFrame, str]:
    decision_path = path / "decision.json"
    table_path = path / "held_per_query.csv.gz"
    checkpoint_path = path / "final_shared_encoder.pt"
    if missing := [
        str(value) for value in (decision_path, table_path, checkpoint_path)
        if not value.is_file()
    ]:
        raise FileNotFoundError(missing)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    configuration = decision.get("configuration", {})
    contracts = decision.get("contracts", {})
    injection_mode = configuration.get("materialized_injection_mode", "one_best_e4")
    optimizer_boundary_mode = configuration.get(
        "optimizer_boundary_mode", "ordinary_adamw",
    )
    mode_contract = (
        contracts.get("materialized_action_loss_is_historical_e4") is True
        if injection_mode == "one_best_e4"
        else (
            injection_mode == "multi_action_balanced"
            and contracts.get(
                "materialized_action_loss_is_balanced_multi_action_direct"
            ) is True
            and contracts.get("materialized_all_strict_actions_preserved") is True
            and contracts.get(
                "materialized_satisfied_action_rank_gradient_gated"
            ) is True
            and contracts.get(
                "materialized_identity_family_equal_effective_dose"
            ) is True
        )
    )
    if (
        decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
        or decision.get("formal") is not True
        or configuration.get("action_selection") != "materialized_routed"
        or configuration.get("materialized_action_arm") != arm
        or configuration.get("direct_transfer_mode") != "symmetric"
        or configuration.get("rank_reference_mode") != "shared"
        or not mode_contract
        or contracts.get("materialized_action_embedding_target_used") is not False
        or contracts.get("materialized_action_margin_target_used") is not False
        or contracts.get("teacher") != "outer_train_action_routing_only_no_teacher_target"
    ):
        raise RuntimeError(f"historical E4 materialized-action contract failed: {path}")
    if optimizer_boundary_mode == "action_injector_v1":
        optimizer_injection = decision.get("optimizer_boundary_injection", {})
        if (
            injection_mode != "multi_action_balanced"
            or contracts.get("action_injector_v1_used") is not True
            or contracts.get(
                "action_injector_v1_receives_complete_historical_e4_gradient"
            ) is not True
            or contracts.get(
                "action_injector_v1_action_selection_or_tensor_mutation"
            ) is not False
            or contracts.get("action_injector_v1_historical_e4_loss_mutation")
            is not False
            or contracts.get("action_injector_v1_every_optimizer_step_audited")
            is not True
            or optimizer_injection.get("gate_passed") is not True
        ):
            raise RuntimeError(f"E4 ActionInjectorV1 contract failed: {path}")
    elif optimizer_boundary_mode != "ordinary_adamw":
        raise RuntimeError(f"unknown E4 optimizer boundary: {optimizer_boundary_mode}")
    table = pd.read_csv(table_path, low_memory=False).sort_values(
        "query_index", kind="stable",
    ).reset_index(drop=True)
    required = {
        "query_index", "query_formula", "has_near", "baseline_rank",
        "initialization_rank", "final_rank",
    }
    if missing_columns := required - set(table.columns):
        raise RuntimeError(f"E4 held table lacks columns: {sorted(missing_columns)}")
    if table["query_index"].duplicated().any():
        raise RuntimeError("E4 held table repeats a query")
    return decision, table, sha256_file(checkpoint_path)


def _top1(left: np.ndarray, right: np.ndarray, near: np.ndarray) -> dict[str, object]:
    corrected = (left > 1) & (right == 1)
    introduced = (left == 1) & (right > 1)

    def counts(mask: np.ndarray) -> dict[str, int]:
        fixed = int(np.sum(corrected & mask))
        broken = int(np.sum(introduced & mask))
        return {
            "corrected": fixed,
            "introduced": broken,
            "risk_net_lambda2": fixed - 2 * broken,
        }

    return {
        **counts(np.ones(len(left), dtype=bool)),
        "near": counts(np.asarray(near, dtype=bool)),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    targeted, targeted_table, targeted_checkpoint = _load(
        args.targeted_dir, "targeted",
    )
    shuffled, shuffled_table, shuffled_checkpoint = _load(
        args.shuffled_dir, "shuffled",
    )
    identity = ["query_index", "query_formula", "has_near", "initialization_rank"]
    if not targeted_table[identity].equals(shuffled_table[identity]):
        raise RuntimeError("targeted and shuffled E4 arms do not share the held ledger")
    target_mode = targeted["configuration"].get(
        "materialized_injection_mode", "one_best_e4",
    )
    shuffle_mode = shuffled["configuration"].get(
        "materialized_injection_mode", "one_best_e4",
    )
    if target_mode != shuffle_mode:
        raise RuntimeError("targeted and shuffled E4 arms use different injection modes")
    target_optimizer_boundary = targeted["configuration"].get(
        "optimizer_boundary_mode", "ordinary_adamw",
    )
    shuffle_optimizer_boundary = shuffled["configuration"].get(
        "optimizer_boundary_mode", "ordinary_adamw",
    )
    if target_optimizer_boundary != shuffle_optimizer_boundary:
        raise RuntimeError("targeted and shuffled E4 arms use different optimizer boundaries")
    shared_provenance = (
        "graph_sha256", "source_manifest_sha256", "official_checkpoint_sha256",
        "initial_student_checkpoint_sha256", "materialized_action_report_sha256",
        "materialized_training_actions_sha256", "materialized_action_spectra_sha256",
        "script_sha256", "action_injector_v1_sha256",
        "e4_action_injector_v1_bridge_sha256",
    )
    for key in shared_provenance:
        if targeted.get("provenance", {}).get(key) != shuffled.get(
            "provenance", {}
        ).get(key):
            raise RuntimeError(f"E4 causal arms differ in provenance: {key}")
    schedule_keys = (
        "action_sampling_schedule_sha256", "action_exposure_schedule_sha256",
        "safety_sampling_schedule_sha256", "steps",
    )
    targeted_history = targeted.get("history", [])
    shuffled_history = shuffled.get("history", [])
    if len(targeted_history) != len(shuffled_history) or not targeted_history:
        raise RuntimeError("E4 causal arms do not share a non-empty epoch schedule")
    for left, right in zip(targeted_history, shuffled_history):
        if any(left.get(key) != right.get(key) for key in schedule_keys):
            raise RuntimeError("E4 causal arms differ in sampling schedule")
    targeted_control = targeted.get("materialized_action_control", {})
    shuffled_control = shuffled.get("materialized_action_control", {})
    selection_keys = (
        "qualifying_strict_top1_corrective_rows",
        "selected_best_action_union_rows",
        "selected_strict_top1_corrective_queries",
        "qualifying_sources", "sources", "families", "selected_action_rows_preserved",
        "one_maximum_margin_action_per_query",
        "strict_clean_wrong_action_top1_only",
        "causal_control_donors_restricted_to_selected_union",
        "selected_actions_equal_unit_weight",
        "multi_action_panel_all_strict",
        "identity_family_effective_dose_normalized",
        "effective_action_weight_sum",
        "effective_action_weight_per_identity",
        "routing_scores_not_used_as_loss_targets",
        "historical_identity_exposure_budget_restored",
        "identity_effective_dose_restored",
        "all_strict_actions_exposed_before_recycling",
        "all_strict_actions_exposed_exactly_once",
        "physical_action_recycling_used",
        "static_preclip_loss_scale",
        "identity_exposure_target_per_identity",
        "identities_exceeding_historical_budget",
    )
    if any(targeted_control.get(key) != shuffled_control.get(key) for key in selection_keys):
        raise RuntimeError("E4 causal arms do not share the exact best-action union")
    if target_mode == "one_best_e4":
        if (
            targeted_control.get("one_maximum_margin_action_per_query") is not True
            or int(targeted_control.get("selected_best_action_union_rows", 0))
            != int(targeted_control.get("selected_strict_top1_corrective_queries", -1))
        ):
            raise RuntimeError("E4 best-action union is not one winning action per query")
    elif target_mode == "multi_action_balanced" and (
        targeted_control.get("multi_action_panel_all_strict") is not True
        or targeted_control.get("identity_family_effective_dose_normalized") is not True
        or targeted_control.get("all_strict_actions_exposed_before_recycling") is not True
        or targeted_control.get("historical_identity_exposure_budget_restored") is not True
        or int(targeted_control.get("selected_best_action_union_rows", 0))
        <= int(targeted_control.get("selected_strict_top1_corrective_queries", -1))
    ):
        raise RuntimeError("E4 multi-action coverage/effective-dose contract failed")
    elif target_mode not in {"one_best_e4", "multi_action_balanced"}:
        raise RuntimeError(f"unknown materialized injection mode: {target_mode}")

    targeted_metrics = targeted["held_clean"]["corrected_graph_registered_metrics"]
    shuffled_metrics = shuffled["held_clean"]["corrected_graph_registered_metrics"]
    if targeted_metrics["initialization"] != shuffled_metrics["initialization"]:
        raise RuntimeError("E4 causal arms do not share exact initial metrics")
    initial = targeted_table["initialization_rank"].to_numpy(np.int64)
    target_rank = targeted_table["final_rank"].to_numpy(np.int64)
    shuffle_rank = shuffled_table["final_rank"].to_numpy(np.int64)
    exact_recall_checks = {
        "target_initialization": (
            float(np.mean(initial == 1)),
            float(targeted_metrics["initialization"]["retrieval"]["recall@1"]),
        ),
        "target_student": (
            float(np.mean(target_rank == 1)),
            float(targeted_metrics["student"]["retrieval"]["recall@1"]),
        ),
        "shuffle_student": (
            float(np.mean(shuffle_rank == 1)),
            float(shuffled_metrics["student"]["retrieval"]["recall@1"]),
        ),
    }
    if any(
        not np.isclose(left, right, rtol=0.0, atol=1e-12)
        for left, right in exact_recall_checks.values()
    ):
        raise RuntimeError(
            f"held-table and registered exact Recall@1 disagree: {exact_recall_checks}"
        )
    formulas = targeted_table["query_formula"].astype(str).to_numpy()
    near = targeted_table["has_near"].to_numpy(bool)
    target_effect = (target_rank == 1).astype(float) - (initial == 1).astype(float)
    shuffle_effect = (target_rank == 1).astype(float) - (shuffle_rank == 1).astype(float)
    ci_initial = formula_cluster_ci(
        formulas, target_effect, args.bootstrap_resamples, args.seed,
        familywise_hypotheses=2,
    )
    ci_shuffled = formula_cluster_ci(
        formulas, shuffle_effect, args.bootstrap_resamples, args.seed + 1,
        familywise_hypotheses=2,
    )
    delta_initial = _complete_metric_delta(
        targeted_metrics["initialization"], targeted_metrics["student"],
    )
    delta_shuffled = _complete_metric_delta(
        shuffled_metrics["student"], targeted_metrics["student"],
    )
    outcomes_initial = _top1(initial, target_rank, near)
    outcomes_shuffled = _top1(shuffle_rank, target_rank, near)
    recall1_delta_pp = float(100.0 * np.mean(target_effect))
    recall1_vs_shuffled_pp = float(100.0 * np.mean(shuffle_effect))
    directional_initial = bool(all(value >= -1e-12 for value in delta_initial.values()))
    directional_shuffled = bool(all(value >= -1e-12 for value in delta_shuffled.values()))
    gates = {
        "targeted_gain_at_least_4pp_vs_initial_E8": recall1_delta_pp >= 4.0,
        "targeted_beats_shuffled_on_recall1": recall1_vs_shuffled_pp > 0.0,
        "formula_cluster_ci_low_strict_positive_vs_initial": (
            float(ci_initial["ci_low_pp"]) > 0.0
        ),
        "formula_cluster_ci_low_strict_positive_vs_shuffled": (
            float(ci_shuffled["ci_low_pp"]) > 0.0
        ),
        "risk_net_positive_vs_initial": outcomes_initial["risk_net_lambda2"] > 0,
        "risk_net_positive_vs_shuffled": outcomes_shuffled["risk_net_lambda2"] > 0,
        "all_registered_metrics_nonnegative_vs_initial": directional_initial,
        "all_registered_metrics_nonnegative_vs_shuffled": directional_shuffled,
    }
    if target_optimizer_boundary == "action_injector_v1":
        gates["action_injector_v1_signal_gate_passed_both_arms"] = bool(
            targeted.get("optimizer_boundary_injection", {}).get("gate_passed")
            is True
            and shuffled.get("optimizer_boundary_injection", {}).get("gate_passed")
            is True
        )
    report = {
        "status": "noise_e4_native_best_actions_summary_complete",
        "formal": True,
        "held_queries": int(len(targeted_table)),
        "method": (
            f"{target_mode} E4 symmetric/shared direct loss with the frozen "
            f"materialized action ledger and {target_optimizer_boundary}; "
            "no teacher embedding or margin target"
        ),
        "targeted_vs_initial_E8": {
            "recall1_delta_pp": recall1_delta_pp,
            "formula_cluster_recall1_delta": ci_initial,
            "top1_outcomes": outcomes_initial,
            "complete_metric_deltas": delta_initial,
        },
        "targeted_vs_shuffled": {
            "recall1_delta_pp": recall1_vs_shuffled_pp,
            "formula_cluster_recall1_delta": ci_shuffled,
            "top1_outcomes": outcomes_shuffled,
            "complete_metric_deltas": delta_shuffled,
        },
        "gates": gates,
        "four_pp_full_metric_gate_passed": bool(all(gates.values())),
        "promotion_authorized": bool(all(gates.values())),
        "contracts": {
            "same_initial_E8": True,
            "same_held_query_ledger": True,
            "same_action_rows_and_schedule_budget": True,
            "only_action_view_differs_between_causal_arms": True,
            "complete_registered_metric_family_compared": True,
            "held_table_and_registered_recall1_exactly_identical": True,
            "massspecgym_pairwise_not_nist20_replication": True,
            "teacher_embedding_or_margin_target_used": False,
            "optimizer_boundary_mode": target_optimizer_boundary,
            "action_injector_v1_signal_gate_passed_both_arms": bool(
                target_optimizer_boundary == "action_injector_v1"
                and targeted.get("optimizer_boundary_injection", {}).get(
                    "gate_passed"
                ) is True
                and shuffled.get("optimizer_boundary_injection", {}).get(
                    "gate_passed"
                ) is True
            ),
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "targeted_checkpoint_sha256": targeted_checkpoint,
            "shuffled_checkpoint_sha256": shuffled_checkpoint,
            **{
                ("trainer_script_sha256" if key == "script_sha256" else key):
                targeted["provenance"][key]
                for key in shared_provenance
            },
            "summary_script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Corrected train-side formula-held development result. A 4--5 pp claim "
            "requires four_pp_full_metric_gate_passed=true; the old action union alone "
            "does not satisfy this encoder gate."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        targeted_table.assign(
            shuffled_rank=shuffle_rank,
            targeted_correct_vs_initial=(initial > 1) & (target_rank == 1),
            targeted_introduced_vs_initial=(initial == 1) & (target_rank > 1),
            targeted_correct_vs_shuffled=(shuffle_rank > 1) & (target_rank == 1),
            targeted_introduced_vs_shuffled=(shuffle_rank == 1) & (target_rank > 1),
        ).to_csv(staging / "paired_per_query.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
