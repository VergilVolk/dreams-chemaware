"""Validate an atomic L2 paired counterfactual pilot output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [
        args.output_dir / "report.json", args.output_dir / "training_actions.csv.gz",
        args.output_dir / "held_per_query.csv.gz",
        args.output_dir / "matched_random_shared_encoder.pt",
        args.output_dir / "targeted_shared_encoder.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    contracts = report.get("contracts", {})
    if report.get("status") != "noise_final_l2_paired_counterfactual_complete":
        raise RuntimeError("wrong L2 status")
    if contracts.get("P2b") != "forbidden" or contracts.get("P3_consumed"):
        raise RuntimeError("L2 violates P2b/P3 boundary")
    if not all(contracts.get(name) for name in (
        "mature_n_arm_action_universe_only", "zero_clean_oof_selector_is_routed_to_no_op",
        "selection_uses_formula_oof_clean_input_predictions",
        "multiple_actions_retained_in_ledger", "same_schedule_optimizer_and_loss_for_both_arms",
        "arm_difference_is_primary_action_path_only", "inference_clean_spectrum_only",
        "one_query_per_identity_per_epoch", "action_priority_uses_clean_oof_predictions_only",
    )):
        raise RuntimeError("L2 contract is incomplete")
    actions = pd.read_csv(required[1], low_memory=False)
    held = pd.read_csv(required[2], low_memory=False)
    if actions.empty or held.empty or actions["query_formula"].astype(str).isin(set(held["query_formula"].astype(str))).any():
        raise RuntimeError("L2 training/held formula isolation failed")
    active_selectors = set(actions["selector"].astype(str))
    if not active_selectors or not active_selectors.issubset({"candidate_gradient", "role_confounder"}):
        raise RuntimeError(f"L2 has an invalid active selector set: {sorted(active_selectors)}")
    if sorted(active_selectors) != report.get("active_l1_deployable_selectors"):
        raise RuntimeError("L2 active selector report differs from its training artifact")
    required_control_columns = {"selected_control_index", "selected_control_path", "alternate_control_path"}
    if not required_control_columns.issubset(actions.columns):
        raise RuntimeError("L2 did not materialize its frozen matched-control assignment")
    if not set(actions["selected_control_index"].astype(int)).issubset({0, 1}):
        raise RuntimeError("L2 selected-control indices are invalid")
    for row in actions[[
        "matched_control_paths", "selected_control_index",
        "selected_control_path", "alternate_control_path",
    ]].itertuples(index=False):
        controls = str(row.matched_control_paths).split(";")
        index = int(row.selected_control_index)
        if len(controls) != 2 or str(row.selected_control_path) != controls[index] or str(row.alternate_control_path) != controls[1 - index]:
            raise RuntimeError("L2 materialized matched-control path does not match its frozen index")
    exposure = report.get("selection", {}).get("exposure", {})
    if exposure.get("within_epoch_action_recycling") or exposure.get("maximum_action_exposure", 0) > exposure.get("epochs", 0):
        raise RuntimeError("L2 action exposure violates the bounded rotating schedule")
    observed_preservation_gate = True
    for arm in ("matched_random", "targeted"):
        preservation = report.get("arms", {}).get(arm, {}).get("summary", {}).get("initialization_preservation", {})
        observed_preservation_gate &= preservation.get("mean", 0.0) >= 0.995
        advantage = report.get("arms", {}).get(arm, {}).get("gradient_audit", {}).get("paired_advantage", {})
        if advantage.get("groups", 0) < 1 or advantage.get("nonzero_groups") != advantage.get("groups"):
            raise RuntimeError(f"{arm} paired-advantage gradient audit failed")
        risk = report.get("arms", {}).get(arm, {}).get("gradient_audit", {}).get("risk_total", {})
        if risk.get("groups", 0) < 1 or risk.get("nonzero_groups") != risk.get("groups"):
            raise RuntimeError(f"{arm} risk/no-op gradient audit failed")
    if report.get("gates", {}).get("both_arms_preservation_ge_0_995") != observed_preservation_gate:
        raise RuntimeError("L2 preservation gate is inconsistent with the reported arm summaries")
    if len(held) != report.get("selection", {}).get("full_held_fold_queries"):
        raise RuntimeError("L2 held per-query artifact is not the complete reported formula fold")
    if not report.get("formal") and report.get("pass_to_second_seed"):
        raise RuntimeError("smoke output cannot authorize another seed")
    print(f"[validate_noise_final_l2_paired_counterfactual] PASS queries={len(held):,}", flush=True)


if __name__ == "__main__":
    main()
