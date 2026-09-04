"""Fail-closed validator for one E14 outer-fold-excluded action teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    action_path = args.output_dir / "selected_actions.csv.gz"
    safety_path = args.output_dir / "action_safety.csv.gz"
    risk_path = args.output_dir / "risk_controls.csv.gz"
    outcome_path = args.output_dir / "action_outcomes.npz"
    if not all(path.is_file() for path in (
        report_path, action_path, safety_path, risk_path, outcome_path,
    )):
        raise FileNotFoundError("E14 teacher output is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e14_crossfit_p_teacher_complete"
        or not report.get("formal")
    ):
        raise RuntimeError("E14 action teacher report is incomplete or non-formal")
    capacity_authorized = bool(report.get("pass_to_shared_encoder_transfer"))
    amendment_path = args.output_dir / "capacity_amendment.json"
    amendment = {}
    if not capacity_authorized and amendment_path.is_file():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        capacity_authorized = bool(
            amendment.get("status") == "noise_final_e14_capacity_amendment_complete"
            and amendment.get("formal")
            and amendment.get("posthoc_amendment")
            and amendment.get("original_report_unchanged")
            and amendment.get("pass_to_shared_encoder_transfer")
            and amendment.get("provenance", {}).get("report_sha256")
            == sha256_file(report_path)
            and amendment.get("provenance", {}).get("selected_actions_sha256")
            == sha256_file(action_path)
            and amendment.get("provenance", {}).get("action_outcomes_sha256")
            == sha256_file(outcome_path)
        )
    if not capacity_authorized:
        failed = {
            key: value for key, value in report.get("gates", {}).items()
            if not bool(value)
        }
        diagnostics = {
            "outer_formula_fold": report.get("outer_formula_fold"),
            "queries": report.get("queries"),
            "official_errors": report.get("official_errors"),
            "mature_crossfit_errors": report.get("mature_crossfit_errors"),
            "selected_corrective_queries": report.get("selected_corrective_queries"),
            "selected_corrective_identities": report.get("selected_corrective_identities"),
            "selected_corrective_formulas": report.get("selected_corrective_formulas"),
            "selected_incremental_headroom_beyond_mature_crossfit": report.get(
                "selected_incremental_headroom_beyond_mature_crossfit"
            ),
            "prior_safe_actions": report.get("prior_action_filter", {}).get(
                "prior_safe_actions"
            ),
            "outer_train_replicated_safe_actions": report.get(
                "outer_train_replicated_safe_actions"
            ),
            "risk_controls": report.get("risk_controls"),
            "failed_gates": failed,
        }
        raise RuntimeError(
            "E14 action teacher capacity gate failed:\n"
            + json.dumps(diagnostics, indent=2, sort_keys=True)
        )
    contracts = report.get("contracts", {})
    required_contracts = {
        "teacher_checkpoint_excludes_student_outer_formula_fold": True,
        "mature_checkpoint_and_decision_fold_verified": True,
        "held_fold_membership_reconstructed_fail_closed": True,
        "all_selected_queries_exclude_student_outer_formula_fold": True,
        "selected_query_formula_fold_is_materialized": True,
        "prior_fixed_cell_safety_filter_applied": True,
        "outer_train_multifold_action_safety_filter_applied": True,
        "action_specific_risk_controls_materialized": True,
        "one_selected_action_per_query": True,
        "only_official_errors_selected": True,
        "selected_action_rank_is_one": True,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    for key, expected in required_contracts.items():
        if contracts.get(key) != expected:
            raise RuntimeError(f"E14 teacher contract failed: {key}")
    actions = pd.read_csv(action_path)
    safety = pd.read_csv(safety_path)
    risk = pd.read_csv(risk_path)
    if len(actions) != int(report["selected_corrective_queries"]):
        raise RuntimeError("E14 teacher action count drifted")
    if actions["query_index"].duplicated().any():
        raise RuntimeError("E14 teacher repeats a query")
    eligible = set(safety.loc[safety["replicated_safe"].astype(bool), "action_id"].astype(str))
    if not eligible or not set(actions["action_id"].astype(str)) <= eligible:
        raise RuntimeError("E14 selected an action outside the replicated-safe set")
    if len(risk) != int(report.get("risk_controls", -1)):
        raise RuntimeError("E14 risk-control count drifted")
    if risk[["query_index", "action_id"]].duplicated().any():
        raise RuntimeError("E14 risk controls repeat a query/action pair")
    if not (
        risk["crossfit_clean_rank"].astype(int).eq(1).all()
        and risk["control_kind"].astype(str).isin(
            {"introduced", "protected_boundary"}
        ).all()
        and set(risk["action_id"].astype(str)) <= eligible
    ):
        raise RuntimeError("E14 risk controls violate the mature-clean safety contract")
    if not (
        actions["official_rank"].astype(int).ne(1).all()
        and actions["crossfit_clean_rank"].astype(int).ne(1).all()
        and actions["teacher_rank"].astype(int).eq(1).all()
        and actions["teacher_margin"].astype(float).gt(0).all()
    ):
        raise RuntimeError("E14 teacher contains a non-corrective action")
    print(
        "[validate_noise_final_e14_crossfit_teacher] PASS "
        f"actions={len(actions):,} identities={actions['query_ik14'].nunique():,} "
        f"authorization={'amendment' if amendment else 'original'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
