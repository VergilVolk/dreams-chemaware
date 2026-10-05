"""Paired formula-cluster summary for global and error-routed rule teachers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from train_chemaware_full_candidate_alignment import formula_bootstrap  # noqa: E402


ARMS = (
    "none", "mass", "rule_response", "rule_mass", "rule_mass_shifted",
    "mass_error", "rule_mass_error", "rule_mass_shifted_error",
)
EXPECTED = {
    "none": ("none", 0.0, "all"),
    "mass": ("mass", 0.1, "all"),
    "rule_response": ("rule_response", 0.2, "all"),
    "rule_mass": ("rule_mass", 0.2, "all"),
    "rule_mass_shifted": ("rule_mass_shifted", 0.2, "all"),
    "mass_error": ("mass", 0.1, "official_error"),
    "rule_mass_error": ("rule_mass", 0.2, "official_error"),
    "rule_mass_shifted_error": ("rule_mass_shifted", 0.2, "official_error"),
}
TARGETS = ("rule_mass", "rule_mass_error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")
    reports = {}
    arrays = {}
    for arm in ARMS:
        report_path = args.root / arm / "report.json"
        query_path = args.root / arm / "inner_per_query.npz"
        if not report_path.is_file() or not query_path.is_file():
            raise FileNotFoundError(f"incomplete arm: {arm}")
        reports[arm] = json.loads(report_path.read_text(encoding="utf-8"))
        if reports[arm].get("status") not in ("ARM_COMPLETE", "PASS"):
            raise RuntimeError(f"arm is not artifact-complete: {arm}")
        with np.load(query_path) as loaded:
            arrays[arm] = {key: loaded[key] for key in loaded.files}
        expected_teacher, expected_beta, expected_scope = EXPECTED[arm]
        if reports[arm]["optimization"]["teacher_arm"] != expected_teacher:
            raise RuntimeError(f"teacher arm mismatch: {arm}")
        if not np.isclose(
            float(reports[arm]["optimization"]["teacher_beta"]), expected_beta,
        ):
            raise RuntimeError(f"teacher beta mismatch: {arm}")
        if reports[arm]["optimization"]["teacher_scope"] != expected_scope:
            raise RuntimeError(f"teacher scope mismatch: {arm}")
    paired_keys = (
        "inner_fold", "outer_fold", "seed", "epochs", "batch_queries",
        "references_per_molecule", "max_train_identities",
        "error_identity_fraction", "clean_safety_selection", "training_mass",
        "unfreeze_blocks", "backbone_lr", "head_lr", "temperature",
        "lambda_spectrum", "lambda_inbatch_spectrum", "lambda_teacher",
        "teacher_temperature", "lambda_margin_floor",
        "lambda_preserve", "grad_clip",
    )
    anchor_optimization = reports["rule_mass_error"]["optimization"]
    for arm in ARMS:
        differing = [
            key for key in paired_keys
            if reports[arm]["optimization"].get(key) != anchor_optimization.get(key)
        ]
        if differing:
            raise RuntimeError(f"arm optimization mismatch: {arm}/{differing}")
    anchor = arrays["rule_mass_error"]
    for arm in ARMS:
        for key in ("query", "formula", "old_rank"):
            if not np.array_equal(anchor[key], arrays[arm][key]):
                raise RuntimeError(f"paired schedule/evaluation mismatch: {arm}/{key}")
    old_hit = anchor["old_rank"] == 1

    def absolute_for(target: str, offset: int) -> dict:
        target_hit = arrays[target]["new_rank"] == 1
        return formula_bootstrap(
            target_hit.astype(float) - old_hit.astype(float), anchor["formula"],
            args.seed + 901 + 41 * offset, args.bootstrap_draws,
        )

    def compare(target: str, control: str, offset: int) -> dict:
        target_hit = arrays[target]["new_rank"] == 1
        control_hit = arrays[control]["new_rank"] == 1
        delta = target_hit.astype(float) - control_hit.astype(float)
        return {
            "recall1_advantage": float(np.mean(delta)),
            **formula_bootstrap(
                delta, anchor["formula"], args.seed + 1201 + 41 * offset,
                args.bootstrap_draws,
            ),
        }

    absolute = {
        target: absolute_for(target, offset)
        for offset, target in enumerate(TARGETS)
    }
    requested_comparisons = (
        ("rule_mass", "none"),
        ("rule_mass", "mass"),
        ("rule_mass", "rule_response"),
        ("rule_mass", "rule_mass_shifted"),
        ("rule_mass_error", "none"),
        ("rule_mass_error", "mass_error"),
        ("rule_mass_error", "rule_mass_shifted_error"),
        ("rule_mass_error", "rule_mass"),
    )
    comparisons = {
        f"{target}_minus_{control}": compare(target, control, offset)
        for offset, (target, control) in enumerate(requested_comparisons)
    }
    target_gates = {}
    control_by_target = {
        "rule_mass": ("mass", "rule_mass_shifted"),
        "rule_mass_error": ("mass_error", "rule_mass_shifted_error"),
    }
    for target in TARGETS:
        summary = reports[target]["final_inner"]
        mass_control, shifted_control = control_by_target[target]
        target_gates[target] = {
            "student_improves_official": absolute[target]["formula_cluster_bootstrap_95ci"][0] > 0,
            "continuation_point_estimate_positive": comparisons[f"{target}_minus_none"]["recall1_advantage"] > 0,
            "beats_continuation_formula_ci": comparisons[f"{target}_minus_none"]["formula_cluster_bootstrap_95ci"][0] > 0,
            "mass_control_point_estimate_positive": comparisons[f"{target}_minus_{mass_control}"]["recall1_advantage"] > 0,
            "beats_dose_matched_mass_teacher_formula_ci": comparisons[f"{target}_minus_{mass_control}"]["formula_cluster_bootstrap_95ci"][0] > 0,
            "shifted_control_point_estimate_positive": comparisons[f"{target}_minus_{shifted_control}"]["recall1_advantage"] > 0,
            "beats_dose_matched_shifted_rules_formula_ci": comparisons[f"{target}_minus_{shifted_control}"]["formula_cluster_bootstrap_95ci"][0] > 0,
            "corrected_exceeds_introduced": summary["corrected"] > summary["introduced"],
            "model_preserved": reports[target]["preservation"] >= 0.995,
            "selected_trained_step": reports[target]["selected_step"] > 0,
        }
    eligible = [target for target in TARGETS if all(target_gates[target].values())]
    selected = (
        max(
            eligible,
            key=lambda target: (
                reports[target]["final_inner"]["recall1"],
                reports[target]["final_inner"]["mrr"],
                reports[target]["preservation"],
            ),
        )
        if eligible else None
    )
    report = {
        "status": (
            "CAUSAL_CHEMISTRY_DEVELOPMENT_PASS"
            if selected is not None else "CAUSAL_CHEMISTRY_DEVELOPMENT_FAIL"
        ),
        "scope": (
            "development eight-arm global/routed direct shared-DreaMS "
            "chemical-rule transfer; outer untouched"
        ),
        "selected_target_for_outer_confirmation": selected,
        "release_eligible": False,
        "reason_not_release_eligible": (
            "development causal gate passed but untouched outer confirmation is required"
            if selected is not None else
            "no arm passed the matched causal chemistry gates"
        ),
        "arms": {arm: {
            "status": reports[arm]["status"],
            "selected_step": reports[arm]["selected_step"],
            "final_inner": reports[arm]["final_inner"],
            "preservation": reports[arm]["preservation"],
            "mean_clip_fraction": reports[arm]["mean_clip_fraction"],
        } for arm in ARMS},
        "target_absolute_formula_bootstrap": absolute,
        "paired_formula_cluster_comparisons": comparisons,
        "target_gates": target_gates,
        "contracts": {
            "same_query_order_all_arms": True,
            "same_official_initialization": True,
            "same_schedule_except_teacher_variant_beta_and_training_only_scope": True,
            "global_and_routed_shifted_rule_controls": True,
            "routed_mass_and_shifted_controls_are_scope_matched": True,
            "clean_spectrum_student_at_inference": True,
            "outer_fold_evaluated": False,
            "single_arm_status_never_implies_chemistry": True,
            "development_pass_never_implies_release": True,
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
