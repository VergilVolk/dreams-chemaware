"""Refined, selection-aware audit of the completed Noise-v3 S3A matrix.

The first S3A report deliberately retained every result, but raw cell counts
cannot be compared across steps because longer paths have fewer eligible
queries.  This audit adds fixed-action confidence intervals, six-step
complete-case trajectories, conservative safety gates, selector-level unions,
and an explicitly exploratory rule-evidence analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3a-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s3a_extended_matrix"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def cluster_summary(
    frame: pd.DataFrame, value: str, cluster: str, bootstrap: int, seed: int,
) -> dict:
    cluster_values = frame.groupby(cluster, sort=False)[value].mean().dropna().to_numpy(float)
    if not len(cluster_values):
        raise RuntimeError(f"no finite {value} values")
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(cluster_values, len(cluster_values), replace=True).mean()
        for _ in range(bootstrap)
    ])
    return {
        "clusters": int(len(cluster_values)),
        "macro_mean": float(cluster_values.mean()),
        "bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def cell_name(selector: str, attenuation: float, step: int) -> str:
    return f"{selector}|a={float(attenuation):.2f}|step={int(step)}"


def main() -> None:
    args = parse_args()
    paired_path = args.s3a_dir / "paired_interventions.csv.gz"
    transition_path = args.s3a_dir / "transition_audit.csv.gz"
    decision_path = args.s3a_dir / "decision.json"
    validation_path = args.s3a_dir / "matrix_validation.json"
    for path in (paired_path, transition_path, decision_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "noise_v3_s3a_matrix_validation_passed":
        raise RuntimeError("S3A fail-closed validation did not pass")
    original = json.loads(decision_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path)
    transitions = pd.read_csv(transition_path)
    total_queries = int(original["queries"])

    fixed_actions: dict[str, dict] = {}
    safe_cells: set[str] = set()
    for position, (key, group) in enumerate(paired.groupby(
        ["selector", "attenuation", "step"], sort=True,
    )):
        selector, attenuation, step = key
        name = cell_name(str(selector), float(attenuation), int(step))
        local = group.assign(
            delta_top1=(
                group["target_rank"].eq(1).astype(float)
                - group["baseline_rank"].eq(1).astype(float)
            )
        )
        corrected = int(local["corrected"].sum())
        introduced = int(local["introduced"].sum())
        formula_effect = cluster_summary(
            local, "delta_top1", "query_formula", args.bootstrap,
            args.seed + position,
        )
        identity_effect = cluster_summary(
            local, "delta_top1", "query_ik14", args.bootstrap,
            args.seed + 10_000 + position,
        )
        specificity = original["action_results"][name]["specificity_gate"]
        conservative_safe = bool(
            specificity
            and formula_effect["bootstrap_95ci"][0] > 0
            and identity_effect["bootstrap_95ci"][0] > 0
            and corrected >= 2 * introduced
        )
        if conservative_safe:
            safe_cells.add(name)
        fixed_actions[name] = {
            "eligible_queries": int(len(local)),
            "eligible_fraction": float(len(local) / total_queries),
            "corrected": corrected,
            "introduced": introduced,
            "net": corrected - introduced,
            "full_graph_delta_recall1": float((corrected - introduced) / total_queries),
            "conditional_delta_recall1": float((corrected - introduced) / len(local)),
            "formula_cluster_effect": formula_effect,
            "identity_cluster_effect": identity_effect,
            "matched_random_specificity_gate": bool(specificity),
            "corrected_ge_twice_introduced": bool(corrected >= 2 * introduced),
            "conservative_safe_cell": conservative_safe,
        }

    selector_unions: dict[str, dict] = {}
    for selector, group in paired.groupby("selector", sort=True):
        corrected = set(map(int, group.loc[group["corrected"], "query_index"]))
        introduced = set(map(int, group.loc[group["introduced"], "query_index"]))
        selector_unions[str(selector)] = {
            "unique_corrected_queries": len(corrected),
            "unique_introduced_queries": len(introduced),
            "unique_changed_overlap": len(corrected & introduced),
            "warning": "Union is outcome-selected headroom/risk, not a fixed policy.",
        }

    safe_corrected: set[int] = set()
    safe_introduced: set[int] = set()
    for (selector, attenuation, step), group in paired.groupby(
        ["selector", "attenuation", "step"], sort=True,
    ):
        name = cell_name(str(selector), float(attenuation), int(step))
        if name in safe_cells:
            safe_corrected.update(map(int, group.loc[group["corrected"], "query_index"]))
            safe_introduced.update(map(int, group.loc[group["introduced"], "query_index"]))

    trajectories: dict[str, dict] = {}
    trajectory_rows: list[dict] = []
    for action_position, ((selector, attenuation), group) in enumerate(paired.groupby(
        ["selector", "attenuation"], sort=True,
    )):
        complete = set(map(int, group.loc[group["step"].eq(6), "query_index"]))
        local = group.loc[group["query_index"].isin(complete)].copy()
        if not complete:
            continue
        previous = None
        steps: dict[str, dict] = {}
        for step in range(1, 7):
            current = local.loc[local["step"].eq(step)].copy()
            if len(current) != len(complete):
                raise RuntimeError("complete-case trajectory lost queries")
            current = current.sort_values("query_index")
            target_top1 = current["target_rank"].eq(1).astype(float).to_numpy()
            if previous is None:
                previous = current["baseline_rank"].eq(1).astype(float).to_numpy()
            current["marginal_top1"] = target_top1 - previous
            marginal = cluster_summary(
                current, "marginal_top1", "query_formula", args.bootstrap,
                args.seed + 20_000 + action_position * 10 + step,
            )
            corrected = int((current["baseline_rank"].gt(1) & current["target_rank"].eq(1)).sum())
            introduced = int((current["baseline_rank"].eq(1) & current["target_rank"].gt(1)).sum())
            steps[str(step)] = {
                "corrected_from_clean_baseline": corrected,
                "introduced_from_clean_baseline": introduced,
                "net_from_clean_baseline": corrected - introduced,
                "mean_target_minus_random_top1": float(current["target_minus_random_top1"].mean()),
                "mean_target_minus_random_margin": float(
                    current["target_minus_random_margin_change"].mean()
                ),
                "marginal_vs_previous_step_formula_cluster": marginal,
            }
            trajectory_rows.append({
                "selector": selector, "attenuation": attenuation, "step": step,
                "complete_queries": len(complete), "corrected": corrected,
                "introduced": introduced, "net": corrected - introduced,
                "marginal_formula_macro": marginal["macro_mean"],
                "marginal_formula_ci_low": marginal["bootstrap_95ci"][0],
                "marginal_formula_ci_high": marginal["bootstrap_95ci"][1],
            })
            previous = target_top1
        trajectories[f"{selector}|a={float(attenuation):.2f}"] = {
            "complete_six_step_queries": len(complete),
            "steps": steps,
            "interpretation": "All six steps are compared on the identical complete-case query set.",
        }

    rule_auc: dict[str, dict] = {}
    transitions["is_corrected"] = transitions["transition"].eq("corrected").astype(int)
    for selector, group in transitions.groupby("selector", sort=True):
        entry: dict[str, object] = {
            "rows": int(len(group)),
            "unique_queries": int(group["query_index"].nunique()),
            "corrected_fraction": float(group["is_corrected"].mean()),
        }
        for feature in (
            "baseline_rule_jaccard", "target_rule_jaccard",
            "baseline_rule_jaccard_core", "target_rule_jaccard_core",
            "baseline_rule_jaccard_massbank", "target_rule_jaccard_massbank",
        ):
            finite = group.dropna(subset=[feature])
            if finite["is_corrected"].nunique() == 2:
                entry[f"auc_lower_{feature}_predicts_correction"] = float(
                    roc_auc_score(finite["is_corrected"], -finite[feature])
                )
        rule_auc[str(selector)] = entry

    introduced = transitions.loc[transitions["transition"].eq("introduced")]
    tie_destinations = int(introduced["winner_mces_grade_name"].eq("identity").sum())
    report = {
        "status": "noise_v3_s3a_refined_audit_complete",
        "integrity": {
            "formal_queries": total_queries,
            "matrix_cells": int(len(fixed_actions)),
            "paired_action_rows": int(len(paired)),
            "validation_status": validation["status"],
        },
        "fixed_action_results": fixed_actions,
        "conservative_safe_cells": sorted(safe_cells),
        "conservative_safe_action_union": {
            "unique_correctable_queries": len(safe_corrected),
            "unique_at_risk_queries": len(safe_introduced),
            "note": (
                "The two sets are separate outcome-selected unions. Their difference is not "
                "a deployable net gain and must not be reported as model performance."
            ),
        },
        "selector_unions": selector_unions,
        "complete_case_trajectories": trajectories,
        "rule_evidence_exploratory": {
            "by_selector": rule_auc,
            "claim_limit": (
                "AUCs reuse query-action rows across steps and are descriptive only. Formula-group "
                "OOF policy ablation is required before claiming incremental chemical-prior value."
            ),
        },
        "introduced_error_destination_audit": {
            "unique_queries": int(introduced["query_index"].nunique()),
            "same_formula_fraction_across_transition_rows": float(
                introduced["winner_same_formula"].mean()
            ),
            "near_destination_fraction_across_transition_rows": float(
                introduced["winner_mces_grade_name"].eq("near").mean()
            ),
            "strict_tie_rows_with_identity_argmax_metadata": tie_destinations,
            "tie_warning": (
                "For strict-rank ties, numpy argmax may retain the identity candidate even though "
                "a tied negative makes rank>1. These rows require adversarial-negative metadata "
                "in the next audit and must not be interpreted as true identity winners."
            ),
        },
        "headroom_decision": {
            "previous_s1c_s2_headroom": int(
                original["no_op_aware_headroom"]["combined_s1c_s2_s3a_recoverable"]
                - original["no_op_aware_headroom"]["s3a_unique_beyond_s1c_s2"]
            ),
            "all_s3a_combined_headroom": int(
                original["no_op_aware_headroom"]["combined_s1c_s2_s3a_recoverable"]
            ),
            "four_point_requirement": int(np.ceil(total_queries * 0.04)),
            "shortfall_to_four_points_even_under_outcome_oracle": int(
                np.ceil(total_queries * 0.04)
                - original["no_op_aware_headroom"]["combined_s1c_s2_s3a_recoverable"]
            ),
            "advance_directly_to_policy_training": False,
            "reason": (
                "The preregistered 1000-error action-space gate failed, and the all-action oracle "
                "still cannot support a four-point gain. Expand exact and positive-deficit action "
                "coverage before fitting a nonlinear action policy."
            ),
        },
    }
    (args.s3a_dir / "refined_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    trajectory_frame = pd.DataFrame(trajectory_rows)
    trajectory_frame.to_csv(args.s3a_dir / "complete_case_trajectories.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for (selector, attenuation), group in trajectory_frame.groupby(
        ["selector", "attenuation"], sort=True,
    ):
        label = f"{selector} a={float(attenuation):.2f}"
        axes[0].plot(group["step"], group["net"], marker="o", label=label)
        axes[1].plot(group["step"], group["marginal_formula_macro"], marker="o", label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Net corrections on identical six-step-complete queries")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Corrected - introduced")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Marginal Top-1 effect versus previous step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Formula-macro delta")
    axes[1].legend(fontsize=7, loc="best")
    figure.savefig(args.s3a_dir / "s3a_complete_case_trajectories.png", dpi=220)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
