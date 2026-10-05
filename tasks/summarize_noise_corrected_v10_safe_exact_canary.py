"""Summarize the V10 hard-safe, optimizer-dose-matched direct canary."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile

from summarize_noise_corrected_v9_functional_canary import (
    _all_required_metric_directions,
    _held_delta,
    _held_metric_delta,
    _load,
    _read_csv_gz,
    _sha256,
    _train_delta,
)


ARMS = (
    "full_action_view",
    "scalar_transfer_only",
    "matched_shuffled",
    "clean_control",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-action-view-dir", type=Path, required=True)
    parser.add_argument("--scalar-transfer-only-dir", type=Path, required=True)
    parser.add_argument("--matched-shuffled-dir", type=Path, required=True)
    parser.add_argument("--clean-control-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _panel(path: Path, panel: str) -> list[dict[str, object]]:
    rows = _read_csv_gz(path / "development_per_query.csv.gz")
    selected = []
    for row in rows:
        if row.get("panel") != panel:
            continue
        if not row.get("query_formula"):
            raise RuntimeError(f"{panel} ledger has no query formula: {path}")
        selected.append({
            "query_index": int(row["query_index"]),
            "query_formula": str(row["query_formula"]),
            "near": False,
            "initial_E8_rank": int(row["rank_initial"]),
            "candidate_rank": int(row["rank_final"]),
            "rank_initial": int(row["rank_initial"]),
            "rank_final": int(row["rank_final"]),
            "margin_initial": float(row["margin_initial"]),
            "margin_final": float(row["margin_final"]),
        })
    if not selected:
        raise RuntimeError(f"empty required {panel} ledger: {path}")
    return selected


def _initial_counterfactual(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [{**row, "candidate_rank": int(row["initial_E8_rank"])} for row in rows]


def _calibration_signature(decision: dict) -> dict[str, object]:
    calibration = decision["gradient_calibration"]
    return {
        "calibrated_effective_branch_scale": calibration.get(
            "calibrated_effective_branch_scale",
            calibration.get("effective_branch_scale"),
        ),
        "effective_action_to_risk_scale": calibration.get(
            "effective_action_to_risk_scale"
        ),
        "effective_global_gradient_scale": calibration.get(
            "effective_global_gradient_scale"
        ),
        "calibration_corrective_branch_mode": calibration.get(
            "calibration_corrective_branch_mode"
        ),
        "calibration_action_bank_sha256": decision.get(
            "calibration_action_bank", {}
        ).get("calibration_action_bank_sha256"),
    }


def _signal(decision: dict) -> dict[str, object]:
    values = decision.get("signal_transmission", {})
    return {
        "optimizer_action_attributable_update_fraction_p10": values.get(
            "optimizer_action_attributable_update_fraction_p10"
        ),
        "optimizer_update_restoration_all_group_targets_reached_fraction": (
            values.get(
                "optimizer_update_restoration_all_group_targets_reached_fraction"
            )
        ),
        "optimizer_update_restoration_minimum_observed_group_risk_retention": (
            values.get(
                "optimizer_update_restoration_minimum_observed_group_risk_retention"
            )
        ),
        "optimizer_update_hard_protective_floor_gate_passed": values.get(
            "optimizer_update_hard_protective_floor_gate_passed"
        ),
        "optimizer_update_exact_fraction_max_abs_error": values.get(
            "optimizer_update_exact_fraction_max_abs_error"
        ),
        "optimizer_update_exact_fraction_gate_passed": values.get(
            "optimizer_update_exact_fraction_gate_passed"
        ),
        "optimizer_update_safe_exact_max_norm_ratio": values.get(
            "optimizer_update_safe_exact_max_norm_ratio"
        ),
        "optimizer_update_safe_exact_norm_ratio_gate_passed": values.get(
            "optimizer_update_safe_exact_norm_ratio_gate_passed"
        ),
        "all_signal_gates_passed": values.get("all_signal_gates_passed"),
    }


def _strict_positive(result: dict[str, object], *, familywise: bool) -> bool:
    interval = (
        result["formula_cluster_ci_familywise"]
        if familywise else result["formula_cluster_ci_95"]
    )
    return bool(
        float(result["delta_recall1_pp"]) > 0
        and int(result["risk_net_lambda2"]) > 0
        and float(interval["ci_low_pp"]) > 0
    )


def summarize(
    paths: dict[str, Path], *, repeats: int, seed: int,
    expected_direct_contract: str = "best_action_v10_safe_exact",
    expected_action_bank_contract: str | None = None,
    status: str = "noise_corrected_v10_safe_exact_canary_complete",
    result_name: str = "V10",
    interpretation: str | None = None,
) -> dict[str, object]:
    loaded = {
        "full_action_view": _load(
            paths["full_action_view"], legacy_v8=False,
        ),
        "scalar_transfer_only": _load(
            paths["scalar_transfer_only"], legacy_v8=False,
        ),
        "matched_shuffled": _load(
            paths["matched_shuffled"], legacy_v8=True,
        ),
        "clean_control": _load(paths["clean_control"], legacy_v8=True),
    }
    decisions = {name: value["decision"] for name, value in loaded.items()}
    expected_arm = {
        "full_action_view": "routed_direct",
        "scalar_transfer_only": "routed_direct",
        "matched_shuffled": "shuffled_action_control",
        "clean_control": "clean_control",
    }
    for name, arm in expected_arm.items():
        decision = decisions[name]
        if decision.get("arm") != arm:
            raise RuntimeError(f"{result_name} arm label drifted: {name}")
        configuration = decision.get("configuration", {})
        if configuration.get("direct_contract") != expected_direct_contract:
            raise RuntimeError(f"{result_name} direct contract drifted: {name}")
        if (
            expected_action_bank_contract is not None
            and configuration.get("action_bank_contract")
            != expected_action_bank_contract
        ):
            raise RuntimeError(f"{result_name} action-bank contract drifted: {name}")
        if configuration.get("inner_holdout_unit") != "formula":
            raise RuntimeError(
                f"{result_name} inner holdout is not formula-disjoint: {name}"
            )
        if int(configuration.get("inner_holdout_fold", -1)) < 0:
            raise RuntimeError(
                f"{result_name} inner formula holdout is disabled: {name}"
            )
        if configuration.get("calibration_corrective_branch_mode") != (
            "full_action_view"
        ):
            raise RuntimeError(
                f"{result_name} common calibration mode drifted: {name}"
            )
        if decision.get("contracts", {}).get(
            "teacher_embedding_or_distillation_target_used"
        ) is not False:
            raise RuntimeError(
                f"{result_name} used a teacher or distillation target"
            )

    if decisions["full_action_view"].get("corrective_branch_mode") != (
        "full_action_view"
    ):
        raise RuntimeError(f"{result_name} full arm lost its action view")
    if decisions["scalar_transfer_only"].get("corrective_branch_mode") != (
        "scalar_transfer_only"
    ):
        raise RuntimeError(f"{result_name} scalar arm retained the wrong policy")
    scalar_scale = decisions["scalar_transfer_only"]["gradient_calibration"].get(
        "effective_branch_scale", {}
    )
    if scalar_scale.get("payload") != 0.0 or scalar_scale.get("consistency") != 0.0:
        raise RuntimeError(
            f"{result_name} scalar arm retained a disabled live branch"
        )

    signatures = {
        name: _calibration_signature(decisions[name]) for name in ARMS
    }
    common_calibration = bool(all(
        signatures[name] == signatures["full_action_view"] for name in ARMS[1:]
    ))
    if not common_calibration:
        raise RuntimeError(
            f"{result_name} calibration was not frozen across arms: {signatures}"
        )

    inner = {name: _panel(paths[name], "inner_held_corrective") for name in ARMS}
    train = {name: loaded[name]["train"] for name in ARMS}
    held = {name: loaded[name]["held"] for name in ARMS}
    comparisons = (
        ("full_vs_scalar", "scalar_transfer_only", "full_action_view"),
        ("full_vs_shuffled", "matched_shuffled", "full_action_view"),
        ("full_vs_clean", "clean_control", "full_action_view"),
        ("scalar_vs_clean", "clean_control", "scalar_transfer_only"),
    )
    outer_results = {}
    outer_metrics = {}
    inner_results = {}
    train_results = {}
    for offset, (label, baseline, candidate) in enumerate(comparisons):
        outer_results[label] = _held_delta(
            held[baseline], held[candidate], repeats=repeats,
            seed=seed + offset * 10, familywise_hypotheses=len(comparisons),
        )
        outer_metrics[label] = _held_metric_delta(
            decisions[baseline], decisions[candidate]
        )
        inner_results[label] = _held_delta(
            inner[baseline], inner[candidate], repeats=repeats,
            seed=seed + 100 + offset * 10,
            familywise_hypotheses=len(comparisons),
        )
        train_results[label] = _train_delta(train[baseline], train[candidate])

    inner_vs_initial = {
        name: _held_delta(
            _initial_counterfactual(inner[name]), inner[name],
            repeats=repeats, seed=seed + 500 + offset,
            familywise_hypotheses=len(ARMS),
        )
        for offset, name in enumerate(ARMS)
    }
    signals = {
        name: _signal(decisions[name])
        for name in ("full_action_view", "scalar_transfer_only", "matched_shuffled")
    }
    actual_fractions = {
        name: float(values["optimizer_action_attributable_update_fraction_p10"])
        for name, values in signals.items()
    }
    optimizer_fraction_matched = bool(
        max(actual_fractions.values()) - min(actual_fractions.values()) <= 2e-6
        and all(abs(value - 0.25) <= 2e-6 for value in actual_fractions.values())
    )
    optimizer_safety = bool(all(
        values["all_signal_gates_passed"] is True
        and values["optimizer_update_hard_protective_floor_gate_passed"] is True
        and values["optimizer_update_exact_fraction_gate_passed"] is True
        and float(values[
            "optimizer_update_restoration_all_group_targets_reached_fraction"
        ]) == 1.0
        and float(values[
            "optimizer_update_restoration_minimum_observed_group_risk_retention"
        ]) >= 0.90 - 1e-6
        for values in signals.values()
    ))
    primary_inner = inner_results["full_vs_scalar"]
    primary_outer = outer_results["full_vs_scalar"]
    clean_inner = inner_results["full_vs_clean"]
    clean_outer = outer_results["full_vs_clean"]
    shuffled_inner = inner_results["full_vs_shuffled"]
    shuffled_outer = outer_results["full_vs_shuffled"]
    all_metric_gates = {
        label: _all_required_metric_directions(outer_metrics[label])
        for label in ("full_vs_scalar", "full_vs_shuffled", "full_vs_clean")
    }
    advance = bool(
        common_calibration
        and optimizer_fraction_matched
        and optimizer_safety
        and _strict_positive(primary_inner, familywise=False)
        and _strict_positive(clean_inner, familywise=True)
        and _strict_positive(shuffled_inner, familywise=True)
        and _strict_positive(primary_outer, familywise=False)
        and _strict_positive(clean_outer, familywise=True)
        and _strict_positive(shuffled_outer, familywise=True)
        and all(all_metric_gates.values())
    )
    return {
        "status": status,
        "formal": False,
        "training_or_embedding_promotion": False,
        "newly_trained_arms": list(ARMS),
        "input_artifacts": {
            name: {
                "path": loaded[name]["path"],
                "sha256": loaded[name]["artifact_sha256"],
            }
            for name in ARMS
        },
        "common_calibration": {
            "signatures": signatures,
            "all_arms_exactly_matched": common_calibration,
        },
        "optimizer_space_dose": {
            "action_attributable_fraction_p10": actual_fractions,
            "matched_to_0p25_within_2e_6": optimizer_fraction_matched,
            "signal_and_hard_safety_passed": optimizer_safety,
            "per_arm": signals,
        },
        "formula_disjoint_inner_corrective": {
            "arm_vs_initial_E8": inner_vs_initial,
            "cross_arm": inner_results,
        },
        "training_corrective": train_results,
        "outer_formula_held": outer_results,
        "outer_full_metric_deltas": outer_metrics,
        "gates": {
            "common_calibration_exact": common_calibration,
            "actual_optimizer_corrective_fraction_matched": (
                optimizer_fraction_matched
            ),
            "hard_protective_floor_and_signal_pass": optimizer_safety,
            "full_beats_scalar_on_inner_formula_holdout": _strict_positive(
                primary_inner, familywise=False,
            ),
            "full_beats_clean_on_inner_formula_holdout": _strict_positive(
                clean_inner, familywise=True,
            ),
            "full_beats_matched_shuffle_on_inner_formula_holdout": (
                _strict_positive(shuffled_inner, familywise=True)
            ),
            "full_beats_scalar_on_outer_formula_holdout": _strict_positive(
                primary_outer, familywise=False,
            ),
            "full_beats_clean_on_outer_formula_holdout": _strict_positive(
                clean_outer, familywise=True,
            ),
            "full_beats_matched_shuffle_on_outer_formula_holdout": (
                _strict_positive(shuffled_outer, familywise=True)
            ),
            "full_improves_every_required_outer_metric": all_metric_gates,
            "advance_to_larger_direct_finetuning_pilot": advance,
        },
        "interpretation": interpretation or (
            "V10 keeps the frozen best-action bank and direct E4/E8 objective, "
            "but makes the protective floor hard, matches the materialized "
            "optimizer-space corrective fraction, freezes calibration across "
            "arms, and requires formula-disjoint transfer before outer-held use."
        ),
        "claim_limit": (
            "Bounded direct-finetuning causal canary; not a 4-5 pp claim and "
            "not a promoted shared encoder."
        ),
    }


def main() -> None:
    args = arguments()
    paths = {
        "full_action_view": args.full_action_view_dir,
        "scalar_transfer_only": args.scalar_transfer_only_dir,
        "matched_shuffled": args.matched_shuffled_dir,
        "clean_control": args.clean_control_dir,
    }
    report = summarize(paths, repeats=args.bootstrap_resamples, seed=args.seed)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
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
