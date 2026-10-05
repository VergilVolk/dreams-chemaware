"""Summarize V9's matched-budget direct-injection branch decomposition.

Only the full-action-view and scalar-transfer arms are newly trained.  Frozen
V8 shuffled and clean controls are admitted only when the new full arm exactly
bridges the frozen V8 full arm on every clean-query rank ledger.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile


PRIMARY_ARMS = ("full_action_view", "scalar_transfer_only")
BRIDGE_ARMS = ("v8_full_action_view", "v8_shuffled", "v8_clean")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-action-view-dir", type=Path, required=True)
    parser.add_argument("--scalar-transfer-only-dir", type=Path, required=True)
    parser.add_argument("--v8-root", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, *, legacy_v8: bool) -> dict[str, object]:
    decision_path = path / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(f"missing decision: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("formal") is not False:
        raise RuntimeError(f"development canary accepted a formal arm: {path}")
    held_path = path / "held_per_query.csv.gz"
    expected_held_sha = decision.get("provenance", {}).get(
        "held_per_query_sha256"
    )
    if expected_held_sha is not None and _sha256(held_path) != expected_held_sha:
        raise RuntimeError(f"held ledger checksum differs from decision: {path}")
    held_raw = _read_csv_gz(held_path)
    held_required = {
        "query_index", "query_formula", "near", "initial_E8_rank",
        "candidate_rank",
    }
    if held_raw and (missing := held_required - set(held_raw[0])):
        raise RuntimeError(f"held table misses {sorted(missing)}: {path}")
    held = [{
        "query_index": int(row["query_index"]),
        "query_formula": str(row["query_formula"]),
        "near": _parse_bool(row["near"]),
        "initial_E8_rank": int(row["initial_E8_rank"]),
        "candidate_rank": int(row["candidate_rank"]),
    } for row in held_raw]
    development_raw = _read_csv_gz(path / "development_per_query.csv.gz")
    train = [{
        "query_index": int(row["query_index"]),
        "rank_initial": int(row["rank_initial"]),
        "rank_final": int(row["rank_final"]),
        "margin_initial": float(row["margin_initial"]),
        "margin_final": float(row["margin_final"]),
    } for row in development_raw if row.get("panel") == "train_corrective"]
    if not held or not train:
        raise RuntimeError(f"empty held or train-corrective ledger: {path}")
    if not legacy_v8 and decision.get("corrective_branch_mode") not in PRIMARY_ARMS:
        raise RuntimeError(f"V9 arm has no registered branch mode: {path}")
    return {
        "decision": decision,
        "held": held,
        "train": train,
        "path": str(path),
        "artifact_sha256": {
            "decision.json": _sha256(decision_path),
            "held_per_query.csv.gz": _sha256(held_path),
            "development_per_query.csv.gz": _sha256(
                path / "development_per_query.csv.gz"
            ),
        },
    }


def _aligned(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    keys: tuple[str, ...],
) -> None:
    if len(baseline) != len(candidate) or any(
        tuple(left[key] for key in keys) != tuple(right[key] for key in keys)
        for left, right in zip(baseline, candidate)
    ):
        raise RuntimeError(f"paired arm ledgers are not aligned on {keys}")


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _formula_cluster_ci(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    repeats: int,
    seed: int,
    confidence: float,
    near_only: bool = False,
) -> dict[str, float | int]:
    if repeats < 100:
        raise ValueError("formula bootstrap needs at least 100 resamples")
    _aligned(
        baseline, candidate,
        ("query_index", "query_formula", "near", "initial_E8_rank"),
    )
    grouped: dict[str, list[int]] = {}
    selected = 0
    for left, right in zip(baseline, candidate):
        if near_only and not bool(right["near"]):
            continue
        value = int(int(right["candidate_rank"]) == 1) - int(
            int(left["candidate_rank"]) == 1
        )
        bucket = grouped.setdefault(str(right["query_formula"]), [0, 0])
        bucket[0] += value
        bucket[1] += 1
        selected += 1
    if not grouped or not selected:
        raise RuntimeError("formula bootstrap selection is empty")
    clusters = sorted(grouped)
    rng = random.Random(int(seed))
    boot = []
    for _ in range(repeats):
        value_sum = 0
        count_sum = 0
        for _ in clusters:
            values = grouped[clusters[rng.randrange(len(clusters))]]
            value_sum += values[0]
            count_sum += values[1]
        boot.append(100.0 * value_sum / count_sum)
    alpha = (1.0 - float(confidence)) / 2.0
    point = 100.0 * sum(value[0] for value in grouped.values()) / selected
    return {
        "queries": selected,
        "formula_clusters": len(grouped),
        "delta_pp": float(point),
        "confidence": float(confidence),
        "ci_low_pp": float(_quantile(boot, alpha)),
        "ci_high_pp": float(_quantile(boot, 1.0 - alpha)),
        "resamples": int(repeats),
    }


def _held_delta(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    repeats: int,
    seed: int,
    familywise_hypotheses: int = 3,
) -> dict[str, object]:
    keys = ("query_index", "query_formula", "near", "initial_E8_rank")
    _aligned(baseline, candidate, keys)
    before = [int(row["candidate_rank"]) for row in baseline]
    after = [int(row["candidate_rank"]) for row in candidate]
    corrected = [left > 1 and right == 1 for left, right in zip(before, after)]
    introduced = [left == 1 and right > 1 for left, right in zip(before, after)]
    near = [bool(row["near"]) for row in candidate]
    near_count = sum(near)
    confidence = 1.0 - 0.05 / familywise_hypotheses
    return {
        "queries": len(after),
        "delta_recall1_pp": 100.0 * (
            sum(value == 1 for value in after)
            - sum(value == 1 for value in before)
        ) / len(after),
        "delta_mrr_pp": 100.0 * sum(
            1.0 / right - 1.0 / left for left, right in zip(before, after)
        ) / len(after),
        "mean_rank_change": sum(
            right - left for left, right in zip(before, after)
        ) / len(after),
        "corrected": sum(corrected),
        "introduced": sum(introduced),
        "risk_net_lambda2": sum(corrected) - 2 * sum(introduced),
        "near_delta_recall1_pp": (
            100.0 * sum(
                int(right == 1) - int(left == 1)
                for left, right, selected in zip(before, after, near) if selected
            ) / near_count if near_count else 0.0
        ),
        "formula_cluster_ci_95": _formula_cluster_ci(
            baseline, candidate, repeats=repeats, seed=seed,
            confidence=0.95,
        ),
        "formula_cluster_ci_familywise": _formula_cluster_ci(
            baseline, candidate, repeats=repeats, seed=seed + 1000,
            confidence=confidence,
        ),
        "near_formula_cluster_ci_95": (
            _formula_cluster_ci(
                baseline, candidate, repeats=repeats, seed=seed + 2000,
                confidence=0.95, near_only=True,
            ) if near_count else None
        ),
    }


def _train_delta(
    baseline: list[dict[str, object]], candidate: list[dict[str, object]],
) -> dict[str, float | int]:
    _aligned(baseline, candidate, ("query_index", "rank_initial"))
    before = [int(row["rank_final"]) for row in baseline]
    after = [int(row["rank_final"]) for row in candidate]
    corrected = sum(left > 1 and right == 1 for left, right in zip(before, after))
    introduced = sum(left == 1 and right > 1 for left, right in zip(before, after))
    return {
        "queries": len(after),
        "delta_recall1_pp": 100.0 * sum(
            int(right == 1) - int(left == 1)
            for left, right in zip(before, after)
        ) / len(after),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "mean_margin_change_vs_baseline_arm": sum(
            float(right["margin_final"]) - float(left["margin_final"])
            for left, right in zip(baseline, candidate)
        ) / len(after),
    }


def _held_metric_delta(baseline: dict, candidate: dict) -> dict[str, object]:
    base = baseline["evaluation"]["development_held_graph"]["candidate"]
    cand = candidate["evaluation"]["development_held_graph"]["candidate"]

    def retrieval_section(name: str) -> dict[str, float]:
        output: dict[str, float] = {}
        probability_keys = (
            "mrr", "macro_query_auroc", "macro_query_auprc",
            "recall@1", "recall@2", "recall@3", "recall@5",
            "recall@10", "recall@20",
        )
        for key in probability_keys:
            output[f"{key}_delta_pp"] = 100.0 * (
                float(cand[name][key]) - float(base[name][key])
            )
        for key in (
            "mean_positive_vs_best_negative_margin", "mean_top1_top2_gap",
            "mean_signed_top1_top2_gap",
        ):
            output[f"{key}_change"] = (
                float(cand[name][key]) - float(base[name][key])
            )
        return output

    retrieval = retrieval_section("retrieval")
    for key in ("mean_rank", "median_rank"):
        retrieval[f"{key}_change"] = (
            float(cand["retrieval"][key]) - float(base["retrieval"][key])
        )
    near = retrieval_section("near_subset")
    for key in ("mean_rank", "median_rank"):
        near[f"{key}_change"] = (
            float(cand["near_subset"][key]) - float(base["near_subset"][key])
        )
    return {
        "retrieval": retrieval,
        "near_subset": near,
        "micro_candidate": {
            f"{key}_delta_pp": 100.0 * (
                float(cand["micro_candidate"][key])
                - float(base["micro_candidate"][key])
            ) for key in ("auroc", "auprc")
        },
        "massspecgym_10ppm_pooled_pairwise": {
            f"{key}_delta_pp": 100.0 * (
                float(cand["massspecgym_10ppm_pooled_pairwise"][key])
                - float(base["massspecgym_10ppm_pooled_pairwise"][key])
            ) for key in ("auroc", "auprc")
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            f"{key}_delta_pp": 100.0 * (
                float(cand["massspecgym_mh_10ppm_pooled_pairwise"][key])
                - float(base["massspecgym_mh_10ppm_pooled_pairwise"][key])
            ) for key in ("auroc", "auprc")
        },
    }


def _bridge_report(current: dict[str, object], historical: dict[str, object]) -> dict[str, object]:
    current_held = current["held"]
    historical_held = historical["held"]
    _aligned(
        historical_held, current_held,
        ("query_index", "query_formula", "near", "initial_E8_rank"),
    )
    current_train = current["train"]
    historical_train = historical["train"]
    _aligned(historical_train, current_train, ("query_index", "rank_initial"))
    held_rank_mismatches = sum(
        int(left["candidate_rank"]) != int(right["candidate_rank"])
        for left, right in zip(historical_held, current_held)
    )
    train_rank_mismatches = sum(
        int(left["rank_final"]) != int(right["rank_final"])
        for left, right in zip(historical_train, current_train)
    )
    maximum_train_margin_error = max(
        abs(float(left["margin_final"]) - float(right["margin_final"]))
        for left, right in zip(historical_train, current_train)
    )
    passed = bool(
        held_rank_mismatches == 0
        and train_rank_mismatches == 0
        and maximum_train_margin_error <= 1e-6
    )
    return {
        "held_queries": len(current_held),
        "train_corrective_queries": len(current_train),
        "held_candidate_rank_mismatches": held_rank_mismatches,
        "train_candidate_rank_mismatches": train_rank_mismatches,
        "maximum_train_margin_abs_error": maximum_train_margin_error,
        "maximum_allowed_train_margin_abs_error": 1e-6,
        "passed": passed,
    }


def _signal(decision: dict) -> dict[str, object]:
    values = decision.get("signal_transmission", {})
    return {
        "action_retention_p10": values.get("action_retention_p10"),
        "optimizer_action_attributable_update_fraction_p10": values.get(
            "optimizer_action_attributable_update_fraction_p10"
        ),
        "optimizer_update_restoration_target_coverage_gate_passed": values.get(
            "optimizer_update_restoration_target_coverage_gate_passed"
        ),
        "legacy_90pct_end_to_end_loss_reproduced": values.get(
            "legacy_90pct_end_to_end_loss_reproduced"
        ),
        "all_signal_gates_passed": values.get("all_signal_gates_passed"),
    }


def _all_required_metric_directions(delta: dict[str, object]) -> bool:
    strict_keys = {
        "retrieval": {
            "mrr_delta_pp", "macro_query_auroc_delta_pp",
            "macro_query_auprc_delta_pp", "recall@1_delta_pp",
            "mean_positive_vs_best_negative_margin_change",
            "mean_top1_top2_gap_change", "mean_signed_top1_top2_gap_change",
        },
        "near_subset": {
            "mrr_delta_pp", "macro_query_auroc_delta_pp",
            "macro_query_auprc_delta_pp", "recall@1_delta_pp",
            "mean_positive_vs_best_negative_margin_change",
            "mean_top1_top2_gap_change", "mean_signed_top1_top2_gap_change",
        },
        "micro_candidate": {"auroc_delta_pp", "auprc_delta_pp"},
        "massspecgym_10ppm_pooled_pairwise": {
            "auroc_delta_pp", "auprc_delta_pp",
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            "auroc_delta_pp", "auprc_delta_pp",
        },
    }
    for section, required_positive in strict_keys.items():
        for name, value in delta[section].items():
            if name.endswith("_change") and name.startswith((
                "mean_rank", "median_rank",
            )):
                if float(value) > 0:
                    return False
            elif name in required_positive and float(value) <= 0:
                return False
            elif name not in required_positive and float(value) < 0:
                return False
    return True


def summarize(
    paths: dict[str, Path], *, repeats: int, seed: int,
) -> dict[str, object]:
    loaded = {
        "full_action_view": _load(paths["full_action_view"], legacy_v8=False),
        "scalar_transfer_only": _load(
            paths["scalar_transfer_only"], legacy_v8=False,
        ),
        "v8_full_action_view": _load(
            paths["v8_full_action_view"], legacy_v8=True,
        ),
        "v8_shuffled": _load(paths["v8_shuffled"], legacy_v8=True),
        "v8_clean": _load(paths["v8_clean"], legacy_v8=True),
    }
    decision = {name: value["decision"] for name, value in loaded.items()}
    expected_legacy_arm = {
        "v8_full_action_view": "routed_direct",
        "v8_shuffled": "shuffled_action_control",
        "v8_clean": "clean_control",
    }
    for name, expected in expected_legacy_arm.items():
        if decision[name].get("arm") != expected:
            raise RuntimeError(f"frozen V8 control arm drifted: {name}")
    for name in PRIMARY_ARMS:
        if decision[name].get("arm") != "routed_direct":
            raise RuntimeError(f"V9 primary arm is not routed direct: {name}")
        if decision[name].get("corrective_branch_mode") != name:
            raise RuntimeError(f"V9 corrective branch mode drifted: {name}")
        observed_locality = decision[name].get(
            "corrective_gradient_locality",
            decision[name].get("contracts", {}).get(
                "corrective_gradient_locality"
            ),
        )
        if observed_locality != "query_action_only":
            raise RuntimeError(f"V9 arm escaped query-local injection: {name}")
        if decision[name].get("contracts", {}).get(
            "teacher_embedding_or_distillation_target_used"
        ) is not False:
            raise RuntimeError("V9 used a teacher or distillation target")
    full_calibration = decision["full_action_view"]["gradient_calibration"]
    scalar_calibration = decision["scalar_transfer_only"]["gradient_calibration"]
    if scalar_calibration.get("effective_branch_scale", {}).get("payload") != 0.0:
        raise RuntimeError("scalar-only arm retained payload gradient")
    if scalar_calibration.get("effective_branch_scale", {}).get("consistency") != 0.0:
        raise RuntimeError("scalar-only arm retained consistency gradient")
    if not all(
        float(full_calibration.get("effective_branch_scale", {}).get(name, 0.0)) > 0
        for name in ("payload", "consistency")
    ):
        raise RuntimeError("full action-view arm lost a live action branch")
    budget = {
        name: decision[name]["gradient_calibration"].get(
            "effective_dense_corrective_to_risk_gradient_ratio"
        ) for name in PRIMARY_ARMS
    }
    budget_matched = bool(
        all(value is not None and math.isfinite(float(value)) for value in budget.values())
        and abs(float(budget["full_action_view"]) - float(
            budget["scalar_transfer_only"]
        )) <= 1e-8
    )
    bridge = _bridge_report(
        loaded["full_action_view"], loaded["v8_full_action_view"]
    )
    primary = {
        "full_action_view_vs_scalar_transfer_only": _held_delta(
            loaded["scalar_transfer_only"]["held"],
            loaded["full_action_view"]["held"],
            repeats=repeats, seed=seed,
        ),
    }
    primary_train = {
        "full_action_view_vs_scalar_transfer_only": _train_delta(
            loaded["scalar_transfer_only"]["train"],
            loaded["full_action_view"]["train"],
        ),
    }
    secondary: dict[str, object] = {}
    secondary_train: dict[str, object] = {}
    secondary_metrics: dict[str, object] = {}
    if bridge["passed"]:
        comparisons = (
            ("full_action_view_vs_v8_shuffled", "v8_shuffled", "full_action_view"),
            ("full_action_view_vs_v8_clean", "v8_clean", "full_action_view"),
            ("scalar_transfer_only_vs_v8_clean", "v8_clean", "scalar_transfer_only"),
        )
        for offset, (label, baseline, candidate) in enumerate(comparisons, start=1):
            secondary[label] = _held_delta(
                loaded[baseline]["held"], loaded[candidate]["held"],
                repeats=repeats, seed=seed + 10 * offset,
            )
            secondary_train[label] = _train_delta(
                loaded[baseline]["train"], loaded[candidate]["train"],
            )
            secondary_metrics[label] = _held_metric_delta(
                decision[baseline], decision[candidate],
            )
    primary_metrics = {
        "full_action_view_vs_scalar_transfer_only": _held_metric_delta(
            decision["scalar_transfer_only"], decision["full_action_view"],
        )
    }
    primary_result = primary["full_action_view_vs_scalar_transfer_only"]
    primary_train_result = primary_train[
        "full_action_view_vs_scalar_transfer_only"
    ]
    branch_addition_gate = bool(
        primary_result["delta_recall1_pp"] > 0
        and primary_result["risk_net_lambda2"] > 0
        and primary_result["formula_cluster_ci_95"]["ci_low_pp"] > 0
    )
    functional_conversion_gate = bool(
        primary_train_result["delta_recall1_pp"] > 0
        and primary_train_result["risk_net_lambda2"] > 0
    )
    signal = {name: _signal(decision[name]) for name in PRIMARY_ARMS}
    signal_gate = bool(all(
        signal[name]["all_signal_gates_passed"] is True
        and signal[name]["legacy_90pct_end_to_end_loss_reproduced"] is False
        for name in PRIMARY_ARMS
    ))
    primary_all_metrics_gate = _all_required_metric_directions(
        primary_metrics["full_action_view_vs_scalar_transfer_only"]
    )
    semantic_gate = False
    clean_benefit_gate = False
    full_vs_clean_all_metrics_gate = False
    bridged_functional_conversion_gate = False
    if bridge["passed"]:
        semantic = secondary["full_action_view_vs_v8_shuffled"]
        clean_benefit = secondary["full_action_view_vs_v8_clean"]
        semantic_gate = bool(
            semantic["delta_recall1_pp"] > 0
            and semantic["risk_net_lambda2"] > 0
            and semantic["formula_cluster_ci_familywise"]["ci_low_pp"] > 0
        )
        clean_benefit_gate = bool(
            clean_benefit["delta_recall1_pp"] > 0
            and clean_benefit["risk_net_lambda2"] > 0
            and clean_benefit["formula_cluster_ci_familywise"]["ci_low_pp"] > 0
        )
        full_vs_clean_all_metrics_gate = _all_required_metric_directions(
            secondary_metrics["full_action_view_vs_v8_clean"]
        )
        bridged_train = secondary_train["full_action_view_vs_v8_clean"]
        bridged_functional_conversion_gate = bool(
            bridged_train["delta_recall1_pp"] > 0
            and bridged_train["risk_net_lambda2"] > 0
        )
    return {
        "status": "noise_corrected_v9_functional_canary_complete",
        "formal": False,
        "training_or_embedding_promotion": False,
        "newly_trained_arms": list(PRIMARY_ARMS),
        "frozen_v8_bridge_controls": list(BRIDGE_ARMS),
        "input_artifacts": {
            name: {
                "path": values["path"],
                "sha256": values["artifact_sha256"],
            }
            for name, values in loaded.items()
        },
        "corrective_gradient_budget": {
            "effective_dense_corrective_to_risk_ratio": budget,
            "matched_within_absolute_1e_8": budget_matched,
        },
        "historical_control_bridge": bridge,
        "primary_same_source_comparison": primary,
        "primary_train_corrective_clean_query_conversion": primary_train,
        "primary_full_metric_deltas": primary_metrics,
        "bridged_secondary_comparisons": secondary,
        "bridged_train_corrective_clean_query_conversion": secondary_train,
        "bridged_secondary_full_metric_deltas": secondary_metrics,
        "signal_transmission": signal,
        "gates": {
            "corrective_gradient_budget_matched": budget_matched,
            "full_action_view_adds_clean_query_value_over_scalar_transfer": (
                branch_addition_gate
            ),
            "full_action_view_improves_every_required_metric_over_scalar": (
                primary_all_metrics_gate
            ),
            "action_view_functionally_converts_on_training_clean_queries": (
                functional_conversion_gate
            ),
            "new_full_arm_exactly_bridges_v8_full_arm": bridge["passed"],
            "routed_full_action_view_beats_matched_v8_shuffle": semantic_gate,
            "routed_full_action_view_beats_v8_clean_control": clean_benefit_gate,
            "full_action_view_improves_every_required_metric_over_clean": (
                full_vs_clean_all_metrics_gate
            ),
            "full_action_view_beats_clean_on_training_clean_queries": (
                bridged_functional_conversion_gate
            ),
            "optimizer_signal_and_restoration_safety": signal_gate,
            "advance_to_larger_direct_finetuning_pilot": bool(
                budget_matched
                and branch_addition_gate
                and primary_all_metrics_gate
                and functional_conversion_gate
                and bridge["passed"]
                and semantic_gate
                and clean_benefit_gate
                and full_vs_clean_all_metrics_gate
                and bridged_functional_conversion_gate
                and signal_gate
            ),
        },
        "interpretation": (
            "The primary comparison changes only whether payload and symmetric "
            "live action-view gradients are present. Both arms retain scalar "
            "margin transfer and use the same calibrated corrective-to-risk "
            "gradient budget. Historical controls are conditional on an exact "
            "clean-rank bridge and cannot rescue a failed primary comparison."
        ),
        "claim_limit": (
            "Bounded direct-finetuning development ablation; not a 4-5 pp "
            "performance claim and not a promoted shared encoder."
        ),
    }


def main() -> None:
    args = arguments()
    v8_arms = args.v8_root / "arms"
    paths = {
        "full_action_view": args.full_action_view_dir,
        "scalar_transfer_only": args.scalar_transfer_only_dir,
        "v8_full_action_view": v8_arms / "query_local_routed",
        "v8_shuffled": v8_arms / "query_local_shuffled",
        "v8_clean": v8_arms / "clean_control",
    }
    report = summarize(
        paths, repeats=args.bootstrap_resamples, seed=args.seed,
    )
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
