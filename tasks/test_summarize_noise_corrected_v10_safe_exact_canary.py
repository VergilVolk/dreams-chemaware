"""CPU-only synthetic tests for the V10 causal summary gates."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import tempfile

from summarize_noise_corrected_v10_safe_exact_canary import summarize


FORMULAS = tuple(f"F{index // 2}" for index in range(12))
INITIAL = (2,) * len(FORMULAS)


def _metrics(ranks: tuple[int, ...]) -> dict[str, object]:
    recall = lambda cutoff: sum(rank <= cutoff for rank in ranks) / len(ranks)
    summary = {
        "queries": len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": float(sorted(ranks)[len(ranks) // 2]),
        "macro_query_auroc": 0.70 + 0.02 * recall(1),
        "macro_query_auprc": 0.60 + 0.02 * recall(1),
        "mean_positive_vs_best_negative_margin": -0.1 + 0.02 * recall(1),
        "mean_top1_top2_gap": 0.1 + 0.02 * recall(1),
        "mean_signed_top1_top2_gap": -0.1 + 0.02 * recall(1),
        **{
            f"recall@{cutoff}": recall(cutoff)
            for cutoff in (1, 2, 3, 5, 10, 20)
        },
    }
    return {
        "retrieval": dict(summary),
        "near_subset": dict(summary),
        "micro_candidate": {
            "auroc": 0.75 + 0.02 * recall(1),
            "auprc": 0.65 + 0.02 * recall(1),
        },
        "massspecgym_10ppm_pooled_pairwise": {
            "auroc": 0.75 + 0.02 * recall(1),
            "auprc": 0.65 + 0.02 * recall(1),
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            "auroc": 0.75 + 0.02 * recall(1),
            "auprc": 0.65 + 0.02 * recall(1),
        },
    }


def _write_arm(
    root: Path,
    *,
    ranks: tuple[int, ...],
    mode: str,
    arm: str,
    action_fraction: float = 0.25,
) -> None:
    root.mkdir(parents=True)
    calibrated_scale = {
        "transfer": 1.0,
        "payload": 0.2,
        "consistency": 0.3,
        "robust": 0.1,
        "harmful": 0.1,
    }
    effective_scale = dict(calibrated_scale)
    if mode == "scalar_transfer_only":
        effective_scale["payload"] = 0.0
        effective_scale["consistency"] = 0.0
    decision = {
        "formal": False,
        "arm": arm,
        "corrective_branch_mode": mode,
        "configuration": {
            "direct_contract": "best_action_v10_safe_exact",
            "inner_holdout_unit": "formula",
            "inner_holdout_fold": 0,
            "calibration_corrective_branch_mode": "full_action_view",
        },
        "contracts": {
            "teacher_embedding_or_distillation_target_used": False,
        },
        "gradient_calibration": {
            "calibrated_effective_branch_scale": calibrated_scale,
            "effective_branch_scale": effective_scale,
            "effective_action_to_risk_scale": 3.0,
            "effective_global_gradient_scale": 0.01,
            "calibration_corrective_branch_mode": "full_action_view",
        },
        "calibration_action_bank": {
            "calibration_action_bank_sha256": "same-targeted-bank",
        },
        "signal_transmission": {
            "optimizer_action_attributable_update_fraction_p10": action_fraction,
            "optimizer_update_restoration_all_group_targets_reached_fraction": 1.0,
            "optimizer_update_restoration_minimum_observed_group_risk_retention": 0.9,
            "optimizer_update_hard_protective_floor_gate_passed": True,
            "optimizer_update_exact_fraction_max_abs_error": abs(
                action_fraction - 0.25
            ),
            "optimizer_update_exact_fraction_gate_passed": (
                abs(action_fraction - 0.25) <= 2e-6
            ),
            "optimizer_update_safe_exact_max_norm_ratio": 1.2,
            "optimizer_update_safe_exact_norm_ratio_gate_passed": True,
            "all_signal_gates_passed": abs(action_fraction - 0.25) <= 2e-6,
        },
        "evaluation": {
            "development_held_graph": {"candidate": _metrics(ranks)},
        },
    }
    (root / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    with gzip.open(
        root / "held_per_query.csv.gz", "wt", encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "query_index", "query_formula", "near", "initial_E8_rank",
            "candidate_rank",
        ))
        writer.writeheader()
        for index, rank in enumerate(ranks):
            writer.writerow({
                "query_index": index,
                "query_formula": FORMULAS[index],
                "near": "True",
                "initial_E8_rank": INITIAL[index],
                "candidate_rank": rank,
            })
    with gzip.open(
        root / "development_per_query.csv.gz",
        "wt", encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "panel", "query_index", "query_formula", "rank_initial",
            "rank_final", "margin_initial", "margin_final",
        ))
        writer.writeheader()
        for panel in ("train_corrective", "inner_held_corrective"):
            for index, rank in enumerate(ranks):
                writer.writerow({
                    "panel": panel,
                    "query_index": index,
                    "query_formula": FORMULAS[index],
                    "rank_initial": INITIAL[index],
                    "rank_final": rank,
                    "margin_initial": -0.1,
                    "margin_final": 0.1 if rank == 1 else -0.1,
                })


def _fixture(root: Path, *, scalar_fraction: float = 0.25) -> dict[str, Path]:
    full = (1,) * len(FORMULAS)
    scalar = tuple(1 if index % 2 == 0 else 2 for index in range(len(FORMULAS)))
    initial = INITIAL
    paths = {
        "full_action_view": root / "full",
        "scalar_transfer_only": root / "scalar",
        "matched_shuffled": root / "shuffled",
        "clean_control": root / "clean",
    }
    _write_arm(
        paths["full_action_view"], ranks=full,
        mode="full_action_view", arm="routed_direct",
    )
    _write_arm(
        paths["scalar_transfer_only"], ranks=scalar,
        mode="scalar_transfer_only", arm="routed_direct",
        action_fraction=scalar_fraction,
    )
    _write_arm(
        paths["matched_shuffled"], ranks=initial,
        mode="full_action_view", arm="shuffled_action_control",
    )
    _write_arm(
        paths["clean_control"], ranks=initial,
        mode="full_action_view", arm="clean_control",
    )
    return paths


def test_v10_summary_requires_inner_formula_transfer_and_actual_update_match() -> None:
    with tempfile.TemporaryDirectory() as temp:
        report = summarize(_fixture(Path(temp)), repeats=200, seed=11)
    assert report["common_calibration"]["all_arms_exactly_matched"] is True
    assert report["optimizer_space_dose"][
        "matched_to_0p25_within_2e_6"
    ] is True
    assert report["gates"][
        "full_beats_scalar_on_inner_formula_holdout"
    ] is True
    assert report["gates"]["advance_to_larger_direct_finetuning_pilot"] is True


def test_v10_summary_rejects_preoptimizer_only_budget_match() -> None:
    with tempfile.TemporaryDirectory() as temp:
        report = summarize(
            _fixture(Path(temp), scalar_fraction=0.2334),
            repeats=100,
            seed=17,
        )
    assert report["optimizer_space_dose"][
        "matched_to_0p25_within_2e_6"
    ] is False
    assert report["gates"]["advance_to_larger_direct_finetuning_pilot"] is False


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_summarize_noise_corrected_v10_safe_exact_canary] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()
