"""CPU-only synthetic replay for the V9 causal summarizer."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import tempfile

from summarize_noise_corrected_v9_functional_canary import summarize


FORMULAS = ("F1", "F1", "F2", "F2", "F3", "F3")
INITIAL = (2, 2, 2, 2, 2, 2)


def _metrics(ranks: tuple[int, ...]) -> dict[str, object]:
    recall = lambda cutoff: sum(rank <= cutoff for rank in ranks) / len(ranks)
    summary = {
        "queries": len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": float(sorted(ranks)[len(ranks) // 2]),
        "macro_query_auroc": 0.70 + 0.01 * recall(1),
        "macro_query_auprc": 0.60 + 0.01 * recall(1),
        "mean_positive_vs_best_negative_margin": -0.1 + 0.01 * recall(1),
        "mean_top1_top2_gap": 0.1 + 0.01 * recall(1),
        "mean_signed_top1_top2_gap": -0.1 + 0.01 * recall(1),
        **{f"recall@{cutoff}": recall(cutoff) for cutoff in (1, 2, 3, 5, 10, 20)},
    }
    return {
        "retrieval": dict(summary),
        "near_subset": dict(summary),
        "micro_candidate": {
            "molecules": 12, "auroc": 0.75 + 0.01 * recall(1),
            "auprc": 0.65 + 0.01 * recall(1),
        },
        "massspecgym_10ppm_pooled_pairwise": {
            "spectrum_pairs": 24, "positive_pairs": 6, "negative_pairs": 18,
            "auroc": 0.75 + 0.01 * recall(1),
            "auprc": 0.65 + 0.01 * recall(1),
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            "spectrum_pairs": 12, "positive_pairs": 3, "negative_pairs": 9,
            "auroc": 0.75 + 0.01 * recall(1),
            "auprc": 0.65 + 0.01 * recall(1),
        },
    }


def _write_arm(
    root: Path, *, ranks: tuple[int, ...], mode: str | None,
    arm: str = "routed_direct",
) -> None:
    root.mkdir(parents=True)
    decision = {
        "formal": False,
        "arm": arm,
        "corrective_branch_mode": mode,
        "contracts": {
            "corrective_gradient_locality": "query_action_only",
            "teacher_embedding_or_distillation_target_used": False,
        },
        "gradient_calibration": {
            "effective_branch_scale": {
                "transfer": 1.0,
                "payload": 0.0 if mode == "scalar_transfer_only" else 1.0,
                "consistency": 0.0 if mode == "scalar_transfer_only" else 1.0,
            },
            "effective_dense_corrective_to_risk_gradient_ratio": 1.0,
        },
        "signal_transmission": {
            "action_retention_p10": 0.99,
            "optimizer_action_attributable_update_fraction_p10": 0.25,
            "optimizer_update_restoration_target_coverage_gate_passed": True,
            "legacy_90pct_end_to_end_loss_reproduced": False,
            "all_signal_gates_passed": True,
        },
        "evaluation": {
            "development_held_graph": {"candidate": _metrics(ranks)}
        },
    }
    (root / "decision.json").write_text(
        json.dumps(decision), encoding="utf-8",
    )
    with gzip.open(root / "held_per_query.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "query_index", "query_formula", "near", "initial_E8_rank",
            "candidate_rank",
        ))
        writer.writeheader()
        for index, rank in enumerate(ranks):
            writer.writerow({
                "query_index": index, "query_formula": FORMULAS[index],
                "near": "True", "initial_E8_rank": INITIAL[index],
                "candidate_rank": rank,
            })
    with gzip.open(root / "development_per_query.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "panel", "query_index", "rank_initial", "rank_final",
            "margin_initial", "margin_final",
        ))
        writer.writeheader()
        for index, rank in enumerate(ranks):
            writer.writerow({
                "panel": "train_corrective", "query_index": index,
                "rank_initial": INITIAL[index], "rank_final": rank,
                "margin_initial": -0.1, "margin_final": 0.1 if rank == 1 else -0.1,
            })


def _fixture(root: Path) -> dict[str, Path]:
    full = (1, 1, 1, 1, 1, 1)
    scalar = (1, 2, 1, 2, 1, 2)
    clean = INITIAL
    paths = {
        "full_action_view": root / "full",
        "scalar_transfer_only": root / "scalar",
        "v8_full_action_view": root / "v8_full",
        "v8_shuffled": root / "v8_shuffled",
        "v8_clean": root / "v8_clean",
    }
    _write_arm(paths["full_action_view"], ranks=full, mode="full_action_view")
    _write_arm(paths["scalar_transfer_only"], ranks=scalar, mode="scalar_transfer_only")
    _write_arm(paths["v8_full_action_view"], ranks=full, mode=None)
    _write_arm(
        paths["v8_shuffled"], ranks=clean, mode=None,
        arm="shuffled_action_control",
    )
    _write_arm(paths["v8_clean"], ranks=clean, mode=None, arm="clean_control")
    return paths


def test_summary_separates_branch_value_and_exact_bridge() -> None:
    with tempfile.TemporaryDirectory() as temp:
        report = summarize(_fixture(Path(temp)), repeats=200, seed=7)
    assert report["historical_control_bridge"]["passed"] is True
    assert report["corrective_gradient_budget"][
        "matched_within_absolute_1e_8"
    ] is True
    primary = report["primary_same_source_comparison"][
        "full_action_view_vs_scalar_transfer_only"
    ]
    assert primary["delta_recall1_pp"] == 50.0
    assert primary["corrected"] == 3
    assert primary["introduced"] == 0
    assert primary["formula_cluster_ci_95"]["ci_low_pp"] > 0
    assert report["gates"]["advance_to_larger_direct_finetuning_pilot"] is True


def test_failed_bridge_suppresses_historical_control_conclusions() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        paths = _fixture(root)
        held = _read_rows(paths["v8_full_action_view"] / "held_per_query.csv.gz")
        held[0]["candidate_rank"] = "2"
        _rewrite_rows(paths["v8_full_action_view"] / "held_per_query.csv.gz", held)
        report = summarize(paths, repeats=100, seed=9)
    assert report["historical_control_bridge"]["passed"] is False
    assert report["bridged_secondary_comparisons"] == {}
    assert report["gates"]["advance_to_larger_direct_finetuning_pilot"] is False


def _read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_summarize_noise_corrected_v9_functional_canary] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()
