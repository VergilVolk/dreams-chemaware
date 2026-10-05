"""CPU/static gates for broad measured-positive Noise Stage-4."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tasks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
BUILDER = ROOT / "tasks/build_noise_dreams_native_broad_positive_stage4.py"
TRAINER = ROOT / "tasks/train_noise_dreams_native_residual_stage2.py"
SUMMARY = ROOT / "tasks/summarize_noise_dreams_native_residual_stage2.py"
GNPS_SUMMARY = ROOT / "tasks/summarize_noise_dreams_native_stage4_gnps.py"
SBATCH = ROOT / "tasks/run_noise_dreams_native_broad_positive_stage4_2gpu.sbatch"


def _function(path: Path, name: str, namespace: dict) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        body for body in tree.body
        if isinstance(body, ast.FunctionDef) and body.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_positive_difficulty_is_measured_and_excludes_extreme_tail() -> None:
    tiers = _function(BUILDER, "tier_indices", {"np": np})
    choose = _function(
        BUILDER, "choose_measured_relation", {"np": np, "tier_indices": tiers},
    )
    scores = np.asarray([0.10, 0.20, 0.30, 0.40, 0.50])
    indices = tiers(scores)
    assert indices == {"easy": 4, "medium": 2, "hard": 1}
    assert indices["hard"] != int(np.argmin(scores))
    relation = choose(
        np.arange(10, 15), np.asarray([20, 21]), scores,
        np.asarray([0.28, 0.25]), "hard",
        minimum_margin=-0.1, maximum_margin=0.1,
    )
    assert relation is not None
    assert relation["positive_row"] == 11
    assert relation["negative_row"] == 20
    assert np.isclose(relation["triplet_margin"], -0.08)
    assert -0.1 <= relation["triplet_margin"] < 0.1


def test_formula_round_robin_is_balanced_and_query_unique() -> None:
    import hashlib

    stable = _function(
        BUILDER, "stable_digest_int", {"hashlib": hashlib},
    )
    select = _function(
        BUILDER, "formula_round_robin",
        {"defaultdict": __import__("collections").defaultdict,
         "stable_digest_int": stable},
    )
    records = [
        {"query_formula": formula, "query_ik14": f"IK{query}", "query_index": query}
        for formula, queries in (("A", range(5)), ("B", range(5, 8)), ("C", range(8, 10)))
        for query in queries
    ]
    selected = select(records, 6)
    counts = pd.Series([row["query_formula"] for row in selected]).value_counts()
    assert len({row["query_index"] for row in selected}) == 6
    assert set(counts.index) == {"A", "B", "C"}
    assert int(counts.max() - counts.min()) == 0


def test_formula_coverage_first_reaches_the_budget_constrained_maximum() -> None:
    import hashlib

    stable = _function(
        BUILDER, "stable_digest_int", {"hashlib": hashlib},
    )
    round_robin = _function(
        BUILDER, "formula_round_robin",
        {"defaultdict": __import__("collections").defaultdict,
         "stable_digest_int": stable},
    )
    select = _function(
        BUILDER, "formula_coverage_first",
        {"formula_round_robin": round_robin},
    )
    records = [
        {
            "query_formula": formula,
            "query_ik14": f"IK{query}",
            "query_index": query,
            "broadens_action_query_coverage": query not in {0, 5},
        }
        for formula, queries in (
            ("A", range(0, 3)),
            ("B", range(3, 5)),
            ("C", range(5, 8)),
            ("D", range(8, 10)),
        )
        for query in queries
    ]
    selected = select(records, 6, {"A"})
    assert len(selected) == 6
    assert len({int(row["query_index"]) for row in selected}) == 6
    # B/C/D are all the formulas absent from the retained action ledger, so a
    # six-event budget must cover every one of them before filling extra rows.
    assert {"B", "C", "D"}.issubset(
        {str(row["query_formula"]) for row in selected}
    )


class _Registry:
    HDF5 = 0
    ACTION = 1

    def __init__(self) -> None:
        self.entries: list[tuple[int, int]] = []
        self.positions: dict[tuple[int, int], int] = {}

    def add(self, kind: int, source: int) -> int:
        key = (int(kind), int(source))
        if key not in self.positions:
            self.positions[key] = len(self.entries)
            self.entries.append(key)
        return self.positions[key]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([value[0] for value in self.entries], dtype=np.int8),
            np.asarray([value[1] for value in self.entries], dtype=np.int64),
        )


def test_pool_retains_action_units_and_clean_fallback_without_tensor_replay() -> None:
    make_pool = _function(
        BUILDER, "make_pool",
        {
            "np": np,
            "pd": pd,
            "Registry": _Registry,
            "fixed_unicode": lambda values: np.asarray(list(values), dtype="U16"),
        },
    )
    selected = pd.DataFrame({
        "query_index": [1, 1, 3, 4],
        "query_row": [10, 10, 30, 40],
        "positive_rows": [(11, 13), (21,), (31, 33, 35), (41, 43)],
        "negative_rows": [(12, 14), (22, 24), (32,), (42, 44, 46)],
        "query_formula": ["A", "B", "C", "D"],
        "is_action_event": [True, True, True, False],
        "action_anchor_representable": [True, False, True, False],
    })
    pool, actions = make_pool(selected)
    assert actions == 3
    assert len(pool["anchor_idx"]) == 4
    assert len(np.unique(pool["event_query"])) == 3
    assert np.array_equal(pool["event_kind"], [2, 1, 2, 0])
    assert np.array_equal(pool["event_action_index"], [0, 1, 2, -1])
    # The unrepresentable action uses the measured query row as anchor.
    assert pool["registry_kind"][pool["anchor_idx"][1]] == _Registry.HDF5
    assert np.array_equal(np.diff(pool["positive_ptr"]), [2, 1, 3, 2])
    assert np.array_equal(np.diff(pool["negative_ptr"]), [2, 2, 1, 3])


def test_residual_difficulty_pools_avoid_old_boundary_when_possible() -> None:
    tiers = _function(BUILDER, "tier_indices", {"np": np})
    select = _function(
        BUILDER, "three_level_rows", {"np": np, "tier_indices": tiers},
    )
    residual = _function(
        BUILDER,
        "residual_negative_rows",
        {"np": np, "three_level_rows": select},
    )
    rows = np.asarray([10, 11, 12, 13, 14])
    selected = select(rows, np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert selected == [14, 12, 11]
    embeddings = np.asarray([
        [1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]
    ])
    positions = {100 + index: index for index in range(5)}
    negative, reused = residual(
        [np.asarray([100, 101]), np.asarray([102]), np.asarray([103, 104])],
        positions, embeddings, np.asarray([1.0, 0.0]), 100,
    )
    assert 100 not in negative
    assert reused is False


def test_robust_dynamic_pool_removes_one_bad_cross_pair_without_losing_signal() -> None:
    select = _function(
        BUILDER, "robust_dynamic_subpool", {"np": np},
    )
    valid = np.asarray([
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ])
    quality = np.asarray([
        [0.3, 0.2, -1.0],
        [0.2, 0.1, -1.0],
        [-1.0, -1.0, 0.4],
    ])
    positive, negative = select(valid, quality)
    assert np.array_equal(positive, [0, 1])
    assert np.array_equal(negative, [0, 1])
    assert bool(np.all(valid[np.ix_(positive, negative)]))
    # The original clean query is a mandatory hard-positive target.  Requiring
    # row 2 must choose its smaller valid relation instead of silently keeping
    # the larger rectangle that drops the scientific target.
    positive, negative = select(
        valid, quality, required_positive_index=2,
    )
    assert np.array_equal(positive, [2])
    assert np.array_equal(negative, [2])


def test_hard_positive_eligibility_uses_identity_and_native_difficulty_only() -> None:
    bounded = _function(
        BUILDER, "bounded_hinge_active_pairs", {"np": np},
    )
    margins = np.asarray([[-0.10, -0.11, 0.00, 0.099, 0.10]])
    assert np.array_equal(
        bounded(margins, minimum_margin=-0.1, maximum_margin=0.1),
        [[True, False, True, True, False]],
    )


def test_stage1_action_retention_detects_any_boundary_mutation() -> None:
    audit = _function(
        BUILDER, "audit_stage1_action_retention", {"np": np, "pd": pd},
    )
    actions = pd.DataFrame({
        "action_positive_row": [11, 21, 31],
        "action_hard_negative_row": [12, 22, 32],
    })
    retained = pd.DataFrame({
        "stage1_action_index": [2, 0, 1],
        "positive_row": [31, 11, 21],
        "negative_row": [32, 12, 22],
        "action_anchor_representable": [True, True, False],
    })
    capable = np.asarray([True, False, True])
    assert all(audit(retained, actions, capable).values())
    mutated = retained.copy()
    mutated.loc[mutated.stage1_action_index.eq(1), "negative_row"] = 999
    report = audit(mutated, actions, capable)
    assert report["every_stage1_effective_action_is_retained_exactly_once"]
    assert not report[
        "every_stage1_action_keeps_its_exact_positive_negative_boundary"
    ]


def test_runtime_and_auc_contracts_change_no_training_parameter() -> None:
    builder = BUILDER.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    summary = SUMMARY.read_text(encoding="utf-8")
    script = SBATCH.read_text(encoding="utf-8")
    for token in (
        "STAGE4_TOTAL_EVENTS = 33575",
        "STAGE4_OPTIMIZER_STEPS = 8394",
        "Stage-1 already consumed every registered action",
        "all_seven_registered_sources_entered_residual_selection",
        "selected_actions_retain_all_seven_registered_sources",
        "exactly_one_action_event_per_query",
        "every_action_query_has_one_clean_preservation_event",
        "hard_positive_membership_is_identity_grounded",
        "pretraining_target_control_advantage_not_used_for_eligibility",
        "all_dynamic_pairs_are_bounded_and_hinge_active",
        "stage1_exact_positive_is_absent_from_every_residual_pool",
        "every_residual_positive_pool_contains_original_clean_query",
        "clean_stream_is_only_paired_preservation_plus_256_protection",
        "old_exact_positive_boundaries_replayed",
        "action view -> original clean query plus additional",
        "valid_dynamic_pairs = bounded_hinge_active_pairs(",
        "if formula in validation_formulas",
    ):
        assert token in builder
    assert '"broad_positive_stage4"' in trainer
    assert '"lr": 5e-6' in trainer
    assert '"maximum_steps": STAGE4_OPTIMIZER_STEPS' in trainer
    stage4_contract = trainer.split('"broad_positive_stage4"', 1)[1].split("},", 1)[0]
    assert '"exact_steps"' not in stage4_contract
    assert "RestoreNativeAdamState" in trainer
    assert '"lr": 1e-6' in stage4_contract
    assert "optimizer_restore = None" in trainer
    assert '"broad_positive_stage4", "negative_residual_stage5"' in trainer
    assert "and not np.all(capable)" in trainer
    assert '"stage4"' in summary
    assert "all_auc_auprc_improve" in summary
    assert "macro_auc_formula_cis_positive" in summary
    assert "auc_metric_comparisons" in summary
    assert "query_auc_formula_cluster_comparisons" in summary
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "--curriculum broad_positive_stage4" in script
    assert "--lr 1e-6" in script
    assert "--weight-decay 0" in script
    assert "--triplet-loss-margin 0.1" in script
    assert "--batch-size 4" in script
    assert "--max-epochs 1" in script
    assert "--no-write-held-metric-evidence" in script
    assert '--output "$EVALUATION/stage1"' in script
    assert '--stage1 "$EVALUATION/stage1"' in script
    assert "summarize_noise_dreams_native_residual_stage2.py" in script
    assert "gnps_gold_silver_10ppm_benchmark_v1" in script
    assert "encode_gnps_gold_silver_10ppm_checkpoint.py" in script
    assert "evaluate_gnps_gold_silver_10ppm_embeddings.py" in script
    assert "summarize_noise_dreams_native_stage4_gnps.py" in script
    assert '"$RUN_ROOT/candidate/targeted_slim.pt"' in script
    assert 'STAGE4_BUILDER_VERSION = "noise_native_identity_grounded_residual_v6"' in builder
    assert "NOISE_STAGE4V6_TRIPLETS" in script
    assert "NOISE_STAGE4V5_TRIPLETS" not in script
    assert "NOISE_STAGE4R_TRIPLETS" not in script
    assert "--action-advantage-floor" not in script
    assert "huggingface.co" not in script
    assert "curl " not in script


def test_gnps_summary_reads_every_registered_metric_and_ci() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage4_gnps_summary", GNPS_SUMMARY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    metric = {
        "retrieval": {
            "recall@1": 0.8, "mrr": 0.85,
            "macro_query_auroc": 0.9, "macro_query_auprc": 0.85,
        },
        "near_subset": {
            "recall@1": 0.7, "mrr": 0.75,
            "macro_query_auroc": 0.8, "macro_query_auprc": 0.75,
        },
        "micro_candidate": {"auroc": 0.9, "auprc": 0.8},
        "gnps_10ppm_pooled_pairwise": {"auroc": 0.88, "auprc": 0.78},
    }
    candidate = {
        key: ({subkey: value + 0.01 for subkey, value in body.items()})
        for key, body in metric.items()
    }
    ci = {
        name: {"ci_low_pp": 0.01}
        for name in ("recall@1", "mrr", "macro_query_auroc", "macro_query_auprc")
    }
    panel = {
        "baseline": metric, "candidate": candidate,
        "paired": {
            "formula_cluster_paired_ci": ci,
            "near_formula_cluster_paired_ci": ci,
            "risk_net_lambda2": 3, "near_risk_net_lambda2": 1,
        },
    }
    report = {
        "status": "gnps_gold_silver_10ppm_embedding_evaluation_complete",
        "panels": {name: panel for name in module.PANELS},
    }
    summary = module.comparison_summary(report)
    assert all(
        body["all_registered_metrics_improve"]
        and body["all_registered_formula_cis_positive"]
        for body in summary.values()
    )


def test_all_stage4_sources_parse() -> None:
    for path in (BUILDER, TRAINER, SUMMARY, GNPS_SUMMARY):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_noise_dreams_native_broad_positive_stage4] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()
