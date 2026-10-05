"""Static and data-contract tests for direct noise augmentation fine-tuning."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from train_noise_final_e4a_direct_augmentation import (  # noqa: E402
    DirectExample, flatten_direct, frozen_reference_margins, identity_balanced_epoch,
    audit_official_reencoding_zero_change, full_graph_query_details,
    gated_materialized_action_rank,
    live_shared_v3_single_arm_promotion,
    materialized_direct_mean, molecule_margin_vectors, refresh_candidate_references,
    streaming_evaluate_embeddings,
)

SCRIPT = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"


def main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    required = [
        "clean_and_augmented_raw_spectra_train_same_encoder",
        "model_weights_changed",
        "inference_clean_spectrum_only",
        '"training_only_action_mining"',
        'else "forbidden"',
        '"P2b": "forbidden"',
        "candidate_gradient",
        "role_confounder",
        "final_shared_encoder.pt",
        "args.safety_stream_weight * safety_dose_multiplier * safe_loss",
        "changes coverage, not loss magnitude",
        "training_only_outcome_mined_actions",
        "action_outcomes_used_in_loss_or_sample_weight",
        "direct_transfer_mode",
        "rank_reference_mode",
        "frozen_official_raw_action_embedding",
        "v2_molecule_max",
        "candidate_boundary_clean_primary_corrective_dose",
        "materialized_routed",
        "materialized_action_loss_is_historical_e4",
        "materialized_all_actions_exposed_before_recycling",
        "one_maximum_margin_action_per_query",
        "E4ActionInjectorV1Bridge",
        "e4_action_injector.capture_corrective_()",
        "e4_action_injector.step_and_inject_",
        "one-best compression is forbidden",
        "action_injector_v1_receives_complete_historical_e4_gradient",
        "action_injector_v1_action_selection_or_tensor_mutation",
        "action_injector_v1_historical_e4_loss_mutation",
        "action_injector_v1_every_optimizer_step_audited",
    ]
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"direct augmentation contract missing: {missing}")
    forbidden = ["p2b_frozen"]
    found = [token for token in forbidden if token in source.lower()]
    if found:
        raise RuntimeError(f"teacher/reranker dependency entered direct training: {found}")

    r0 = ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a"
    if r0.is_dir() and (r0 / "training_actions.csv.gz").is_file():
        frame = pd.read_csv(r0 / "training_actions.csv.gz", nrows=10)
        outcome = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
        if outcome.intersection(frame.columns):
            raise RuntimeError("R0 action manifest contains post-outcome leakage")
    dummy = []
    for identity, policies in (("A", ("candidate_gradient",)),
                               ("B", ("candidate_gradient", "role_confounder"))):
        for index, policy in enumerate(policies):
            dummy.append(DirectExample(
                query_index=index, query_row=index, identity=identity, formula="F",
                positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
                official_rank=2, sample_weight=1.0, policy=policy,
            ))
    sampled = identity_balanced_epoch(dummy, np.random.default_rng(1), 2)
    if {identity: sum(item.identity == identity for item in sampled) for identity in ("A", "B")} != {"A": 2, "B": 2}:
        raise RuntimeError("identity-balanced sampler is not exact")
    b_policy = {item.policy for item in sampled if item.identity == "B"}
    if b_policy != {"candidate_gradient", "role_confounder"}:
        raise RuntimeError("combined-policy round robin dropped a fixed policy")
    # Frozen-reference ranking must be exact and must propagate gradients only
    # to the query vector, never to the numpy anchors.
    import torch
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    example = DirectExample(
        query_index=0, query_row=0, identity="A", formula="F",
        positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
        official_rank=2, sample_weight=1.0,
    )
    margin = frozen_reference_margins(
        query, [example], {1: np.asarray([1.0, 0.0], np.float32),
                           2: np.asarray([0.0, 1.0], np.float32)},
    )
    if not np.allclose(margin.detach().numpy(), [1.0]):
        raise RuntimeError("frozen reference margin is numerically wrong")
    margin.sum().backward()
    if query.grad is None or not np.allclose(query.grad.numpy(), [[1.0, -1.0]]):
        raise RuntimeError("frozen reference margin gradient is wrong")
    # Repaired materialized rank pressure must vanish only for an action that
    # already clears the registered margin. Weighted reduction must retain the
    # fixed run-level weights instead of renormalizing each microbatch.
    action_margin = torch.tensor([0.0, 0.10], requires_grad=True)
    gated, active = gated_materialized_action_rank(
        action_margin, rank_margin=0.05, temperature=0.10,
        injection_mode="multi_action_balanced",
    )
    if active.tolist() != [True, False] or float(gated[1]) != 0.0:
        raise RuntimeError("satisfied materialized action rank was not gated")
    gated.sum().backward()
    if action_margin.grad is None or not (
        float(action_margin.grad[0]) < 0 and float(action_margin.grad[1]) == 0.0
    ):
        raise RuntimeError("materialized action-rank gate propagated a wrong gradient")
    weighted_examples = [
        DirectExample(
            query_index=index, query_row=index, identity=str(index), formula="F",
            positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
            official_rank=2, sample_weight=weight,
        )
        for index, weight in enumerate((0.25, 2.0))
    ]
    reduced = materialized_direct_mean(
        torch.tensor([4.0, 3.0]), weighted_examples, "multi_action_balanced",
    )
    if not torch.allclose(reduced, torch.tensor(3.5)):
        raise RuntimeError("multi-action reduction renormalized fixed sample weights")
    # The evaluator uses the best true-identity reference. A weak second
    # positive must not create an additional bad training edge.
    encoded = torch.tensor([
        [1.0, 0.0],   # query
        [0.9, 0.0],   # strong true reference
        [-0.7, 0.0],  # weak true reference
        [0.8, 0.0],   # negative molecule 1
        [0.4, 0.0],   # negative molecule 2
    ], requires_grad=True)
    layout = [{"clean": 0, "positive": [1, 2], "negative": [3, 4]}]
    molecular = molecule_margin_vectors(encoded, layout, "clean")[0]
    if not torch.allclose(molecular, torch.tensor([0.1, 0.5])):
        raise RuntimeError("v2 molecule-max boundary differs from evaluator")

    # A materialized action must enter the exact historical E4 tensor position.
    # This catches the regression where a rich raw action was reduced to a new
    # scalar target or silently replaced by the clean spectrum.
    class Store:
        values = {
            0: torch.tensor([[100.0, 0.2], [150.0, 0.8], [200.0, 0.5]]),
            1: torch.tensor([[101.0, 0.4], [151.0, 0.6], [201.0, 0.3]]),
            2: torch.tensor([[102.0, 0.7], [152.0, 0.3], [202.0, 0.2]]),
        }

        def one(self, row: int) -> torch.Tensor:
            return self.values[int(row)].clone()

        def get(self, rows) -> list[torch.Tensor]:
            return [self.one(int(row)) for row in rows]

    procedural = DirectExample(
        query_index=0, query_row=0, identity="A", formula="F",
        positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
        official_rank=2, sample_weight=1.0, target_path=(1,), attenuation=0.5,
    )
    procedural_tensors, procedural_layout, _ = flatten_direct(Store(), [procedural], True)
    expected_action = procedural_tensors[int(procedural_layout[0]["action"])]
    materialized = DirectExample(
        query_index=0, query_row=0, identity="A", formula="F",
        positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
        official_rank=2, sample_weight=1.0, action_id="A0",
        materialized_action_index=0,
    )
    materialized_tensors, materialized_layout, _ = flatten_direct(
        Store(), [materialized], True,
        materialized_action_spectra=expected_action.numpy()[None],
        materialized_action_arm="targeted",
    )
    observed_action = materialized_tensors[int(materialized_layout[0]["action"])]
    if not torch.equal(expected_action, observed_action):
        raise RuntimeError("materialized action changed before the historical E4 loss")
    duplicate_tensors, duplicate_layout, _ = flatten_direct(
        Store(), [materialized], True,
        materialized_action_spectra=expected_action.numpy()[None],
        materialized_action_arm="clean_duplicate",
    )
    if not torch.equal(
        duplicate_tensors[int(duplicate_layout[0]["clean"])],
        duplicate_tensors[int(duplicate_layout[0]["action"])],
    ):
        raise RuntimeError("materialized clean-duplicate control changed the clean query")

    # Full-graph replay must use the router's bounded query block, not expand
    # query embeddings once per candidate edge.  This synthetic block also
    # checks strict-tie/rank and hardest-negative indexing.
    class TinyGraph:
        query_row = np.asarray([10], dtype=np.int64)
        query_has_near = np.asarray([True])

        @staticmethod
        def query_block(query: int):
            if query != 0:
                raise IndexError(query)
            return (
                slice(0, 3), np.asarray([20, 21, 22], dtype=np.int64),
                np.asarray([0, 1, 2, 3], dtype=np.int64), 0,
            )

    graph_rows = np.asarray([10, 20, 21, 22], dtype=np.int64)
    graph_embeddings = np.asarray([
        [1.0, 0.0], [0.5, 0.8660254], [0.7, 0.7141428], [0.1, 0.9949874],
    ], dtype=np.float32)
    graph_rank, graph_top, graph_margin = full_graph_query_details(
        TinyGraph(), graph_rows, graph_embeddings, np.asarray([0], dtype=np.int64),
    )
    if not (
        graph_rank.tolist() == [2]
        and graph_top.tolist() == [1]
        and np.allclose(graph_margin, [-0.2], atol=1e-6)
    ):
        raise RuntimeError("bounded router-geometry evaluator is numerically wrong")
    streamed_rank, streamed_summary = streaming_evaluate_embeddings(
        TinyGraph(), graph_rows, graph_embeddings, np.asarray([0], dtype=np.int64),
    )
    if streamed_rank.tolist() != [2] or streamed_summary["errors"] != 1:
        raise RuntimeError("streaming retrieval summary changed bounded strict rank")

    # A cache/fresh-forward score perturbation may flip strict rank only at an
    # almost exact tie.  The zero-change gate must accept that discontinuity,
    # but reject the same rank flip when its edge drift exceeds tolerance.
    class ZeroGateGraph:
        query_ptr = np.asarray([0, 2], dtype=np.int64)
        molecule_ptr = np.asarray([0, 1, 2], dtype=np.int64)
        query_row = np.asarray([10], dtype=np.int64)
        pair_candidate_row = np.asarray([20, 21], dtype=np.int64)

    zero_rows = np.asarray([10, 20, 21], dtype=np.int64)
    official_zero = np.asarray([
        [1.0, 0.0],
        [0.50000, np.sqrt(1.0 - 0.50000 ** 2)],
        [0.50001, np.sqrt(1.0 - 0.50001 ** 2)],
    ], dtype=np.float32)
    fresh_zero = np.asarray([
        [1.0, 0.0],
        [0.50002, np.sqrt(1.0 - 0.50002 ** 2)],
        [0.50000, np.sqrt(1.0 - 0.50000 ** 2)],
    ], dtype=np.float32)
    tie_report = audit_official_reencoding_zero_change(
        ZeroGateGraph(), zero_rows, official_zero, fresh_zero,
        np.asarray([0], dtype=np.int64), score_atol=5e-4,
    )
    if not (
        tie_report["gate_passed"]
        and tie_report["rank_mismatches"] == 1
        and tie_report["boundary_rank_mismatches"] == 1
        and tie_report["nonboundary_rank_mismatches"] == 0
    ):
        raise RuntimeError("tie-aware official zero-change gate rejected a valid tie")
    drifted_zero = fresh_zero.copy()
    drifted_zero[1] = np.asarray([0.7, np.sqrt(1.0 - 0.7 ** 2)], np.float32)
    drift_report = audit_official_reencoding_zero_change(
        ZeroGateGraph(), zero_rows, official_zero, drifted_zero,
        np.asarray([0], dtype=np.int64), score_atol=5e-4,
    )
    if drift_report["gate_passed"]:
        raise RuntimeError("official zero-change gate accepted substantive score drift")

    # A lower negative may cross the positive while the best negative remains
    # far above it.  That changes rank 2<->3 without changing the Top-1 margin;
    # the audit must inspect each rank-contributing candidate comparison.
    class LowerTieGraph:
        query_ptr = np.asarray([0, 3], dtype=np.int64)
        molecule_ptr = np.asarray([0, 1, 2, 3], dtype=np.int64)
        query_row = np.asarray([30], dtype=np.int64)
        pair_candidate_row = np.asarray([31, 32, 33], dtype=np.int64)

    lower_rows = np.asarray([30, 31, 32, 33], dtype=np.int64)
    official_lower = np.asarray([
        [1.0, 0.0],
        [0.50, np.sqrt(1.0 - 0.50 ** 2)],
        [0.60, 0.80],
        [0.49999, np.sqrt(1.0 - 0.49999 ** 2)],
    ], dtype=np.float32)
    fresh_lower = np.asarray([
        [1.0, 0.0],
        [0.50, np.sqrt(1.0 - 0.50 ** 2)],
        [0.60, 0.80],
        [0.50001, np.sqrt(1.0 - 0.50001 ** 2)],
    ], dtype=np.float32)
    lower_tie_report = audit_official_reencoding_zero_change(
        LowerTieGraph(), lower_rows, official_lower, fresh_lower,
        np.asarray([0], dtype=np.int64), score_atol=5e-4,
    )
    if not (
        lower_tie_report["gate_passed"]
        and lower_tie_report["rank_mismatches"] == 1
        and lower_tie_report["boundary_rank_mismatches"] == 1
        and lower_tie_report["nonboundary_rank_mismatches"] == 0
        and lower_tie_report["mismatch_detail_first_64"][0]
        ["switched_candidate_comparisons"] == 1
    ):
        raise RuntimeError(
            "tie-aware zero-change gate rejected a valid lower-rank tie"
        )

    # A materialized action can activate a different spectrum row of the same
    # candidate molecule.  Both the clean-active and action-active rows must be
    # encoded; substituting one for the other destroys the clean margin.
    class BoundaryUnionGraph:
        query_ptr = np.asarray([0, 3], dtype=np.int64)
        molecule_ptr = np.asarray([0, 2, 4, 5], dtype=np.int64)
        pair_candidate_row = np.asarray([100, 101, 200, 201, 300], dtype=np.int64)

    union_rows = np.asarray([10, 100, 101, 200, 201, 300], dtype=np.int64)
    union_encoded = np.asarray([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
        [0.7, 0.3], [0.2, 0.8], [0.6, 0.4],
    ], dtype=np.float32)
    union_example = DirectExample(
        query_index=0, query_row=10, identity="A", formula="F",
        positive_rows=(101,), negative_rows=(201,), official_margin=0.2,
        official_rank=2, sample_weight=1.0,
        forced_positive_row=101, forced_negative_row=201,
    )
    refreshed = refresh_candidate_references(
        BoundaryUnionGraph(), [union_example], union_rows, union_encoded,
        positives=1, negatives=1, preserve_forced_negative=True,
    )[0]
    if refreshed.positive_rows != (100, 101):
        raise RuntimeError("action positive row replaced the clean-active positive row")
    if refreshed.negative_rows != (200, 201):
        raise RuntimeError("action negative row replaced the clean-active negative row")

    # Regression from job 2338326: a nominally positive R@1 delta must not
    # hide zero risk-net, negative near risk, a near R@5 regression, or a gain
    # far below the registered four-point target.
    def registered_panel(recall1: float, near_recall5: float) -> dict:
        higher = {
            "recall@1": recall1,
            "recall@2": 0.80,
            "recall@3": 0.82,
            "recall@5": 0.84,
            "recall@10": 0.86,
            "recall@20": 0.88,
            "mrr": 0.70,
            "macro_query_auroc": 0.75,
            "macro_query_auprc": 0.65,
            "mean_positive_vs_best_negative_margin": 0.10,
            "mean_top1_top2_gap": 0.08,
            "mean_signed_top1_top2_gap": 0.06,
            "mean_rank": 1.5,
            "median_rank": 1.0,
        }
        near = dict(higher)
        near["recall@5"] = near_recall5
        pair = {"auroc": 0.80, "auprc": 0.70}
        return {
            "retrieval": higher,
            "near_subset": near,
            "micro_candidate": pair,
            "massspecgym_10ppm_pooled_pairwise": pair,
            "massspecgym_mh_10ppm_pooled_pairwise": pair,
        }

    initialization_panel = registered_panel(0.934926, 0.994544)
    student_panel = registered_panel(0.936235, 0.994249)
    official_panel = registered_panel(0.932962, 0.993806)
    promotion_gates, promotion_diagnostics = live_shared_v3_single_arm_promotion(
        {
            "official": official_panel,
            "initialization": initialization_panel,
            "student": student_panel,
            "paired_top1_vs_initialization": {
                "corrected": 48,
                "introduced": 24,
                "risk_net_lambda2": 0,
                "near": {
                    "queries": 6781,
                    "corrected": 21,
                    "introduced": 12,
                    "risk_net_lambda2": -3,
                },
            },
        },
        {
            "incremental_delta_recall1": 0.001309,
            "preservation_vs_initialization_mean": 0.999,
        },
        {"mean": 0.001309, "ci_low": 0.000114, "ci_high": 0.002544},
    )
    required_failures = {
        "risk_net_lambda2_positive_vs_initial_e8",
        "near_risk_net_lambda2_positive_vs_initial_e8",
        "all_registered_metrics_nonnegative_vs_initial_e8",
        "all_registered_metrics_strict_or_boundary_vs_initial_e8",
        "strict_four_pp_recall1_gain_vs_official",
    }
    if any(promotion_gates[key] for key in required_failures):
        raise RuntimeError(
            "job-2338326 scientific failure was incorrectly promoted: "
            f"{promotion_gates}"
        )
    if "near_subset.recall@5" not in promotion_diagnostics[
        "registered_metric_violations"
    ]:
        raise RuntimeError("near R@5 regression disappeared from V3 promotion")
    if all(promotion_gates.values()):
        raise RuntimeError("job-2338326 counterexample passed V3 promotion")
    print("[test_noise_final_e4a_direct_augmentation] PASS")


if __name__ == "__main__":
    main()
