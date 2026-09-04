"""Paired multifold decision for ChemAware-v3 PEFT molecule teacher controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from summarize_chemaware_shared_v2_g2 import ARMS, compare  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g1")
    parser.add_argument("--g2-root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g2")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g2_summary.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 73])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument(
        "--chemical-objective",
        choices=("candidate_hardness", "frozen_probe_targeted"),
        default="candidate_hardness",
    )
    return parser.parse_args()


def load_seed(
    root: Path,
    seed: int,
    folds: int,
    graph: CandidateGraph,
    arm: str | None,
) -> dict:
    query_parts, old_parts, new_parts, decisions, provenance = [], [], [], [], []
    capacity_reference = None
    training_contract_reference = None
    expected_status = (
        "chemaware_shared_v3_g1_peft_fold_complete"
        if arm is None else "chemaware_shared_v3_g2_peft_fold_complete"
    )
    for fold in range(folds):
        run = root / f"seed_{seed}" / f"fold_{fold}"
        decision_path = run / "decision.json"
        prediction_path = run / "outer_predictions.npz"
        checkpoint_path = run / "peft.pt"
        if not all(path.is_file() for path in (decision_path, prediction_path, checkpoint_path)):
            raise FileNotFoundError(f"incomplete {'G1' if arm is None else arm} run: {run}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        first_step = decision.get("first_step_audit") or {}
        if (
            decision.get("status") != expected_status
            or decision.get("formal") is not True
            or int(decision.get("seed", -1)) != seed
            or int(decision.get("outer_fold", -1)) != fold
            or bool(decision.get("chemical_supervision")) != (arm is not None)
            or (arm is not None and decision.get("teacher_control") != arm)
            or decision.get("query_reference_encoder_shared") is not True
            or decision.get("candidate_inputs_at_inference") is not False
            or decision.get("P2b_used") is not False
            or float(first_step.get("gradient_l2", 0)) <= 0
            or float(first_step.get("parameter_update_l2", 0)) <= 0
            or int(first_step.get("changed_parameter_tensors", 0)) <= 0
            or decision.get("preflight", {}).get(
                "complete_split_eligible_candidate_groups"
            ) is not True
        ):
            raise RuntimeError(f"invalid {'G1' if arm is None else arm} decision: {decision_path}")
        capacity = decision.get("capacity")
        training_contract = decision.get("training_contract")
        if capacity_reference is None:
            capacity_reference = capacity
            training_contract_reference = training_contract
        elif capacity != capacity_reference or training_contract != training_contract_reference:
            raise RuntimeError(
                f"seed {seed} arm {arm} changes capacity or training contract across folds"
            )
        with np.load(prediction_path) as body:
            query = np.asarray(body["query"], dtype=np.int64)
            old_rank = np.asarray(body["old_rank"], dtype=np.int32)
            new_rank = np.asarray(body["new_rank"], dtype=np.int32)
        if (
            query.ndim != 1
            or not (len(query) == len(old_rank) == len(new_rank))
            or np.any(query < 0)
            or np.any(query >= graph.n_queries)
            or np.any(old_rank < 1)
            or np.any(new_rank < 1)
        ):
            raise RuntimeError(f"malformed {'G1' if arm is None else arm} predictions: {prediction_path}")
        query_parts.append(query)
        old_parts.append(old_rank)
        new_parts.append(new_rank)
        decisions.append(decision)
        provenance.append({
            "fold": fold,
            "decision_sha256": sha256_file(decision_path),
            "predictions_sha256": sha256_file(prediction_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
    query = np.concatenate(query_parts)
    order = np.argsort(query, kind="stable")
    query = query[order]
    old_rank = np.concatenate(old_parts)[order]
    new_rank = np.concatenate(new_parts)[order]
    if not np.array_equal(query, np.arange(graph.n_queries)):
        raise RuntimeError(f"seed {seed} arm {arm} does not cover every query exactly once")
    return {
        "query": query,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "decisions": decisions,
        "provenance": provenance,
        "capacity": capacity_reference,
        "training_contract": training_contract_reference,
    }


def assert_matched(
    g1: dict,
    g2: dict,
    arm: str,
    chemical_objective: str = "candidate_hardness",
) -> None:
    if not np.array_equal(g1["query"], g2["query"]):
        raise RuntimeError(f"G1/{arm} query ledger mismatch")
    if not np.array_equal(g1["old_rank"], g2["old_rank"]):
        raise RuntimeError(f"G1/{arm} official baseline mismatch")
    matched_keys = (
        "training_query_ledger_sha256", "allowed_molecule_ledger_sha256",
        "initial_peft_state_sha256", "capacity", "training_contract",
        "train_queries", "train_identities",
    )
    for fold, (clean, chemical) in enumerate(zip(g1["decisions"], g2["decisions"])):
        valid_objectives = (
            {"frozen_morgan_ridge_probe_targeted_listwise"}
            if chemical_objective == "frozen_probe_targeted" else
            {
                "frozen_morgan_binary_connectivity_candidate_hardness_relative_centered",
                "frozen_morgan_binary_connectivity_candidate_hardness_absolute_bounded",
            }
        )
        if (
            chemical.get("chemical_objective") not in valid_objectives
            or chemical.get("teacher_kind") != "morgan_binary_connectivity"
            or chemical.get("training_only_projector_used") is not False
        ):
            raise RuntimeError(f"G2-PEFT {arm} fold {fold} uses an invalid objective")
        if chemical_objective == "frozen_probe_targeted":
            probe = chemical.get("frozen_probe_fit_audit") or {}
            if (
                chemical.get("training_only_frozen_probe_used") is not True
                or chemical.get("chemical_gradient_absorber_trainable") is not False
                or int(probe.get("trainable_parameters", -1)) != 0
                or probe.get("discarded_at_inference") is not True
            ):
                raise RuntimeError(
                    f"G2b-PEFT {arm} fold {fold} violates frozen-probe contract"
                )
        gradient_history = [
            row.get("chemical_gradient_audit")
            for row in chemical.get("history", [])[1:]
        ]
        if (
            not gradient_history
            or any(
                audit is None
                or float(audit.get("chemical_minus_clean_gradient_norm", 0)) <= 1e-12
                or int(audit.get("chemical_delta_nonzero_parameter_tensors", 0)) <= 0
                or not audit.get("chemical_delta_gradient_signature")
                for audit in gradient_history
            )
        ):
            raise RuntimeError(f"G2-PEFT {arm} fold {fold} lacks chemical-gradient evidence")
        for key in matched_keys:
            if clean.get(key) != chemical.get(key):
                raise RuntimeError(f"G1/G2-PEFT {arm} fold {fold} unmatched {key}")


def signature_cosine(left: dict, right: dict) -> float:
    left_values = np.asarray(left["history"][1]["chemical_gradient_audit"][
        "chemical_delta_gradient_signature"
    ], dtype=np.float64)
    right_values = np.asarray(right["history"][1]["chemical_gradient_audit"][
        "chemical_delta_gradient_signature"
    ], dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 1 or not len(left_values):
        raise RuntimeError("paired G2-PEFT gradient signatures are not aligned")
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    if denominator <= 0:
        raise RuntimeError("paired G2-PEFT gradient signature has zero norm")
    return float(np.dot(left_values, right_values) / denominator)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G2-PEFT summary: {args.output}")
    graph = CandidateGraph(args.graph)
    all_seed = {}
    capacity_reference = None
    training_contract_reference = None
    chemical_contract_reference = None
    for seed_index, seed in enumerate(args.seeds):
        g1 = load_seed(args.g1_root, seed, args.folds, graph, None)
        arms = {
            arm: load_seed(args.g2_root / arm, seed, args.folds, graph, arm)
            for arm in ARMS
        }
        for arm, ledger in arms.items():
            assert_matched(g1, ledger, arm, args.chemical_objective)
        if capacity_reference is None:
            capacity_reference = g1["capacity"]
            training_contract_reference = g1["training_contract"]
        elif (
            g1["capacity"] != capacity_reference
            or g1["training_contract"] != training_contract_reference
        ):
            raise RuntimeError("G1 capacity or training contract differs across seeds")
        for fold in range(args.folds):
            contracts = [
                json.dumps(arms[arm]["decisions"][fold].get("chemical_contract"), sort_keys=True)
                for arm in ARMS
            ]
            if len(set(contracts)) != 1:
                raise RuntimeError(f"G2-PEFT fold {fold} chemical contracts differ")
            if chemical_contract_reference is None:
                chemical_contract_reference = contracts[0]
            elif contracts[0] != chemical_contract_reference:
                raise RuntimeError("G2-PEFT chemical contract differs across folds or seeds")
            global_coverage = [
                arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                    "chemical_effect_queries"
                )
                for arm in ("correct", "identity_permuted", "random_marginal")
            ]
            global_masks = [
                arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                    "teacher_observable_mask_sha256"
                )
                for arm in ("correct", "identity_permuted", "random_marginal")
            ]
            scoped_coverage = [
                arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                    "chemical_effect_queries"
                )
                for arm in ("correct_same_formula_scope", "same_formula_mismatched")
            ]
            scoped_masks = [
                arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                    "teacher_observable_mask_sha256"
                )
                for arm in ("correct_same_formula_scope", "same_formula_mismatched")
            ]
            if None in global_coverage + scoped_coverage + global_masks + scoped_masks:
                raise RuntimeError(f"G2-PEFT fold {fold} lacks teacher coverage audit")
            if (
                len(set(global_coverage)) != 1
                or len(set(scoped_coverage)) != 1
                or len(set(global_masks)) != 1
                or len(set(scoped_masks)) != 1
            ):
                raise RuntimeError(f"G2-PEFT fold {fold} controls use different coverage")
            if args.chemical_objective == "frozen_probe_targeted":
                global_selection = [
                    arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                        "selection_query_ledger_sha256"
                    )
                    for arm in ("correct", "identity_permuted", "random_marginal")
                ]
                scoped_selection = [
                    arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                        "selection_query_ledger_sha256"
                    )
                    for arm in ("correct_same_formula_scope", "same_formula_mismatched")
                ]
                if (
                    None in global_selection + scoped_selection
                    or len(set(global_selection)) != 1
                    or len(set(scoped_selection)) != 1
                ):
                    raise RuntimeError(
                        f"G2b-PEFT fold {fold} controls use different query selection"
                    )
        comparisons = {
            f"{arm}_vs_g1": compare(
                g1["new_rank"], ledger["new_rank"], graph,
                args.bootstrap_resamples,
                args.bootstrap_seed + seed_index * 20 + index,
            )
            for index, (arm, ledger) in enumerate(arms.items())
        }
        comparisons.update({
            "correct_vs_identity_permuted": compare(
                arms["identity_permuted"]["new_rank"], arms["correct"]["new_rank"],
                graph, args.bootstrap_resamples, args.bootstrap_seed + seed_index * 20 + 10,
            ),
            "correct_vs_random_marginal": compare(
                arms["random_marginal"]["new_rank"], arms["correct"]["new_rank"],
                graph, args.bootstrap_resamples, args.bootstrap_seed + seed_index * 20 + 11,
            ),
            "correct_scope_vs_same_formula_mismatched": compare(
                arms["same_formula_mismatched"]["new_rank"],
                arms["correct_same_formula_scope"]["new_rank"],
                graph, args.bootstrap_resamples, args.bootstrap_seed + seed_index * 20 + 12,
            ),
        })
        gradient_direction = []
        for fold in range(args.folds):
            gradient_direction.append({
                "fold": fold,
                "correct_vs_identity_permuted_cosine": signature_cosine(
                    arms["correct"]["decisions"][fold],
                    arms["identity_permuted"]["decisions"][fold],
                ),
                "correct_vs_random_marginal_cosine": signature_cosine(
                    arms["correct"]["decisions"][fold],
                    arms["random_marginal"]["decisions"][fold],
                ),
                "correct_scope_vs_same_formula_mismatched_cosine": signature_cosine(
                    arms["correct_same_formula_scope"]["decisions"][fold],
                    arms["same_formula_mismatched"]["decisions"][fold],
                ),
            })
        all_seed[str(seed)] = {
            "comparisons": comparisons,
            "chemical_gradient_direction": gradient_direction,
            "provenance": {
                "g1": g1["provenance"],
                **{arm: ledger["provenance"] for arm, ledger in arms.items()},
            },
        }
    correct_vs_g1 = [all_seed[str(seed)]["comparisons"]["correct_vs_g1"] for seed in args.seeds]
    correct_vs_permuted = [
        all_seed[str(seed)]["comparisons"]["correct_vs_identity_permuted"]
        for seed in args.seeds
    ]
    correct_vs_random = [
        all_seed[str(seed)]["comparisons"]["correct_vs_random_marginal"]
        for seed in args.seeds
    ]
    scope_vs_mismatch = [
        all_seed[str(seed)]["comparisons"]["correct_scope_vs_same_formula_mismatched"]
        for seed in args.seeds
    ]
    gates = {
        "correct_beats_g1_formula_ci_all_seeds": all(v["delta_recall1_ci_low"] > 0 for v in correct_vs_g1),
        "correct_beats_identity_permuted_ci_all_seeds": all(v["delta_recall1_ci_low"] > 0 for v in correct_vs_permuted),
        "correct_beats_random_marginal_ci_all_seeds": all(v["delta_recall1_ci_low"] > 0 for v in correct_vs_random),
        "correct_scope_beats_same_formula_mismatch_ci_all_seeds": all(v["delta_recall1_ci_low"] > 0 for v in scope_vs_mismatch),
        "correct_near_nonnegative_vs_g1_all_seeds": all(v["near_delta_recall1"] >= 0 for v in correct_vs_g1),
        "correct_risk_net_positive_vs_g1_all_seeds": all(v["risk_net_lambda2"] > 0 for v in correct_vs_g1),
        "correct_gradient_not_identical_to_global_controls_all_folds": all(
            row["correct_vs_identity_permuted_cosine"] < 0.9999
            and row["correct_vs_random_marginal_cosine"] < 0.9999
            for seed in args.seeds
            for row in all_seed[str(seed)]["chemical_gradient_direction"]
        ),
        "correct_scope_gradient_not_identical_to_mismatch_all_folds": all(
            row["correct_scope_vs_same_formula_mismatched_cosine"] < 0.9999
            for seed in args.seeds
            for row in all_seed[str(seed)]["chemical_gradient_direction"]
        ),
    }
    report = {
        "status": "chemaware_shared_v3_g2_peft_multifold_summary_complete",
        "formal": True,
        "folds": args.folds,
        "seeds": args.seeds,
        "arms": list(ARMS),
        "chemical_objective": args.chemical_objective,
        "capacity": capacity_reference,
        "training_contract": training_contract_reference,
        "chemical_contract": json.loads(chemical_contract_reference),
        "per_seed": all_seed,
        "gates": gates,
        "pass_to_G3": bool(all(gates.values())),
        "provenance": {"graph_sha256": sha256_file(args.graph)},
        "claim_limit": (
            "Internal formula-disjoint PEFT G2 attribution only. Passing does not establish "
            "stereochemical resolution or sealed external generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
