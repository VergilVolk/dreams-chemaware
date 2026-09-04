"""Fail-closed decision for the single-fold five-arm ChemAware-v3 G2 pilot.

The pilot is a development gate, not a performance claim.  It permits the
expensive 5-fold x 3-seed x 5-arm matrix only when the correct Morgan teacher
changes the deployable PEFT gradient in an identity-specific direction and
shows a positive, risk-controlled ranking signal relative to G1 and matched
pseudo-teachers on the fixed seed-17/fold-0 outer queries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from summarize_chemaware_shared_v2_g1 import formula_bootstrap  # noqa: E402
from summarize_chemaware_shared_v2_g2 import ARMS  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g1-root", type=Path,
        default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g1",
    )
    parser.add_argument(
        "--g2-root", type=Path,
        default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g2_pilot",
    )
    parser.add_argument(
        "--g1-summary", type=Path,
        default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g1_summary.json",
    )
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_chemaware_shared_v3_g2_pilot_decision.json",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--gradient-cosine-ceiling", type=float, default=0.9999)
    parser.add_argument(
        "--chemical-objective",
        choices=("candidate_hardness", "frozen_probe_targeted"),
        default="candidate_hardness",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="verify an existing passing decision and all pinned pilot artifacts",
    )
    return parser.parse_args()


def _run_dir(root: Path, seed: int, fold: int, arm: str | None) -> Path:
    base = root if arm is None else root / arm
    return base / f"seed_{seed}" / f"fold_{fold}"


def load_run(
    root: Path,
    seed: int,
    fold: int,
    arm: str | None,
    chemical_objective: str = "candidate_hardness",
) -> dict:
    run = _run_dir(root, seed, fold, arm)
    decision_path = run / "decision.json"
    prediction_path = run / "outer_predictions.npz"
    checkpoint_path = run / "peft.pt"
    if not all(path.is_file() for path in (decision_path, prediction_path, checkpoint_path)):
        raise FileNotFoundError(f"incomplete {'G1' if arm is None else arm} pilot run: {run}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    expected_status = (
        "chemaware_shared_v3_g1_peft_fold_complete"
        if arm is None else "chemaware_shared_v3_g2_peft_fold_complete"
    )
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
        or decision.get("preflight", {}).get("complete_split_eligible_candidate_groups") is not True
    ):
        raise RuntimeError(f"invalid {'G1' if arm is None else arm} pilot decision: {decision_path}")
    if arm is not None:
        expected_objective = (
            "frozen_morgan_ridge_probe_targeted_listwise"
            if chemical_objective == "frozen_probe_targeted" else
            "frozen_morgan_binary_connectivity_candidate_hardness_absolute_bounded"
        )
        if (
            decision.get("chemical_objective") != expected_objective
            or decision.get("teacher_kind") != "morgan_binary_connectivity"
            or decision.get("training_only_projector_used") is not False
        ):
            raise RuntimeError(f"invalid Morgan G2 objective: {decision_path}")
        if chemical_objective == "frozen_probe_targeted":
            probe = decision.get("frozen_probe_fit_audit") or {}
            if (
                decision.get("training_only_frozen_probe_used") is not True
                or decision.get("chemical_gradient_absorber_trainable") is not False
                or int(probe.get("trainable_parameters", -1)) != 0
                or probe.get("discarded_at_inference") is not True
            ):
                raise RuntimeError(f"invalid frozen-probe G2b contract: {decision_path}")
    with np.load(prediction_path) as body:
        query = np.asarray(body["query"], dtype=np.int64)
        old_rank = np.asarray(body["old_rank"], dtype=np.int32)
        new_rank = np.asarray(body["new_rank"], dtype=np.int32)
    if (
        query.ndim != 1
        or not (len(query) == len(old_rank) == len(new_rank))
        or not len(query)
        or len(np.unique(query)) != len(query)
        or np.any(old_rank < 1)
        or np.any(new_rank < 1)
    ):
        raise RuntimeError(f"malformed pilot predictions: {prediction_path}")
    order = np.argsort(query, kind="stable")
    return {
        "query": query[order],
        "old_rank": old_rank[order],
        "new_rank": new_rank[order],
        "decision": decision,
        "provenance": {
            "decision_sha256": sha256_file(decision_path),
            "predictions_sha256": sha256_file(prediction_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    }


def assert_matched(g1: dict, g2: dict, arm: str) -> None:
    if not np.array_equal(g1["query"], g2["query"]):
        raise RuntimeError(f"G1/{arm} pilot query ledger mismatch")
    if not np.array_equal(g1["old_rank"], g2["old_rank"]):
        raise RuntimeError(f"G1/{arm} official baseline mismatch")
    keys = (
        "training_query_ledger_sha256", "allowed_molecule_ledger_sha256",
        "initial_peft_state_sha256", "capacity", "training_contract",
        "train_queries", "train_identities",
    )
    for key in keys:
        if g1["decision"].get(key) != g2["decision"].get(key):
            raise RuntimeError(f"G1/{arm} pilot differs in {key}")


def first_chemical_audit(decision: dict) -> dict:
    for row in decision.get("history", [])[1:]:
        audit = row.get("chemical_gradient_audit")
        if audit is not None:
            signature = np.asarray(
                audit.get("chemical_delta_gradient_signature", []), dtype=np.float64
            )
            if (
                float(audit.get("chemical_minus_clean_gradient_norm", 0)) > 1e-12
                and int(audit.get("chemical_delta_nonzero_parameter_tensors", 0)) > 0
                and signature.shape == (128,)
                and np.all(np.isfinite(signature))
                and np.linalg.norm(signature) > 0
            ):
                return {**audit, "signature": signature}
    raise RuntimeError("G2 pilot lacks a valid 128-dimensional chemical gradient audit")


def signature_cosine(left: dict, right: dict) -> float:
    left_signature = first_chemical_audit(left)["signature"]
    right_signature = first_chemical_audit(right)["signature"]
    return float(np.dot(left_signature, right_signature) / (
        np.linalg.norm(left_signature) * np.linalg.norm(right_signature)
    ))


def compare(
    reference: dict,
    candidate: dict,
    graph: CandidateGraph,
    resamples: int,
    seed: int,
) -> dict:
    if not np.array_equal(reference["query"], candidate["query"]):
        raise RuntimeError("pilot comparison query ledgers differ")
    query = reference["query"]
    reference_rank = reference["new_rank"]
    candidate_rank = candidate["new_rank"]
    reference_correct = reference_rank == 1
    candidate_correct = candidate_rank == 1
    near = graph.query_has_near[query]
    corrected = int(np.sum(~reference_correct & candidate_correct))
    introduced = int(np.sum(reference_correct & ~candidate_correct))
    bootstrap = formula_bootstrap(
        reference_rank, candidate_rank, graph.query_formula[query], resamples, seed
    )
    return {
        **bootstrap,
        "queries": int(len(query)),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "rank_changed_queries": int(np.sum(reference_rank != candidate_rank)),
        "near_queries": int(np.sum(near)),
        "near_delta_recall1": (
            float(np.mean(candidate_correct[near]) - np.mean(reference_correct[near]))
            if np.any(near) else None
        ),
    }


def verify_existing(args: argparse.Namespace) -> dict:
    if not args.output.is_file():
        raise FileNotFoundError(args.output)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    if (
        report.get("status") != "chemaware_shared_v3_g2_pilot_decision_complete"
        or report.get("formal") is not True
        or report.get("development_only") is not True
        or report.get("pass_to_full_matrix") is not True
        or not report.get("gates")
        or not all(report["gates"].values())
        or int(report.get("seed", -1)) != args.seed
        or int(report.get("fold", -1)) != args.fold
        or report.get("chemical_objective") != args.chemical_objective
    ):
        raise RuntimeError("existing G2 pilot decision is not a passing formal gate")
    provenance = report.get("provenance") or {}
    if (
        provenance.get("graph_sha256") != sha256_file(args.graph)
        or provenance.get("g1_summary_sha256") != sha256_file(args.g1_summary)
    ):
        raise RuntimeError("existing G2 pilot decision input provenance has changed")
    for arm in (None, *ARMS):
        label = "g1" if arm is None else arm
        expected = provenance.get(label) or {}
        run = _run_dir(
            args.g1_root if arm is None else args.g2_root,
            args.seed, args.fold, arm,
        )
        observed = {
            "decision_sha256": sha256_file(run / "decision.json"),
            "predictions_sha256": sha256_file(run / "outer_predictions.npz"),
            "checkpoint_sha256": sha256_file(run / "peft.pt"),
        }
        if expected != observed:
            raise RuntimeError(f"existing G2 pilot decision no longer pins {label}")
    return report


def main() -> None:
    args = arguments()
    if args.verify_only:
        report = verify_existing(args)
        print(json.dumps({
            "status": "chemaware_shared_v3_g2_pilot_decision_verified",
            "pass_to_full_matrix": report["pass_to_full_matrix"],
            "decision_sha256": sha256_file(args.output),
        }, indent=2), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G2 pilot decision: {args.output}")
    if not (0 < args.gradient_cosine_ceiling < 1):
        raise ValueError("gradient cosine ceiling must be in (0, 1)")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    graph = CandidateGraph(args.graph)
    summary = json.loads(args.g1_summary.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "chemaware_shared_v3_g1_peft_multifold_summary_complete"
        or summary.get("formal") is not True
        or summary.get("matched_capacity_control_ready_for_G2") is not True
        or summary.get("provenance", {}).get("graph_sha256") != sha256_file(args.graph)
    ):
        raise RuntimeError("G1 summary is not a valid matched-capacity prerequisite")

    g1 = load_run(args.g1_root, args.seed, args.fold, None, args.chemical_objective)
    arms = {
        arm: load_run(
            args.g2_root, args.seed, args.fold, arm, args.chemical_objective
        ) for arm in ARMS
    }
    for arm, run in arms.items():
        assert_matched(g1, run, arm)
        first_chemical_audit(run["decision"])

    contracts = {
        json.dumps(run["decision"].get("chemical_contract"), sort_keys=True)
        for run in arms.values()
    }
    if len(contracts) != 1:
        raise RuntimeError("G2 pilot controls use different chemical contracts")
    audits = {
        arm: run["decision"].get("teacher_control_audit") or {}
        for arm, run in arms.items()
    }
    global_arms = ("correct", "identity_permuted", "random_marginal")
    scoped_arms = ("correct_same_formula_scope", "same_formula_mismatched")
    for group in (global_arms, scoped_arms):
        coverages = {audits[arm].get("chemical_effect_queries") for arm in group}
        masks = {audits[arm].get("teacher_observable_mask_sha256") for arm in group}
        if None in coverages or len(coverages) != 1 or min(coverages) <= 0:
            raise RuntimeError(f"G2 pilot controls have unmatched coverage: {group}")
        if None in masks or len(masks) != 1:
            raise RuntimeError(f"G2 pilot controls have unmatched observable masks: {group}")
        if args.chemical_objective == "frozen_probe_targeted":
            selections = {
                audits[arm].get("selection_query_ledger_sha256") for arm in group
            }
            if None in selections or len(selections) != 1:
                raise RuntimeError(f"G2b pilot controls have unmatched query selection: {group}")

    comparisons = {
        "correct_vs_g1": compare(
            g1, arms["correct"], graph, args.bootstrap_resamples,
            args.bootstrap_seed,
        ),
        "correct_vs_identity_permuted": compare(
            arms["identity_permuted"], arms["correct"], graph,
            args.bootstrap_resamples, args.bootstrap_seed + 1,
        ),
        "correct_vs_random_marginal": compare(
            arms["random_marginal"], arms["correct"], graph,
            args.bootstrap_resamples, args.bootstrap_seed + 2,
        ),
        "correct_scope_vs_same_formula_mismatched": compare(
            arms["same_formula_mismatched"],
            arms["correct_same_formula_scope"], graph,
            args.bootstrap_resamples, args.bootstrap_seed + 3,
        ),
    }
    gradient_direction = {
        "correct_vs_identity_permuted_cosine": signature_cosine(
            arms["correct"]["decision"], arms["identity_permuted"]["decision"]
        ),
        "correct_vs_random_marginal_cosine": signature_cosine(
            arms["correct"]["decision"], arms["random_marginal"]["decision"]
        ),
        "correct_scope_vs_same_formula_mismatched_cosine": signature_cosine(
            arms["correct_same_formula_scope"]["decision"],
            arms["same_formula_mismatched"]["decision"],
        ),
    }
    g1_comparison = comparisons["correct_vs_g1"]
    global_comparisons = (
        comparisons["correct_vs_identity_permuted"],
        comparisons["correct_vs_random_marginal"],
    )
    scoped_comparison = comparisons["correct_scope_vs_same_formula_mismatched"]
    gates = {
        "correct_changes_outer_ranking_vs_g1": g1_comparison["rank_changed_queries"] > 0,
        "correct_risk_net_positive_vs_g1": g1_comparison["risk_net_lambda2"] > 0,
        "correct_near_nonnegative_vs_g1": (
            g1_comparison["near_delta_recall1"] is None
            or g1_comparison["near_delta_recall1"] >= 0
        ),
        "correct_risk_net_positive_vs_global_controls": all(
            value["risk_net_lambda2"] > 0 for value in global_comparisons
        ),
        "correct_scope_risk_net_positive_vs_mismatch": (
            scoped_comparison["risk_net_lambda2"] > 0
        ),
        "correct_gradient_differs_from_global_controls": (
            gradient_direction["correct_vs_identity_permuted_cosine"]
            < args.gradient_cosine_ceiling
            and gradient_direction["correct_vs_random_marginal_cosine"]
            < args.gradient_cosine_ceiling
        ),
        "correct_scope_gradient_differs_from_mismatch": (
            gradient_direction["correct_scope_vs_same_formula_mismatched_cosine"]
            < args.gradient_cosine_ceiling
        ),
    }
    report = {
        "status": "chemaware_shared_v3_g2_pilot_decision_complete",
        "formal": True,
        "development_only": True,
        "seed": args.seed,
        "fold": args.fold,
        "chemical_objective": args.chemical_objective,
        "arms": list(ARMS),
        "comparisons": comparisons,
        "chemical_gradient_direction": gradient_direction,
        "gradient_cosine_ceiling": args.gradient_cosine_ceiling,
        "gates": gates,
        "pass_to_full_matrix": bool(all(gates.values())),
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "g1_summary_sha256": sha256_file(args.g1_summary),
            "g1": g1["provenance"],
            **{arm: run["provenance"] for arm, run in arms.items()},
        },
        "claim_limit": (
            "Single seed/fold development gate only. Passing authorizes the paired full "
            "matrix; it is not evidence of a reproducible performance gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_full_matrix"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
