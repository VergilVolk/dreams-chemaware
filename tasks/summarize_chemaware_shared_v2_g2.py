"""Paired multifold decision for ChemAware-v2 MoLFormer G2 and controls."""
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


ARMS = (
    "correct", "identity_permuted", "random_marginal",
    "correct_same_formula_scope", "same_formula_mismatched",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_g1")
    parser.add_argument("--g2-root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_g2")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_g2_summary.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 73])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    return parser.parse_args()


def load_seed(
    root: Path,
    seed: int,
    folds: int,
    graph: CandidateGraph,
    arm: str | None,
) -> dict:
    query_parts, old_parts, new_parts = [], [], []
    decisions, provenance = [], []
    expected_status = (
        "chemaware_shared_v2_g1_fold_complete"
        if arm is None else "chemaware_shared_v2_g2_fold_complete"
    )
    for fold in range(folds):
        run = root / f"seed_{seed}" / f"fold_{fold}"
        decision_path = run / "decision.json"
        prediction_path = run / "outer_predictions.npz"
        if not decision_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"incomplete {'G1' if arm is None else arm} run: {run}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            decision.get("status") != expected_status
            or decision.get("formal") is not True
            or int(decision.get("seed", -1)) != seed
            or int(decision.get("outer_fold", -1)) != fold
            or bool(decision.get("chemical_supervision")) != (arm is not None)
            or (arm is not None and decision.get("teacher_control") != arm)
        ):
            raise RuntimeError(f"invalid {'G1' if arm is None else arm} decision: {decision_path}")
        with np.load(prediction_path) as body:
            query_parts.append(np.asarray(body["query"], dtype=np.int64))
            old_parts.append(np.asarray(body["old_rank"], dtype=np.int32))
            new_parts.append(np.asarray(body["new_rank"], dtype=np.int32))
        decisions.append(decision)
        provenance.append({
            "fold": fold,
            "decision_sha256": sha256_file(decision_path),
            "predictions_sha256": sha256_file(prediction_path),
        })
    query = np.concatenate(query_parts)
    order = np.argsort(query, kind="stable")
    query = query[order]
    old_rank = np.concatenate(old_parts)[order]
    new_rank = np.concatenate(new_parts)[order]
    if not np.array_equal(query, np.arange(graph.n_queries)):
        raise RuntimeError(f"seed {seed} arm {arm} does not cover each query once")
    return {
        "query": query, "old_rank": old_rank, "new_rank": new_rank,
        "decisions": decisions, "provenance": provenance,
    }


def compare(
    reference: np.ndarray,
    candidate: np.ndarray,
    graph: CandidateGraph,
    resamples: int,
    seed: int,
) -> dict:
    reference_correct, candidate_correct = reference == 1, candidate == 1
    near = graph.query_has_near
    bootstrap = formula_bootstrap(
        reference, candidate, graph.query_formula, resamples, seed
    )
    corrected = int(np.sum(~reference_correct & candidate_correct))
    introduced = int(np.sum(reference_correct & ~candidate_correct))
    return {
        **bootstrap,
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "near_delta_recall1": float(
            np.mean(candidate_correct[near]) - np.mean(reference_correct[near])
        ),
    }


def assert_matched(g1: dict, g2: dict, arm: str) -> None:
    if not np.array_equal(g1["query"], g2["query"]):
        raise RuntimeError(f"G1/{arm} query ledger mismatch")
    if not np.array_equal(g1["old_rank"], g2["old_rank"]):
        raise RuntimeError(f"G1/{arm} official baseline mismatch")
    keys = (
        "training_query_ledger_sha256", "allowed_molecule_ledger_sha256",
        "initial_adapter_sha256", "training_contract", "train_queries", "train_identities",
    )
    for fold, (clean, chemical) in enumerate(zip(g1["decisions"], g2["decisions"])):
        if (
            chemical.get("chemical_objective") not in {
                "frozen_teacher_candidate_hardness_reweighting",
                "frozen_teacher_candidate_hardness_relative_centered",
                "frozen_teacher_candidate_hardness_absolute_bounded",
            }
            or chemical.get("training_only_projector_used") is not False
        ):
            raise RuntimeError(f"G2 {arm} fold {fold} uses an invalid chemical objective")
        gradient_history = [
            row.get("gradient_audit") for row in chemical.get("history", [])[1:]
        ]
        if (
            not gradient_history
            or any(
                audit is None
                or float(audit.get("chemical_minus_clean_gradient_norm", 0.0)) <= 1e-12
                or int(audit.get("chemical_delta_nonzero_parameter_tensors", 0)) <= 0
                for audit in gradient_history
            )
        ):
            raise RuntimeError(f"G2 {arm} fold {fold} lacks deployable chemical-gradient evidence")
        for key in keys:
            if clean.get(key) != chemical.get(key):
                raise RuntimeError(f"G1/{arm} fold {fold} unmatched {key}")


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G2 summary: {args.output}")
    graph = CandidateGraph(args.graph)
    all_seed = {}
    for seed_index, seed in enumerate(args.seeds):
        g1 = load_seed(args.g1_root, seed, args.folds, graph, None)
        arms = {
            arm: load_seed(args.g2_root / arm, seed, args.folds, graph, arm)
            for arm in ARMS
        }
        for arm, ledger in arms.items():
            assert_matched(g1, ledger, arm)
        for fold in range(args.folds):
            contracts = [
                json.dumps(arms[arm]["decisions"][fold].get("chemical_contract"), sort_keys=True)
                for arm in ARMS
            ]
            if len(set(contracts)) != 1:
                raise RuntimeError(f"G2 fold {fold} chemical contracts differ across controls")
            global_coverage = [
                arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                    "chemical_effect_queries"
                )
                for arm in ("correct", "identity_permuted", "random_marginal")
            ]
            scoped_coverage = [
                arms[arm]["decisions"][fold].get("teacher_control_audit", {}).get(
                    "chemical_effect_queries"
                )
                for arm in ("correct_same_formula_scope", "same_formula_mismatched")
            ]
            if None in global_coverage + scoped_coverage:
                raise RuntimeError(f"G2 fold {fold} lacks chemical-effect coverage audit")
            if len(set(global_coverage)) != 1 or len(set(scoped_coverage)) != 1:
                raise RuntimeError(f"G2 fold {fold} controls use different observable coverage")
        comparisons = {
            f"{arm}_vs_g1": compare(
                g1["new_rank"], ledger["new_rank"], graph,
                args.bootstrap_resamples, args.bootstrap_seed + seed_index * 20 + index,
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
        all_seed[str(seed)] = {
            "comparisons": comparisons,
            "provenance": {
                "g1": g1["provenance"],
                **{arm: ledger["provenance"] for arm, ledger in arms.items()},
            },
        }
    correct_vs_g1 = [
        all_seed[str(seed)]["comparisons"]["correct_vs_g1"] for seed in args.seeds
    ]
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
    }
    report = {
        "status": "chemaware_shared_v2_g2_multifold_summary_complete",
        "formal": True,
        "folds": args.folds,
        "seeds": args.seeds,
        "arms": list(ARMS),
        "per_seed": all_seed,
        "gates": gates,
        "pass_to_G3": bool(all(gates.values())),
        "provenance": {"graph_sha256": sha256_file(args.graph)},
        "claim_limit": (
            "Internal formula-disjoint G2 attribution only. MoLFormer is a connectivity "
            "teacher that removes stereochemistry; passing does not establish stereochemical "
            "resolution or sealed external generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
