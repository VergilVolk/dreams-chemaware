"""Audit why ChemAware development policies correct or harm different queries.

This diagnostic accepts only the frozen role-3 ``inner_policy.npz`` ledger.
It has no manifest argument and therefore cannot read the consumed role-4
outer labels.  The report is mechanism evidence, not a model-selection result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from chemaware_development_disagreement_core import (
    action_outcome,
    arbitrate_two_policies,
    formula_cluster_ci,
    intervention_labels,
    pair_audit,
    selected_utility_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_INNER_SHA256 = "70e25e6d2459222be3597748a851f206c5121792cf920e8e745ed2caa6dcace3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inner-ledger", type=Path,
        default=ROOT / (
            "data/validation/chemaware_truthblind_candidate_policy/"
            "run_2338337/policy/inner_policy.npz"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_development_policy_disagreement_v1",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if not args.inner_ledger.is_file():
        raise FileNotFoundError(args.inner_ledger)
    observed = sha256(args.inner_ledger)
    if observed != CANONICAL_INNER_SHA256:
        raise RuntimeError(
            f"canonical inner ledger drifted: expected {CANONICAL_INNER_SHA256}, observed {observed}"
        )
    with np.load(args.inner_ledger, allow_pickle=False) as loaded:
        required = {
            "query", "formula", "baseline_rank", "correct_rank",
            "correct_selected_candidate_slot", "correct_candidate_utility",
            "same_feature_direct_rank", "same_feature_direct_selected_candidate_slot",
            "same_feature_direct_candidate_utility", "nuisance_only_rank",
            "nuisance_only_selected_candidate_slot", "nuisance_only_candidate_utility",
            "zero_contrast_rank", "zero_contrast_selected_candidate_slot",
            "alignment_permuted_rank", "alignment_permuted_selected_candidate_slot",
        }
        absent = sorted(required - set(loaded.files))
        if absent:
            raise RuntimeError(f"canonical inner fields absent: {absent}")
        body = {key: np.asarray(loaded[key]) for key in required}

    baseline = body["baseline_rank"]
    policies = {
        "chemaware_residual": {
            "rank": body["correct_rank"],
            "slot": body["correct_selected_candidate_slot"],
            "utility": body["correct_candidate_utility"],
        },
        "same_feature_direct": {
            "rank": body["same_feature_direct_rank"],
            "slot": body["same_feature_direct_selected_candidate_slot"],
            "utility": body["same_feature_direct_candidate_utility"],
        },
        "nuisance_only": {
            "rank": body["nuisance_only_rank"],
            "slot": body["nuisance_only_selected_candidate_slot"],
            "utility": body["nuisance_only_candidate_utility"],
        },
    }
    policy_summary = {}
    for name, policy in policies.items():
        outcome = intervention_labels(baseline, policy["rank"])
        policy_summary[name] = {
            "retrieval": action_outcome(baseline, policy["rank"]),
            "selected_utility": selected_utility_summary(policy["slot"], policy["utility"]),
            "formula_cluster_ci_delta_recall1": formula_cluster_ci(
                body["formula"], outcome, draws=args.bootstrap_draws,
                seed=args.seed + len(policy_summary),
            ),
        }

    pairs = {}
    comparisons = (
        ("chemaware_residual", "same_feature_direct"),
        ("chemaware_residual", "nuisance_only"),
        ("same_feature_direct", "nuisance_only"),
    )
    for left, right in comparisons:
        pairs[f"{left}_vs_{right}"] = pair_audit(
            left, right, baseline,
            policies[left]["rank"], policies[right]["rank"],
            policies[left]["slot"], policies[right]["slot"],
        )

    # These are truth-aware diagnostic strata, never deployable gates.  They
    # quantify whether safer arbitration has headroom before a new model exists.
    residual_action = policies["chemaware_residual"]["slot"] >= 0
    direct_action = policies["same_feature_direct"]["slot"] >= 0
    same_action = residual_action & direct_action & (
        policies["chemaware_residual"]["slot"] == policies["same_feature_direct"]["slot"]
    )
    residual_outcome = intervention_labels(baseline, policies["chemaware_residual"]["rank"])
    diagnostic = {
        "residual_actions_not_confirmed_by_same_direct_action": {
            "queries": int(np.sum(residual_action & ~same_action)),
            "corrections": int(np.sum(residual_outcome[residual_action & ~same_action] == 1)),
            "introductions": int(np.sum(residual_outcome[residual_action & ~same_action] == -1)),
        },
        "residual_actions_confirmed_by_same_direct_action": {
            "queries": int(np.sum(same_action)),
            "corrections": int(np.sum(residual_outcome[same_action] == 1)),
            "introductions": int(np.sum(residual_outcome[same_action] == -1)),
        },
        "warning": "Truth-aware mechanism diagnostic only; no row-level outcome may enter deployment inference.",
    }
    deterministic_arbitration = {}
    arbitration_recipes = {
        "direct_then_residual": ("same_feature_direct", "chemaware_residual"),
        "residual_then_direct": ("chemaware_residual", "same_feature_direct"),
        "direct_then_nuisance": ("same_feature_direct", "nuisance_only"),
        "residual_then_nuisance": ("chemaware_residual", "nuisance_only"),
    }
    for index, (recipe, (primary_name, fallback_name)) in enumerate(arbitration_recipes.items()):
        primary = policies[primary_name]
        fallback = policies[fallback_name]
        rank, slot, source = arbitrate_two_policies(
            baseline, primary["rank"], primary["slot"],
            fallback["rank"], fallback["slot"],
        )
        outcome = intervention_labels(baseline, rank)
        primary_outcome = intervention_labels(baseline, primary["rank"])
        deterministic_arbitration[recipe] = {
            "primary": primary_name,
            "fallback": fallback_name,
            "primary_actions": int(np.sum(source == 1)),
            "fallback_actions": int(np.sum(source == 2)),
            "abstained": int(np.sum(source == 0)),
            "retrieval": action_outcome(baseline, rank),
            "formula_cluster_ci_delta_recall1": formula_cluster_ci(
                body["formula"], outcome, draws=args.bootstrap_draws,
                seed=args.seed + 100 + index,
            ),
            "increment_over_primary_delta_recall1": float(
                np.mean(rank == 1) - np.mean(primary["rank"] == 1)
            ),
            "increment_over_primary_formula_cluster_ci": formula_cluster_ci(
                body["formula"], outcome - primary_outcome,
                draws=args.bootstrap_draws, seed=args.seed + 200 + index,
            ),
            "claim_limit": (
                "Deterministic post-hoc role-3 development composition; "
                "not selected on validation and not evaluated on role 4."
            ),
        }
    report = {
        "status": "CHEMAWARE_DEVELOPMENT_POLICY_DISAGREEMENT_COMPLETE",
        "scope": "canonical run 2338337 formula role 3 only",
        "outer_role_4_accessed": False,
        "model_selected": False,
        "threshold_tuned": False,
        "queries": int(len(baseline)),
        "formulas": int(len(np.unique(body["formula"].astype(str)))),
        "policies": policy_summary,
        "pairwise_action_partitions": pairs,
        "deterministic_two_expert_arbitration": deterministic_arbitration,
        "truth_aware_headroom_diagnostic": diagnostic,
        "provenance": {
            "inner_ledger": str(args.inner_ledger),
            "inner_ledger_sha256": observed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_development_disagreement_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
