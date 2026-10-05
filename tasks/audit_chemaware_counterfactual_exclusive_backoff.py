"""Audit deployment-safe counterfactual-exclusive ChemAware fallback.

The method is fixed before evaluation: same-feature direct is the primary
expert.  Residual ChemAware may fill a direct abstention only when its correct
chemistry arm acts and zero, reversed, and within-query candidate-rotated null
arms all abstain.  The audit fits nothing, tunes no threshold, has no manifest
argument, and reads only formula roles 2 and 3 development ledgers.
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
    arbitrate_exclusive_fallback,
    arbitrate_two_policies,
    formula_cluster_ci,
    intervention_labels,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = (
    "zero_contrast", "reversed_contrast", "candidate_rotated_truthblind",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_counterfactual_exclusive_backoff_v1",
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


def load(path: Path) -> dict[str, np.ndarray]:
    required = {
        "query", "formula", "baseline_rank", "correct_rank",
        "correct_selected_candidate_slot", "same_feature_direct_rank",
        "same_feature_direct_selected_candidate_slot",
    }
    for name in CONTROL_NAMES:
        required.add(f"{name}_selected_candidate_slot")
    with np.load(path, allow_pickle=False) as loaded:
        absent = sorted(required - set(loaded.files))
        if absent:
            raise RuntimeError(
                f"{path} lacks exclusive-backoff fields: {absent}; regenerate "
                "formula roles 0-3 with the updated truth-blind serializer"
            )
        return {key: np.asarray(loaded[key]) for key in required}


def evaluate(body: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, object]:
    baseline = body["baseline_rank"]
    direct_rank = body["same_feature_direct_rank"]
    direct_slot = body["same_feature_direct_selected_candidate_slot"]
    residual_rank = body["correct_rank"]
    residual_slot = body["correct_selected_candidate_slot"]
    controls = np.stack([
        body[f"{name}_selected_candidate_slot"] for name in CONTROL_NAMES
    ], axis=1)
    simple_rank, _, simple_source = arbitrate_two_policies(
        baseline, direct_rank, direct_slot, residual_rank, residual_slot,
    )
    rank, slot, source, exclusive = arbitrate_exclusive_fallback(
        baseline, direct_rank, direct_slot, residual_rank, residual_slot, controls,
    )
    direct_outcome = intervention_labels(baseline, direct_rank)
    outcome = intervention_labels(baseline, rank)
    simple_outcome = intervention_labels(baseline, simple_rank)
    increment = outcome - direct_outcome
    return {
        "direct": action_outcome(baseline, direct_rank),
        "unrestricted_direct_first_backoff": action_outcome(baseline, simple_rank),
        "counterfactual_exclusive_backoff": action_outcome(baseline, rank),
        "primary_actions": int(np.sum(source == 1)),
        "unrestricted_fallback_actions": int(np.sum(simple_source == 2)),
        "exclusive_fallback_actions": int(np.sum(source == 2)),
        "counterfactual_exclusive_candidates": int(np.sum(exclusive)),
        "fallback_rejected_by_any_null": int(np.sum(
            (direct_slot < 0) & (residual_slot >= 0) & np.any(controls >= 0, axis=1)
        )),
        "fallback_corrected": int(np.sum((source == 2) & (outcome == 1))),
        "fallback_introduced": int(np.sum((source == 2) & (outcome == -1))),
        "increment_over_direct_delta_recall1": float(np.mean(increment)),
        "increment_over_direct_formula_cluster_ci": formula_cluster_ci(
            body["formula"], increment, draws=draws, seed=seed,
        ),
        "difference_from_unrestricted_delta_recall1": float(
            np.mean(outcome - simple_outcome)
        ),
        "selected_slot_sha256": hashlib.sha256(
            np.ascontiguousarray(slot).view(np.uint8)
        ).hexdigest(),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    paths = {
        "validation_role_2": args.policy_dir / "validation_policy.npz",
        "held_inner_role_3": args.policy_dir / "inner_policy.npz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    results = {
        name: evaluate(load(path), args.bootstrap_draws, args.seed + index)
        for index, (name, path) in enumerate(paths.items(), start=1)
    }
    validation = results["validation_role_2"]
    inner = results["held_inner_role_3"]
    report = {
        "status": "CHEMAWARE_COUNTERFACTUAL_EXCLUSIVE_BACKOFF_DEVELOPMENT_COMPLETE",
        "scope": "formula role 2 replay and formula role 3 development confirmation",
        "outer_role_4_accessed": False,
        "model_fit": False,
        "threshold_tuned": False,
        "deployment_safe_controls": list(CONTROL_NAMES),
        "policy": (
            "direct first; residual ChemAware only on direct abstention when correct "
            "chemistry acts and every deployment-safe counterfactual null abstains"
        ),
        **results,
        "development_gate": {
            "validation_increment_ci_positive": validation[
                "increment_over_direct_formula_cluster_ci"
            ][0] > 0,
            "inner_increment_ci_positive": inner[
                "increment_over_direct_formula_cluster_ci"
            ][0] > 0,
            "inner_corrected_exceeds_twice_introduced": (
                inner["counterfactual_exclusive_backoff"]["corrected_at_1"]
                > 2 * inner["counterfactual_exclusive_backoff"]["introduced_at_1"]
            ),
        },
        "claim_limit": (
            "Post-outer development method. Passing authorizes only a newly frozen "
            "independent external evaluation and cannot revise consumed formula role 4."
        ),
        "provenance": {name: sha256(path) for name, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_exclusive_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
