"""Audit the direct-first residual-backoff policy on development roles 2 and 3.

The two experts and both thresholds must already be frozen.  This script fits
nothing and has no manifest argument.  It requires role-2 and role-3 ledgers
generated together by the updated development serializer; role 4 is absent.
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
)


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_direct_first_backoff_v1",
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
    with np.load(path, allow_pickle=False) as loaded:
        absent = sorted(required - set(loaded.files))
        if absent:
            raise RuntimeError(
                f"{path} lacks direct-first development fields: {absent}; "
                "regenerate roles 0-3 only with the updated serializer"
            )
        return {key: np.asarray(loaded[key]) for key in required}


def evaluate(body: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, object]:
    rank, slot, source = arbitrate_two_policies(
        body["baseline_rank"],
        body["same_feature_direct_rank"],
        body["same_feature_direct_selected_candidate_slot"],
        body["correct_rank"], body["correct_selected_candidate_slot"],
    )
    backoff_outcome = intervention_labels(body["baseline_rank"], rank)
    direct_outcome = intervention_labels(
        body["baseline_rank"], body["same_feature_direct_rank"],
    )
    increment = backoff_outcome - direct_outcome
    return {
        "direct": action_outcome(body["baseline_rank"], body["same_feature_direct_rank"]),
        "residual": action_outcome(body["baseline_rank"], body["correct_rank"]),
        "direct_first_backoff": action_outcome(body["baseline_rank"], rank),
        "primary_actions": int(np.sum(source == 1)),
        "fallback_actions": int(np.sum(source == 2)),
        "abstained": int(np.sum(source == 0)),
        "increment_over_direct_delta_recall1": float(np.mean(increment)),
        "increment_over_direct_formula_cluster_ci": formula_cluster_ci(
            body["formula"], increment, draws=draws, seed=seed,
        ),
        "fallback_corrected": int(np.sum((source == 2) & (backoff_outcome == 1))),
        "fallback_introduced": int(np.sum((source == 2) & (backoff_outcome == -1))),
        "selected_slot_sha256": hashlib.sha256(
            np.ascontiguousarray(slot).view(np.uint8)
        ).hexdigest(),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    validation_path = args.policy_dir / "validation_policy.npz"
    inner_path = args.policy_dir / "inner_policy.npz"
    for path in (validation_path, inner_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = evaluate(load(validation_path), args.bootstrap_draws, args.seed + 1)
    inner = evaluate(load(inner_path), args.bootstrap_draws, args.seed + 2)
    report = {
        "status": "CHEMAWARE_DIRECT_FIRST_BACKOFF_DEVELOPMENT_COMPLETE",
        "scope": "formula role 2 retrospective replay and role 3 development confirmation",
        "outer_role_4_accessed": False,
        "model_fit": False,
        "threshold_tuned": False,
        "policy": (
            "same-feature direct action has priority; residual ChemAware acts only "
            "when direct abstains; otherwise retain official DreaMS"
        ),
        "validation_role_2": validation,
        "held_inner_role_3": inner,
        "development_gate": {
            "validation_increment_ci_positive": validation[
                "increment_over_direct_formula_cluster_ci"
            ][0] > 0,
            "inner_increment_ci_positive": inner[
                "increment_over_direct_formula_cluster_ci"
            ][0] > 0,
            "inner_corrected_exceeds_twice_introduced": (
                inner["direct_first_backoff"]["corrected_at_1"]
                > 2 * inner["direct_first_backoff"]["introduced_at_1"]
            ),
        },
        "claim_limit": (
            "Post-outer development method. Passing only authorizes a newly frozen "
            "independent external evaluation; it cannot revise role 4."
        ),
        "provenance": {
            "validation_policy_sha256": sha256(validation_path),
            "inner_policy_sha256": sha256(inner_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_direct_first_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
