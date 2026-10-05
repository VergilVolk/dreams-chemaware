"""Development-only cross-calibration of the frozen ChemAware utility.

Thresholds are selected out-of-fold across formula clusters inside role 2.
The final conservative threshold is then audited once on the already-consumed
role 3 development ledger.  No model is fit and role 4 is structurally absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from chemaware_formula_crosscalibration_core import (
    crosscalibrate,
    formula_cluster_ci,
    ranks_at_threshold,
    reward,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_VALIDATION_SHA256 = "327a1e9b4f40e022ead2f09293c1272287cef4acf2fd8c48445091aa4520b6fb"
CANONICAL_INNER_SHA256 = "70e25e6d2459222be3597748a851f206c5121792cf920e8e745ed2caa6dcace3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/validation/chemaware_truthblind_candidate_policy/run_2338337/policy"
    parser.add_argument("--validation-ledger", type=Path, default=base / "validation_policy.npz")
    parser.add_argument("--inner-ledger", type=Path, default=base / "inner_policy.npz")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_formula_crosscalibration_v1",
    )
    parser.add_argument("--roles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--min-selected-formulas", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ledger(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "query", "formula", "baseline_rank", "proposal_rank",
            "correct_candidate_utility", "zero_contrast_candidate_utility",
            "reversed_contrast_candidate_utility", "alignment_permuted_candidate_utility",
        }
        absent = sorted(required - set(loaded.files))
        if absent:
            raise RuntimeError(f"ledger fields absent from {path}: {absent}")
        return {key: np.asarray(loaded[key]) for key in required}


def utilities(body: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "correct": body["correct_candidate_utility"],
        "zero_contrast": body["zero_contrast_candidate_utility"],
        "reversed_contrast": body["reversed_contrast_candidate_utility"],
        "alignment_permuted": body["alignment_permuted_candidate_utility"],
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    expected = {
        args.validation_ledger: CANONICAL_VALIDATION_SHA256,
        args.inner_ledger: CANONICAL_INNER_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != digest:
            raise RuntimeError(f"frozen ledger drifted: {path}: {observed}")
    validation = load_ledger(args.validation_ledger)
    inner = load_ledger(args.inner_ledger)
    calibration = crosscalibrate(
        validation["baseline_rank"], validation["proposal_rank"], utilities(validation),
        validation["formula"], roles=args.roles, seed=args.seed,
        risk_penalty=args.risk_penalty,
        min_selected_formulas=args.min_selected_formulas,
        bootstrap_draws=args.bootstrap_draws,
    )
    threshold = float(calibration["deployment_threshold"])
    inner_result = {}
    inner_rewards = {}
    for index, (name, utility) in enumerate(utilities(inner).items()):
        rank, selected, _ = ranks_at_threshold(
            inner["baseline_rank"], inner["proposal_rank"], utility, threshold,
        )
        values = reward(inner["baseline_rank"], rank, args.risk_penalty)
        inner_rewards[name] = values
        inner_result[name] = {
            **summarize(
                inner["baseline_rank"], rank, selected, inner["formula"], args.risk_penalty,
            ),
            "formula_cluster_ci_mean_risk_reward": formula_cluster_ci(
                inner["formula"], values, draws=args.bootstrap_draws,
                seed=args.seed + 500 + index,
            ),
        }
    for index, name in enumerate(x for x in inner_result if x != "correct"):
        inner_result[name]["correct_minus_control_formula_ci_mean_risk_reward"] = (
            formula_cluster_ci(
                inner["formula"], inner_rewards["correct"] - inner_rewards[name],
                draws=args.bootstrap_draws, seed=args.seed + 600 + index,
            )
        )
    canonical_threshold = float(np.load(args.inner_ledger, allow_pickle=False)["selected_threshold"])
    report = {
        "status": "CHEMAWARE_FORMULA_CROSSCALIBRATION_DEVELOPMENT_COMPLETE",
        "scope": "role 2 cross-calibration; role 3 development audit; role 4 inaccessible",
        "outer_role_4_accessed": False,
        "model_fit": False,
        "candidate_utility_changed": False,
        "canonical_threshold": canonical_threshold,
        "crosscalibrated_threshold": threshold,
        "calibration": calibration,
        "held_inner_development": inner_result,
        "claim_limit": (
            "Post-outer development diagnostic on already-consumed roles 2-3; "
            "requires a new external panel and cannot revise the completed role-4 result."
        ),
        "provenance": {
            "validation_sha256": expected[args.validation_ledger],
            "inner_sha256": expected[args.inner_ledger],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_crosscalibration_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
