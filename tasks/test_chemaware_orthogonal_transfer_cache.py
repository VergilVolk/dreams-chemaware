"""Fail-closed contracts for the orthogonal-policy transfer cache.

The cache is the only boundary between the frozen candidate-conditioned
teacher and a deployable shared-spectrum student.  These checks deliberately
recompute selection and ranks from the dense candidate arrays instead of
trusting summary fields in ``report.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_orthogonal_rule_residual_policy import array_sha256  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402


ARMS = ("correct", "zero_contrast", "reversed_contrast", "alignment_permuted")


def validate_split(
    path: Path, report: dict, split: str, expected_fold: int,
    manifest: dict[str, np.ndarray],
) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {
        "query", "formula", "identity", "baseline_rank", "valid_candidate",
        "proposed_candidate", "proposal_rank", "baseline_candidate",
        "selected_dose", "selected_threshold",
    }
    required.update(f"{arm}_rank" for arm in ARMS)
    required.update(f"{arm}_selected_candidate_slot" for arm in ARMS)
    required.update(f"{arm}_candidate_utility" for arm in ARMS)
    missing = required - set(body)
    if missing:
        raise RuntimeError(f"{split} transfer cache is missing arrays: {sorted(missing)}")

    query = body["query"].astype(np.int64)
    formula = body["formula"].astype(str)
    identity = body["identity"].astype(str)
    valid = body["valid_candidate"].astype(bool)
    proposal_rank = body["proposal_rank"].astype(np.int64)
    proposed_candidate = body["proposed_candidate"].astype(np.int64)
    baseline_candidate = body["baseline_candidate"].astype(np.int64)
    baseline_rank = body["baseline_rank"].astype(np.int64)
    shape = valid.shape
    if shape[0] != len(query) or proposal_rank.shape != shape or proposed_candidate.shape != shape:
        raise RuntimeError(f"{split} candidate array shapes are inconsistent")
    if np.any(valid.sum(axis=1) < 1):
        raise RuntimeError(f"{split} contains a query without a challenger")
    if np.any(proposed_candidate[valid] < 0) or np.any(proposed_candidate[~valid] != -1):
        raise RuntimeError(f"{split} proposed-candidate padding is invalid")
    if np.any(proposed_candidate[valid] == np.repeat(baseline_candidate, valid.sum(axis=1))):
        raise RuntimeError(f"{split} challenger repeats the official baseline candidate")
    if not np.array_equal(formula, manifest["query_formula"][query].astype(str)):
        raise RuntimeError(f"{split} formulas do not match the source manifest")
    if not np.array_equal(identity, manifest["query_ik14"][query].astype(str)):
        raise RuntimeError(f"{split} identities do not match the source manifest")
    folds = stable_formula_folds(
        manifest["query_formula"],
        int(report["replay_contract"]["arguments"]["folds"]),
        int(report["replay_contract"]["arguments"]["fold_seed"]),
    )
    if np.any(folds[query] != expected_fold):
        raise RuntimeError(f"{split} cache contains a query outside formula fold {expected_fold}")
    if array_sha256(query) != report["replay_contract"]["query_sha256"][split]:
        raise RuntimeError(f"{split} query hash does not replay the report")

    threshold = float(body["selected_threshold"])
    if not np.isclose(threshold, float(report["selection"]["threshold"]), rtol=0, atol=1e-12):
        raise RuntimeError(f"{split} threshold differs from the frozen report")
    if not np.isclose(float(body["selected_dose"]), float(report["selection"]["dose"]), rtol=0, atol=1e-12):
        raise RuntimeError(f"{split} chemical dose differs from the frozen report")

    counts: dict[str, int] = {}
    rows = np.arange(len(query))
    for arm in ARMS:
        utility = body[f"{arm}_candidate_utility"].astype(np.float64)
        selected = body[f"{arm}_selected_candidate_slot"].astype(np.int64)
        stored_rank = body[f"{arm}_rank"].astype(np.int64)
        if utility.shape != shape:
            raise RuntimeError(f"{split}/{arm} utility shape is inconsistent")
        if np.any(np.isfinite(utility[~valid])) or np.any(~np.isfinite(utility[valid])):
            raise RuntimeError(f"{split}/{arm} utility padding is not fail-closed")
        best_slot = np.argmax(utility, axis=1)
        best_value = utility[rows, best_slot]
        expected_selected = np.where(best_value >= threshold, best_slot, -1)
        if not np.array_equal(selected, expected_selected):
            raise RuntimeError(f"{split}/{arm} selected slots do not replay utility+threshold")
        expected_rank = baseline_rank.copy()
        active = selected >= 0
        expected_rank[active] = proposal_rank[rows[active], selected[active]]
        if not np.array_equal(stored_rank, expected_rank):
            raise RuntimeError(f"{split}/{arm} stored ranks do not replay selected proposals")
        counts[arm] = int(active.sum())

    if split == "inner":
        if not np.array_equal(body["selected_candidate_slot"], body["correct_selected_candidate_slot"]):
            raise RuntimeError("legacy inner selected-slot alias drifted from the correct arm")
        selected = body["correct_selected_candidate_slot"].astype(np.int64)
        active = selected >= 0
        if not np.allclose(
            body["best_predicted_utility"][active],
            body["correct_candidate_utility"][np.arange(len(query))[active], selected[active]],
            rtol=0, atol=0,
        ):
            raise RuntimeError("inner best-utility alias drifted from the correct arm")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_rule_residual_policy_v3",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    args = parser.parse_args()
    report = json.loads((args.policy_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "CHEMAWARE_ORTHOGONAL_RULE_RESIDUAL_POLICY_COMPLETE":
        raise RuntimeError("orthogonal teacher is incomplete")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError("orthogonal teacher did not pass every registered development gate")
    if report.get("gates", {}).get("outer_fold_untouched") is not True:
        raise RuntimeError("outer-fold seal is absent")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    validation = validate_split(
        args.policy_dir / "validation_policy.npz", report, "validation", 2, manifest,
    )
    inner = validate_split(
        args.policy_dir / "inner_policy.npz", report, "inner", 3, manifest,
    )
    with np.load(args.policy_dir / "validation_policy.npz", allow_pickle=False) as left, np.load(
        args.policy_dir / "inner_policy.npz", allow_pickle=False,
    ) as right:
        if set(left["formula"].astype(str)) & set(right["formula"].astype(str)):
            raise RuntimeError("validation and inner transfer caches leak formulas")
    print(json.dumps({
        "status": "CHEMAWARE_ORTHOGONAL_TRANSFER_CACHE_PASS",
        "policy_dir": str(args.policy_dir.resolve()),
        "validation_selected": validation,
        "inner_selected": inner,
        "outer_fold_untouched": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
