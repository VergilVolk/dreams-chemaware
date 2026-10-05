"""Causal formula-cluster summary for candidate-residual ChemAware arms."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]
from train_chemaware_full_candidate_alignment import formula_bootstrap  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402

ARMS = ("none", "mass", "rule", "rule_shifted", "rule_permuted")
EXPECTED = {
    "none": ("none", 0.0), "mass": ("mass", 0.10),
    "rule": ("rule_response", 0.20),
    "rule_shifted": ("rule_response_shifted", 0.20),
    "rule_permuted": ("rule_response_row_permuted", 0.20),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = {}; arrays = {}
    for arm in ARMS:
        report_path = args.root / arm / "report.json"
        ranks_path = args.root / arm / "inner_per_query.npz"
        if not report_path.is_file() or not ranks_path.is_file():
            raise FileNotFoundError(f"incomplete arm: {arm}")
        reports[arm] = json.loads(report_path.read_text(encoding="utf-8"))
        if reports[arm].get("status") != "ARM_COMPLETE":
            raise RuntimeError(f"arm not complete: {arm}")
        optimization = reports[arm]["optimization"]
        teacher, beta = EXPECTED[arm]
        if (optimization.get("teacher_arm") != teacher
                or not np.isclose(float(optimization.get("teacher_beta", -1)), beta)
                or optimization.get("teacher_objective") != "candidate_residual_huber"
                or optimization.get("teacher_scope") != "all"):
            raise RuntimeError(f"arm contract mismatch: {arm}")
        with np.load(ranks_path) as loaded:
            arrays[arm] = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    paired = (
        "inner_fold", "outer_fold", "seed", "epochs", "batch_queries",
        "references_per_molecule", "max_train_identities", "error_identity_fraction",
        "clean_safety_selection", "training_mass", "unfreeze_blocks", "backbone_lr",
        "head_lr", "temperature", "lambda_spectrum", "lambda_inbatch_spectrum",
        "lambda_teacher", "lambda_margin_floor", "lambda_preserve", "grad_clip",
        "candidate_residual_alpha", "candidate_residual_cap", "candidate_residual_huber",
        "teacher_gradient_ratio", "teacher_gradient_scale_cap",
    )
    anchor = reports["rule"]["optimization"]
    for arm in ARMS:
        different = [key for key in paired if reports[arm]["optimization"].get(key) != anchor.get(key)]
        # A no-teacher arm necessarily has zero gradient-ratio; everything else is paired.
        if arm == "none":
            different = [key for key in different if key != "teacher_gradient_ratio"]
        if different:
            raise RuntimeError(f"unmatched arm schedule: {arm}/{different}")
        for key in ("query", "formula", "old_rank"):
            if not np.array_equal(arrays["rule"][key], arrays[arm][key]):
                raise RuntimeError(f"unpaired evaluation: {arm}/{key}")
    formulas = arrays["rule"]["formula"]
    rule_hit = arrays["rule"]["new_rank"] == 1
    comparisons = {}
    for index, control in enumerate(("none", "mass", "rule_shifted", "rule_permuted")):
        control_hit = arrays[control]["new_rank"] == 1
        delta = rule_hit.astype(float) - control_hit.astype(float)
        comparisons[f"rule_minus_{control}"] = {
            "mean": float(np.mean(delta)),
            **formula_bootstrap(delta, formulas, args.seed + 100 + index, args.bootstrap_draws),
        }
    gates = {
        f"beats_{control}_point": comparisons[f"rule_minus_{control}"]["mean"] > 0
        for control in ("none", "mass", "rule_shifted", "rule_permuted")
    }
    gates.update({
        f"beats_{control}_formula_ci": comparisons[f"rule_minus_{control}"]["formula_cluster_bootstrap_95ci"][0] > 0
        for control in ("none", "mass", "rule_shifted", "rule_permuted")
    })
    gates.update({
        "all_arms_preserved": all(reports[arm]["preservation"] >= 0.995 for arm in ARMS),
        "all_arms_selected_trained_step": all(reports[arm]["selected_step"] > 0 for arm in ARMS),
        "rule_corrected_gt_introduced": (
            reports["rule"]["final_inner"]["corrected"]
            > reports["rule"]["final_inner"]["introduced"]
        ),
    })
    passed = all(gates.values())
    output = {
        "status": "CANDIDATE_RESIDUAL_CAUSAL_DEVELOPMENT_PASS" if passed else "CANDIDATE_RESIDUAL_CAUSAL_DEVELOPMENT_FAIL",
        "selected_for_outer_confirmation": "rule" if passed else None,
        "release_eligible": False,
        "comparisons": comparisons, "gates": gates,
        "contracts": {
            "candidate_residual_not_scalar_margin": True,
            "molecule_balanced_reference_supervision": True,
            "mass_control": True, "mass_shift_control": True,
            "rule_identity_permutation_control": True,
            "same_evaluation_queries": True, "outer_evaluated": False,
        },
        "claim_limit": "Development causal attribution only; untouched outer confirmation is still required.",
        "provenance": {
            "arms": {
                arm: {
                    "report_sha256": sha256_file(args.root / arm / "report.json"),
                    "inner_ranks_sha256": sha256_file(args.root / arm / "inner_per_query.npz"),
                    "checkpoint_sha256": sha256_file(args.root / arm / "final_shared_encoder.pt"),
                }
                for arm in ARMS
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
