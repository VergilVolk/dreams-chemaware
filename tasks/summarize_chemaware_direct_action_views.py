"""Causal summary for the four-arm no-distillation ChemAware pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemaware_direct_action_core import formula_bootstrap


ARMS = ("clean_duplicate", "correct_synthetic", "candidate_swapped", "peak_permuted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260935)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--stage", choices=("primary", "full"), default="full")
    return parser.parse_args()


def load_arm(root: Path, arm: str) -> tuple[dict, pd.DataFrame]:
    complete_path = root / arm / "COMPLETE.json"
    report_path = root / arm / "report.json"
    rows_path = root / arm / "inner_per_query.csv.gz"
    checkpoint_path = root / arm / "final_shared_encoder.pt"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("status") != "CHEMAWARE_DIRECT_ARM_COMPLETE"
        or complete.get("arm") != arm
        or any(not path.is_file() for path in (report_path, rows_path, checkpoint_path))
    ):
        raise RuntimeError(f"arm {arm} is not atomically complete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = pd.read_csv(rows_path).sort_values("query_index").reset_index(drop=True)
    required_columns = {
        "query_index", "formula", "identity", "initial_rank", "final_rank",
        "initial_margin", "final_margin",
    }
    if rows.empty or not required_columns.issubset(rows.columns):
        raise RuntimeError(f"arm {arm} per-query ledger is empty or incomplete")
    if rows["query_index"].duplicated().any():
        raise RuntimeError(f"arm {arm} per-query ledger contains duplicate queries")
    numeric = rows[["initial_rank", "final_rank", "initial_margin", "final_margin"]].to_numpy()
    if not np.all(np.isfinite(numeric)) or (rows[["initial_rank", "final_rank"]] < 1).any().any():
        raise RuntimeError(f"arm {arm} per-query ledger contains invalid ranks or margins")
    if report.get("preflight", {}).get("arm") != arm:
        raise RuntimeError(f"arm label mismatch: {arm}")
    contract = report.get("preflight", {}).get("contracts", {})
    forbidden = (
        contract.get("teacher_score_loss"),
        contract.get("teacher_embedding_loss"),
        contract.get("clean_action_embedding_consistency"),
    )
    if any(value is not False for value in forbidden):
        raise RuntimeError(f"arm {arm} is not a no-distillation run")
    objective = report.get("optimization", {}).get("training_objective")
    guarded = objective in {
        "direct_guarded_listwise", "direct_pcgrad_guarded", "direct_projected_guarded",
    }
    clean_safety = (
        contract.get("official_embedding_preservation_loss"),
        contract.get("official_margin_floor_loss"),
    )
    if guarded and any(value is not True for value in clean_safety):
        raise RuntimeError(f"guarded arm {arm} is missing clean safety regularizers")
    if not guarded and any(value is not False for value in clean_safety):
        raise RuntimeError(f"legacy direct arm {arm} has an unexpected clean safety regularizer")
    if objective in {"direct_pcgrad_guarded", "direct_projected_guarded"}:
        ratio = report.get("optimization", {}).get("maximum_action_gradient_ratio")
        if (
            contract.get("action_gradient_nonconflicting_with_primary") is not True
            or contract.get("action_gradient_norm_cap") != ratio
            or ratio is None or not 0 < float(ratio) <= 1
        ):
            raise RuntimeError(f"PCGrad arm {arm} is missing its gradient-safety contract")
    if report.get("scope", {}).get("outer_fold_evaluated") is not False:
        raise RuntimeError(f"arm {arm} evaluated the outer fold")
    report_gates = report.get("gates")
    if not isinstance(report_gates, dict):
        raise RuntimeError(f"arm {arm} lacks evaluation gates")
    applicable = [value for value in report_gates.values() if value is not None]
    if not applicable or any(not isinstance(value, bool) for value in applicable):
        raise RuntimeError(f"arm {arm} contains invalid evaluation gates")
    expected_status = "PASS" if all(applicable) else "FAIL"
    if report.get("status") != expected_status or complete.get("report_status") != expected_status:
        raise RuntimeError(f"arm {arm} status is inconsistent with its evaluation gates")
    row_delta = float(np.mean(
        (rows["final_rank"].to_numpy() == 1).astype(float)
        - (rows["initial_rank"].to_numpy() == 1).astype(float)
    ))
    reported_delta = report.get("final", {}).get("inner_all", {}).get("delta_recall1")
    if reported_delta is None or not np.isclose(row_delta, float(reported_delta), atol=1e-12):
        raise RuntimeError(f"arm {arm} report and per-query recall delta disagree")
    return report, rows


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_draws < 10_000:
        raise ValueError("causal summary requires at least 10,000 formula-cluster draws")
    required_arms = ARMS[:2] if args.stage == "primary" else ARMS
    loaded = {arm: load_arm(args.root, arm) for arm in required_arms}
    reference = loaded[required_arms[0]][1][["query_index", "formula", "identity"]]
    for arm in required_arms[1:]:
        candidate = loaded[arm][1][["query_index", "formula", "identity"]]
        if not reference.equals(candidate):
            raise RuntimeError(f"stage query rows are not exactly aligned: {arm}")
    formula = reference["formula"].astype(str).to_numpy()
    delta = {}
    arm_summary = {}
    optimization_fingerprints = {}
    for arm, (report, rows) in loaded.items():
        value = (
            (rows["final_rank"].to_numpy() == 1).astype(np.float64)
            - (rows["initial_rank"].to_numpy() == 1).astype(np.float64)
        )
        delta[arm] = value
        arm_summary[arm] = {
            "status": report.get("status"),
            "inner_all_delta_recall1": float(np.mean(value)),
            "formula_bootstrap": formula_bootstrap(
                value, formula, args.seed + ARMS.index(arm), args.bootstrap_draws,
            ),
            "report_sha256": sha256_file(args.root / arm / "report.json"),
        }
        optimization = report.get("optimization", {})
        optimization_fingerprints[arm] = {
            key: optimization.get(key)
            for key in (
                "epochs", "batch_queries", "max_action_identities", "max_safety_identities", "backbone_lr",
                "head_lr", "temperature", "training_objective",
                "lambda_margin_floor", "lambda_preserve", "grad_clip",
                "maximum_action_gradient_ratio",
            )
        }
    if len({json.dumps(value, sort_keys=True) for value in optimization_fingerprints.values()}) != 1:
        raise RuntimeError("stage arms do not have a matched optimization schedule")
    contrasts = {}
    controls = (
        ("clean_duplicate",)
        if args.stage == "primary"
        else ("clean_duplicate", "candidate_swapped", "peak_permuted")
    )
    for index, control in enumerate(controls):
        contrasts[f"correct_minus_{control}"] = formula_bootstrap(
            delta["correct_synthetic"] - delta[control],
            formula, args.seed + 101 + index, args.bootstrap_draws,
        )
    gates = {
        "correct_absolute_formula_ci_positive": (
            arm_summary["correct_synthetic"]["formula_bootstrap"]
            ["formula_cluster_bootstrap_95ci"][0] > 0
        ),
        **{
            f"{name}_formula_ci_positive": value["formula_cluster_bootstrap_95ci"][0] > 0
            for name, value in contrasts.items()
        },
        "correct_global_inner_nonnegative": (
            loaded["correct_synthetic"][0]["final"]["inner_all"]["delta_recall1"] >= 0
        ),
        "correct_run_passed_safety_gates": loaded["correct_synthetic"][0].get("status") == "PASS",
    }
    passed = bool(all(gates.values()))
    if args.stage == "primary":
        status = "CHEMAWARE_DIRECT_ACTION_STAGE1_PASS" if passed else "CHEMAWARE_DIRECT_ACTION_STAGE1_FAIL"
        decision = (
            "The correct action beat equal-budget clean continuation; run the two pseudo-action controls."
            if passed else
            "Stop before pseudo-action controls: the correct action did not beat clean continuation safely."
        )
    else:
        status = "CHEMAWARE_DIRECT_ACTION_CAUSAL_PASS" if passed else "CHEMAWARE_DIRECT_ACTION_CAUSAL_FAIL"
        decision = (
            "The chemical action produced a strict-positive gain over continuation and both matched null actions."
            if passed else
            "The chemical action did not establish a strict-positive causal embedding gain over all matched controls."
        )
    report = {
        "status": status,
        "stage": args.stage,
        "pass_to_matched_controls": passed if args.stage == "primary" else False,
        "pass_to_additional_seed": passed if args.stage == "full" else False,
        "release_eligible": False,
        "decision": decision,
        "queries": int(len(reference)),
        "formulas": int(len(np.unique(formula))),
        "arms": arm_summary,
        "contrasts": contrasts,
        "gates": gates,
        "optimization_fingerprints": optimization_fingerprints,
        "scope": {
            "formula_clustered": True,
            "outer_fold_evaluated": False,
            "no_teacher_score_or_embedding_loss": True,
            "single_seed_pilot": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
