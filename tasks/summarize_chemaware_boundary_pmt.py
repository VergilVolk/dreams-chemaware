"""Paired held-formula summary for the four ChemAware B-PMT arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import formula_bootstrap


ARMS = (
    "clean_duplicate",
    "matched_formula_deranged",
    "alpha025",
    "alpha050",
)


def paired_metrics(
    treatment: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    seed: int,
    draws: int,
) -> dict:
    formula = treatment["formula"].astype(str)
    left = control["new_rank"].astype(np.int64)
    right = treatment["new_rank"].astype(np.int64)
    count = treatment["candidate_count"].astype(np.int64)
    output = {}
    for offset, k in enumerate((1, 5, 10, 20, 50)):
        delta = (right <= k).astype(float) - (left <= k).astype(float)
        output[f"recall{k}"] = {
            "mean": float(np.mean(delta)),
            **formula_bootstrap(delta, formula, seed + offset, draws),
        }
    reciprocal = 1.0 / right - 1.0 / left
    output["mrr"] = {
        "mean": float(np.mean(reciprocal)),
        **formula_bootstrap(reciprocal, formula, seed + 10, draws),
    }
    denominator = np.maximum(count - 1, 1)
    auc_delta = (left - right) / denominator
    output["macro_auc"] = {
        "mean": float(np.mean(auc_delta)),
        **formula_bootstrap(auc_delta, formula, seed + 11, draws),
    }
    output["micro_auc"] = {
        "mean": float(np.sum(left - right) / np.sum(denominator)),
    }
    output["corrected_at_1"] = int(np.sum((left != 1) & (right == 1)))
    output["introduced_at_1"] = int(np.sum((left == 1) & (right != 1)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output.exists() or args.bootstrap_draws < 10_000:
        raise ValueError("invalid or existing B-PMT summary output")
    reports, arrays = {}, {}
    for arm in ARMS:
        root = args.arms_root / arm
        report_path = root / "report.json"
        array_path = root / "inner_per_query.npz"
        checkpoint_path = root / "shared_spectrum_adapter.pt"
        if not all(path.is_file() for path in (report_path, array_path, checkpoint_path)):
            raise FileNotFoundError(f"incomplete B-PMT arm: {arm}")
        reports[arm] = json.loads(report_path.read_text(encoding="utf-8"))
        with np.load(array_path) as source:
            arrays[arm] = {key: np.array(source[key], copy=True) for key in source.files}
        if reports[arm].get("bpmt_arm") != arm:
            raise RuntimeError(f"B-PMT arm label drifted: {arm}")
    invariant_config = (
        "manifest", "token_dir", "folds", "fold_seed", "inner_fold", "outer_fold",
        "seed", "epochs", "max_steps", "warmup_steps", "eval_every_steps",
        "batch_queries", "references_per_molecule", "hidden_dim", "dropout",
        "learning_rate", "weight_decay", "temperature", "lambda_spectrum",
        "lambda_inbatch_spectrum", "lambda_margin_floor", "margin_floor_slack",
        "lambda_preserve", "grad_clip", "max_train_identities",
        "error_identity_fraction", "clean_safety_selection", "training_mass",
        "max_eval_identities", "lambda_bpmt", "bpmt_batch_actions",
        "bpmt_temperature",
    )
    anchor = reports["clean_duplicate"]["optimization"]
    for arm in ARMS:
        drift = [
            key for key in invariant_config
            if reports[arm]["optimization"].get(key) != anchor.get(key)
        ]
        if drift:
            raise RuntimeError(f"unmatched B-PMT arm schedule {arm}: {drift}")
        for key in ("query", "formula", "old_rank", "candidate_count"):
            if not np.array_equal(arrays[arm][key], arrays["clean_duplicate"][key]):
                raise RuntimeError(f"unpaired B-PMT evaluation {arm}/{key}")
    comparisons = {}
    for treatment in ("alpha025", "alpha050"):
        for control in ("clean_duplicate", "matched_formula_deranged"):
            name = f"{treatment}_minus_{control}"
            comparisons[name] = paired_metrics(
                arrays[treatment], arrays[control],
                args.seed + 100 * len(comparisons), args.bootstrap_draws,
            )
    eligible = []
    gates = {}
    for treatment in ("alpha025", "alpha050"):
        per_arm = {}
        for control in ("clean_duplicate", "matched_formula_deranged"):
            current = comparisons[f"{treatment}_minus_{control}"]
            per_arm[f"recall1_ci_positive_vs_{control}"] = (
                current["recall1"]["formula_cluster_bootstrap_95ci"][0] > 0
            )
            per_arm[f"corrected_gt_introduced_vs_{control}"] = (
                current["corrected_at_1"] > current["introduced_at_1"]
            )
            for metric in ("recall5", "recall10", "recall20", "recall50", "mrr", "macro_auc", "micro_auc"):
                per_arm[f"{metric}_nonnegative_vs_{control}"] = current[metric]["mean"] >= 0
        per_arm["preservation"] = reports[treatment]["preservation"]["mean"] >= 0.995
        gates[treatment] = per_arm
        if all(per_arm.values()):
            eligible.append(treatment)
    selected = (
        max(
            eligible,
            key=lambda arm: min(
                comparisons[f"{arm}_minus_{control}"]["recall1"]["mean"]
                for control in ("clean_duplicate", "matched_formula_deranged")
            ),
        )
        if eligible else None
    )
    output = {
        "status": (
            "CHEMAWARE_BOUNDARY_PMT_PHASE_A_PASS"
            if selected else "CHEMAWARE_BOUNDARY_PMT_PHASE_A_FAIL"
        ),
        "formal": False,
        "selected_for_second_seed": selected,
        "arms": {arm: reports[arm]["final_inner"] for arm in ARMS},
        "comparisons": comparisons,
        "gates": gates,
        "claim_limit": "One development formula fold and one seed; reserve fold 4 remains untouched.",
        "provenance": {
            arm: {
                "report_sha256": sha256_file(args.arms_root / arm / "report.json"),
                "per_query_sha256": sha256_file(args.arms_root / arm / "inner_per_query.npz"),
                "checkpoint_sha256": sha256_file(args.arms_root / arm / "shared_spectrum_adapter.pt"),
            }
            for arm in ARMS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

