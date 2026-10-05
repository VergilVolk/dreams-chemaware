"""Paired held-formula summary for the four-arm minimal E4-PMT experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from noise_final_core import sha256_file
from summarize_noise_final_e4a_causal_attribution import paired_summary


EXPECTED = {
    "clean_duplicate": ("clean_duplicate", 0.0),
    "matched_random": ("matched_random", 0.0),
    "alpha025": ("paired_target", 0.25),
    "alpha050": ("paired_target", 0.50),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite PMT summary: {args.output_dir}")
    decisions: dict[str, dict] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for label, (arm, alpha) in EXPECTED.items():
        root = args.arms_root / label
        matches = list(root.rglob("decision.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one completed {label} arm; found {len(matches)}")
        run = matches[0].parent
        decision = json.loads(matches[0].read_text(encoding="utf-8"))
        config = decision.get("configuration", {})
        contracts = decision.get("contracts", {})
        if (
            decision.get("formal") is not True
            or config.get("pmt_arm") != arm
            or not np.isclose(float(config.get("pmt_alpha", -1)), alpha)
            or contracts.get("pmt_harmful_target_weight_exact_zero") is not True
            or contracts.get("pmt_all_corrective_actions_exposed_before_recycling") is not True
            or contracts.get("pmt_M2_predictions_used") is not False
            or contracts.get("pmt_P_actions_used") is not False
        ):
            raise RuntimeError(f"invalid PMT arm contract: {label}")
        ledger = pd.read_csv(run / "held_per_query.csv.gz").sort_values(
            "query_index", kind="stable",
        ).reset_index(drop=True)
        decisions[label] = decision; ledgers[label] = ledger; paths[label] = run
    invariant = [
        "query_index", "query_formula", "has_near", "baseline_rank",
        "initialization_rank", "baseline_top_molecule_local",
        "initialization_top_molecule_local", "baseline_full_margin",
        "initialization_full_margin",
    ]
    reference = ledgers["clean_duplicate"]
    for label, ledger in ledgers.items():
        if label != "clean_duplicate":
            for column in invariant:
                left, right = reference[column], ledger[column]
                if left.dtype.kind == "f":
                    if not np.allclose(left, right, rtol=0, atol=2e-6):
                        raise RuntimeError(f"PMT invariant {column} drifted in {label}")
                elif not left.equals(right):
                    raise RuntimeError(f"PMT invariant {column} drifted in {label}")
    histories = {label: decision.get("history", []) for label, decision in decisions.items()}
    invariant_config = (
        "graph", "data", "embedding_cache", "official_checkpoint",
        "architecture_checkpoint", "initial_student_checkpoint", "pmt_manifest_dir",
        "policy", "action_selection", "action_scope", "outer_fold",
        "formula_fold_seed", "seed", "epochs", "batch_actions",
        "views_per_identity", "positive_spectra", "negative_molecules",
        "unfreeze_blocks", "head_lr", "backbone_lr", "weight_decay",
        "rank_margin", "temperature", "lambda_clean_rank", "lambda_aug_rank",
        "lambda_consistency", "lambda_margin_floor", "lambda_preserve",
        "margin_floor_slack", "safety_ratio", "safety_stream_weight",
        "grad_clip", "amp",
    )
    clean_config = decisions["clean_duplicate"]["configuration"]
    for label, decision in decisions.items():
        drift = {
            key: (clean_config.get(key), decision["configuration"].get(key))
            for key in invariant_config
            if clean_config.get(key) != decision["configuration"].get(key)
        }
        if drift:
            raise RuntimeError(f"non-PMT configuration drift in {label}: {drift}")
    if any(len(history) != 4 for history in histories.values()):
        raise RuntimeError("PMT arms do not all have four epochs")
    for epoch in range(4):
        keys = [
            history[epoch].get("action_sampling_schedule_sha256")
            for history in histories.values()
        ]
        if len(set(keys)) != 1:
            raise RuntimeError(f"PMT action sampling schedule drifted at epoch {epoch + 1}")
    comparisons: dict[str, dict] = {}
    for label in ("alpha025", "alpha050"):
        comparisons[f"{label}_vs_matched_random"] = paired_summary(
            ledgers["matched_random"], ledgers[label],
            args.bootstrap_resamples, args.seed + len(comparisons) * 10,
        )
        comparisons[f"{label}_vs_clean_duplicate"] = paired_summary(
            ledgers["clean_duplicate"], ledgers[label],
            args.bootstrap_resamples, args.seed + len(comparisons) * 10,
        )
    eligible = []
    for label in ("alpha025", "alpha050"):
        random = comparisons[f"{label}_vs_matched_random"]
        clean = comparisons[f"{label}_vs_clean_duplicate"]
        if (
            random["formula_cluster_top1_ci"]["ci_low"] > 0
            and clean["formula_cluster_top1_ci"]["ci_low"] > 0
            and random["corrected"] > random["introduced"]
            and random["risk_net_lambda2"] > 0
            and random["delta_near_recall1"] >= 0
            and random["delta_mrr"] >= 0
        ):
            eligible.append(label)
    report = {
        "status": "noise_final_e4_pmt_summary_complete",
        "formal": True,
        "arms": {
            label: {
                "run": str(paths[label]),
                "decision_sha256": sha256_file(paths[label] / "decision.json"),
                "checkpoint_sha256": sha256_file(paths[label] / "final_shared_encoder.pt"),
                "held": decisions[label]["held_clean"],
            } for label in EXPECTED
        },
        "comparisons": comparisons,
        "eligible_arms": eligible,
        "pass_to_second_seed": bool(eligible),
        "decision": (
            "retain the strongest preregistered alpha for a second seed"
            if eligible else
            "paired margin transfer did not add a strict clean-embedding increment over both controls"
        ),
        "contracts": {
            "mature_E4_initialization": True,
            "identical_corrective_membership": True,
            "identical_action_schedule": True,
            "target_control_same_batch_in_treatment": True,
            "harmful_target_weight_exact_zero": True,
            "full_held_formula_evaluation": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "claim_limit": "One development formula fold and one seed; not multifold or sealed P3 evidence.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e4_pmt_summary_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        combined = reference[["query_index", "query_formula", "has_near", "baseline_rank"]].copy()
        for label in EXPECTED:
            combined[f"{label}_rank"] = ledgers[label]["final_rank"].to_numpy(np.int16)
            combined[f"{label}_margin"] = ledgers[label]["final_full_margin"].to_numpy(np.float32)
        combined.to_csv(staging / "paired_per_query.csv.gz", index=False, compression="gzip")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
