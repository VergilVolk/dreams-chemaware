"""Summarize two source-and-artifact-exact historical E4 replays."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from train_noise_final_r2_shared_encoder import formula_bootstrap_delta


HISTORICAL_TRAINER_SHA256 = (
    "28c3b375d270fc2030783938d9390710c2c5ab8f0926a0b4ff507d62afa26885"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def load_replay(root: Path) -> tuple[dict, dict, pd.DataFrame]:
    decision_path = root / "training" / "decision.json"
    evaluation_path = root / "corrected_evaluation" / "report.json"
    table_path = root / "corrected_evaluation" / "paired_per_query.csv.gz"
    checkpoint_path = root / "training" / "final_shared_encoder.pt"
    for path in (decision_path, evaluation_path, table_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (
        decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
        or decision.get("provenance", {}).get("script_sha256")
        != HISTORICAL_TRAINER_SHA256
        or evaluation.get("status")
        != "noise_e4_faithful_v1_corrected_evaluation_complete"
        or evaluation.get("formal") is not False
        or evaluation.get("evaluation_role")
        != "registered train-side development graph"
        or evaluation.get("provenance", {}).get("student_checkpoint_sha256")
        != sha256_file(checkpoint_path)
        or evaluation.get("provenance", {}).get("training_decision_sha256")
        != sha256_file(decision_path)
    ):
        raise RuntimeError(f"incomplete or non-historical E4 replay: {root}")
    return decision, evaluation, pd.read_csv(table_path, low_memory=False)


def nested_delta(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in candidate.items():
        reference = baseline.get(key)
        if isinstance(value, dict) and isinstance(reference, dict):
            output[key] = nested_delta(value, reference)
        elif (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and isinstance(reference, (int, float)) and not isinstance(reference, bool)
        ):
            output[key] = float(value - reference)
    return output


def historical_summary(decision: dict) -> dict[str, object]:
    held = decision["held_clean"]
    recomputed_risk_net = int(held["corrected"] - 2 * held["introduced"])
    if int(held["risk_net"]) != recomputed_risk_net:
        raise RuntimeError("historical E4 decision risk-net is not lambda=2")
    return {
        "seed": int(decision["configuration"]["seed"]),
        "held_queries": int(held["n_queries"]),
        "recall1": float(held["recall1"]),
        "recall1_delta_pp_vs_official": float(100 * held["delta_recall1"]),
        "mrr_delta_pp_vs_official": float(100 * held["delta_mrr"]),
        "near_recall1_delta_pp_vs_official": float(100 * held["delta_near_recall1"]),
        "corrected": int(held["corrected"]),
        "introduced": int(held["introduced"]),
        "risk_net_lambda2": int(held["risk_net"]),
        "formula_cluster_delta_recall1": held["formula_cluster_delta_recall1"],
    }


def corrected_summary(evaluation: dict) -> dict[str, object]:
    return {
        "held_queries": int(evaluation["held_queries"]),
        "student": evaluation["student"],
        "student_minus_official": evaluation["student_minus_official"],
        "paired_top1_vs_official": evaluation["paired_top1_vs_official"],
    }


def replay_agreement(
    primary: pd.DataFrame, replica: pd.DataFrame, resamples: int, seed: int,
) -> dict[str, object]:
    keys = ["query_index", "query_row", "query_ik14", "query_formula", "near"]
    if primary[keys].to_dict("list") != replica[keys].to_dict("list"):
        raise RuntimeError("E4 replay evaluation ledgers are not exactly aligned")
    primary_rank = primary["candidate_rank"].to_numpy(np.int64)
    replica_rank = replica["candidate_rank"].to_numpy(np.int64)
    p_wins = (primary_rank == 1) & (replica_rank > 1)
    r_wins = (replica_rank == 1) & (primary_rank > 1)
    ci = formula_bootstrap_delta(
        replica_rank, primary_rank,
        primary["query_formula"].astype(str).to_numpy(), resamples, seed,
    )
    return {
        "top1_agreement_fraction": float(np.mean((primary_rank == 1) == (replica_rank == 1))),
        "exact_rank_agreement_fraction": float(np.mean(primary_rank == replica_rank)),
        "primary_only_top1": int(np.sum(p_wins)),
        "replica_only_top1": int(np.sum(r_wins)),
        "primary_minus_replica_recall1_pp": float(
            100 * (np.mean(primary_rank == 1) - np.mean(replica_rank == 1))
        ),
        "formula_cluster_primary_minus_replica": ci,
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    primary_decision, primary_eval, primary_table = load_replay(args.primary)
    replica_decision, replica_eval, replica_table = load_replay(args.replica)

    primary_config = primary_decision["configuration"]
    replica_config = replica_decision["configuration"]
    allowed_differences = {"seed", "output_root"}
    invariant_primary = {
        key: value for key, value in primary_config.items()
        if key not in allowed_differences
    }
    invariant_replica = {
        key: value for key, value in replica_config.items()
        if key not in allowed_differences
    }
    if invariant_primary != invariant_replica:
        raise RuntimeError("historical E4 replays differ outside seed and output root")
    if int(primary_config["seed"]) != 20260830 or int(replica_config["seed"]) != 20260829:
        raise RuntimeError("the preregistered primary/replica seed roles drifted")

    primary_delta_pp = float(
        100 * primary_eval["student_minus_official"]["retrieval"]["recall@1"]
    )
    replica_delta_pp = float(
        100 * replica_eval["student_minus_official"]["retrieval"]["recall@1"]
    )
    report = {
        "status": "noise_e4_faithful_v1_summary_complete",
        "formal": False,
        "evaluation_role": "registered train-side development graph",
        "method": "source-and-registered-artifact-exact historical E4 direct replay",
        "primary_role_frozen_before_replay": "seed_20260830",
        "historical_graph": {
            "primary": historical_summary(primary_decision),
            "replica": historical_summary(replica_decision),
        },
        "corrected_graph": {
            "primary": corrected_summary(primary_eval),
            "replica": corrected_summary(replica_eval),
            "primary_minus_replica_complete_metrics": nested_delta(
                primary_eval["student"], replica_eval["student"],
            ),
            "replay_agreement": replay_agreement(
                primary_table, replica_table,
                args.bootstrap_resamples, args.seed + 17,
            ),
        },
        "gates": {
            "historical_primary_gain_positive": bool(
                primary_decision["held_clean"]["delta_recall1"] > 0
            ),
            "historical_replica_gain_positive": bool(
                replica_decision["held_clean"]["delta_recall1"] > 0
            ),
            "corrected_primary_gain_positive": bool(primary_delta_pp > 0),
            "corrected_replica_gain_positive": bool(replica_delta_pp > 0),
            "corrected_primary_gain_at_least_4pp": bool(primary_delta_pp >= 4.0),
            "corrected_replica_gain_at_least_4pp": bool(replica_delta_pp >= 4.0),
            "primary_formula_ci_low_strict_positive": bool(
                100 * primary_eval["paired_top1_vs_official"]
                ["formula_cluster_delta_recall1"]["ci_low"] > 0
            ),
            "replica_formula_ci_low_strict_positive": bool(
                100 * replica_eval["paired_top1_vs_official"]
                ["formula_cluster_delta_recall1"]["ci_low"] > 0
            ),
        },
        "contracts": {
            "training_source_sha256": HISTORICAL_TRAINER_SHA256,
            "training_source_byte_exact": True,
            "historical_noise_v3_core_byte_exact": True,
            "official_initialization_only": True,
            "historical_r0_action_bank_only": True,
            "injector_used": False,
            "materialized_or_recomputed_action_used": False,
            "teacher_or_distillation_used": False,
            "corrected_graph_is_evaluation_only": True,
            "checkpoint_selected_from_held_metrics": False,
            "query_formula_held_out_only": True,
            "reference_spectrum_formula_isolation_enforced": False,
            "runtime_cuda_stack_byte_exact": False,
        },
        "provenance": {
            "primary_training_decision_sha256": sha256_file(
                args.primary / "training" / "decision.json"
            ),
            "replica_training_decision_sha256": sha256_file(
                args.replica / "training" / "decision.json"
            ),
            "primary_evaluation_sha256": sha256_file(
                args.primary / "corrected_evaluation" / "report.json"
            ),
            "replica_evaluation_sha256": sha256_file(
                args.replica / "corrected_evaluation" / "report.json"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Development-only E4 replay on a registered train-side graph, not a P3 test or a "
            "promised 4 pp result. Query formulas are held out, but historical E4 did not "
            "formula-isolate positive/negative reference spectra. The primary seed role is "
            "fixed before evaluation, the replica is not a selection pool, and the unfrozen "
            "CUDA/runtime stack prevents a bitwise numerical-reproduction claim."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".noise_e4_faithful_summary_", dir=args.output_dir.parent,
    ))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
