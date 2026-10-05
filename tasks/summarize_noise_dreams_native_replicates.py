"""Promotion decision for two preregistered native Noise training seeds."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_noise_dreams_native import formula_cluster_ci


def directional_checks(candidate: dict, reference: dict, prefix: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    higher = (
        "recall@1", "recall@2", "recall@3", "recall@5", "recall@10",
        "recall@20", "mrr", "macro_query_auroc", "macro_query_auprc",
        "mean_positive_vs_best_negative_margin", "mean_signed_top1_top2_gap",
    )
    for panel in ("retrieval", "near_subset"):
        for name in higher:
            checks[f"{prefix}.{panel}.{name}"] = (
                float(candidate[panel][name]) >= float(reference[panel][name])
            )
        for name in ("mean_rank", "median_rank"):
            checks[f"{prefix}.{panel}.{name}"] = (
                float(candidate[panel][name]) <= float(reference[panel][name])
            )
    for panel in (
        "micro_candidate", "massspecgym_10ppm_pooled_pairwise",
        "massspecgym_mh_10ppm_pooled_pairwise",
    ):
        for name in ("auroc", "auprc"):
            checks[f"{prefix}.{panel}.{name}"] = (
                float(candidate[panel][name]) >= float(reference[panel][name])
            )
    return checks


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replicate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta-recall1-pp", type=float, default=2.0)
    parser.add_argument("--maximum-seed-recall1-gap-pp", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def load_arm(path: Path, arm: str) -> tuple[dict, pd.DataFrame]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(path / "held_per_query.csv.gz", low_memory=False)
    if (
        report.get("status") != "NOISE_DREAMS_NATIVE_HELD_EVALUATION_COMPLETE"
        or report.get("arm") != arm
        or len(table) != 18333
        or table["query_index"].duplicated().any()
    ):
        raise RuntimeError(f"native replicate is incomplete: {path}")
    if (
        report.get("mature_e8") is None
        or report.get("candidate_vs_mature_e8") is None
        or report.get("mature_e8_formula_cluster_paired_ci") is None
    ):
        raise RuntimeError(f"native replicate lacks mature-E8 comparison: {path}")
    return report, table


def arm_gates(report: dict, *, prefix: str, minimum_pp: float) -> tuple[dict, dict]:
    candidate = report["candidate"]
    official = report["official"]
    mature = report["mature_e8"]
    delta_official = 100.0 * (
        float(candidate["retrieval"]["recall@1"])
        - float(official["retrieval"]["recall@1"])
    )
    delta_mature = 100.0 * (
        float(candidate["retrieval"]["recall@1"])
        - float(mature["retrieval"]["recall@1"])
    )
    official_ci = report["formula_cluster_paired_ci"]
    mature_ci = report["mature_e8_formula_cluster_paired_ci"]
    official_risk = report["candidate_vs_official"]
    mature_risk = report["candidate_vs_mature_e8"]
    directional = {
        **directional_checks(candidate, official, f"{prefix}_vs_official"),
        **directional_checks(candidate, mature, f"{prefix}_vs_mature_e8"),
    }
    gates = {
        f"{prefix}_gain_vs_official_at_least_registered_pp": delta_official >= minimum_pp,
        f"{prefix}_gain_vs_mature_e8_at_least_registered_pp": delta_mature >= minimum_pp,
        f"{prefix}_recall1_formula_cis_strict_positive": all(
            float(panel["recall1"]["ci_low_pp"]) > 0
            for panel in (official_ci, mature_ci)
        ),
        f"{prefix}_risk_net_lambda2_positive": all(
            int(panel["risk_net_lambda2"]) > 0
            for panel in (official_risk, mature_risk)
        ),
    }
    return gates, {
        "delta_recall1_vs_official_pp": delta_official,
        "delta_recall1_vs_mature_e8_pp": delta_mature,
        "directional_metric_checks": directional,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    primary, primary_table = load_arm(args.primary, "primary")
    replicate, replicate_table = load_arm(args.replicate, "replicate")
    keys = ["query_index", "query_row", "query_ik14", "query_formula", "near", "official_rank"]
    if primary_table[keys].to_dict("list") != replicate_table[keys].to_dict("list"):
        raise RuntimeError("native seed evaluations are not paired")
    primary_gates, primary_summary = arm_gates(
        primary, prefix="primary", minimum_pp=args.minimum_delta_recall1_pp,
    )
    replicate_gates, replicate_summary = arm_gates(
        replicate, prefix="replicate", minimum_pp=args.minimum_delta_recall1_pp,
    )
    primary_rank = primary_table["candidate_rank"].to_numpy(np.int64)
    replicate_rank = replicate_table["candidate_rank"].to_numpy(np.int64)
    formulas = primary_table["query_formula"].astype(str).to_numpy()
    seed_r1_delta = (primary_rank == 1).astype(float) - (replicate_rank == 1).astype(float)
    seed_gap_pp = abs(float(100.0 * np.mean(seed_r1_delta)))
    seed_ci = formula_cluster_ci(
        formulas, seed_r1_delta, repeats=args.bootstrap_resamples,
        seed=args.seed, hypotheses=1,
    )
    stability_gates = {
        "both_preregistered_seeds_pass_independently": all(primary_gates.values()) and all(replicate_gates.values()),
        "seed_recall1_gap_at_most_1pp": seed_gap_pp <= args.maximum_seed_recall1_gap_pp,
    }
    gates = {**primary_gates, **replicate_gates, **stability_gates}
    report = {
        "status": "NOISE_DREAMS_NATIVE_PROMOTION_PASS" if all(gates.values()) else "NOISE_DREAMS_NATIVE_PROMOTION_FAIL",
        "design": "same frozen query-balanced dynamic Noise curriculum; seeds 3407 and 3408; no pseudo-causal training control",
        "primary": primary_summary,
        "replicate": replicate_summary,
        "seed_recall1_gap_pp": seed_gap_pp,
        "primary_vs_replicate_formula_cluster_ci": seed_ci,
        "minimum_delta_recall1_pp": args.minimum_delta_recall1_pp,
        "maximum_seed_recall1_gap_pp": args.maximum_seed_recall1_gap_pp,
        "gates": gates,
        "claim_limit": "Two-seed corrected MassSpecGym formula-held evidence; not a causal action-null or external-dataset claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
