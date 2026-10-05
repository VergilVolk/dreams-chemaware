"""Summarize preregistered direct-boundary-v2 seeds with a strict >=4 pp gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file


STABILITY_METRICS = (
    "recall2", "recall3", "recall5", "recall10", "recall20", "mrr",
    "macro_query_auc", "macro_query_auprc", "micro_candidate_auc",
    "micro_candidate_auprc", "mean_reciprocal_candidate_percentile",
    "mean_positive_vs_best_negative_margin", "selective_one_minus_aurc",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def corrected_formula_ci(
    old: np.ndarray, new: np.ndarray, formulas: np.ndarray, *, resamples: int,
    seed: int, family_size: int,
) -> dict[str, float]:
    unique, inverse = np.unique(formulas.astype(str), return_inverse=True)
    effect = (new == 1).astype(float) - (old == 1).astype(float)
    sums = np.bincount(inverse, weights=effect)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    tail = 0.05 / (2.0 * family_size)
    return {
        "mean": float(np.mean(effect)),
        "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1.0 - tail)),
        "familywise_alpha": 0.05,
        "bonferroni_family_size": family_size,
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.bootstrap_resamples < 1000 or len(set(args.expected_seeds)) != len(args.expected_seeds):
        raise ValueError("summary requires unique seeds and >=1000 resamples")
    matches = list(args.runs_root.rglob("decision.json"))
    by_seed: dict[int, Path] = {}
    for path in matches:
        decision = json.loads(path.read_text(encoding="utf-8"))
        seed = int(decision.get("configuration", {}).get("seed", -1))
        if seed in args.expected_seeds:
            if seed in by_seed:
                raise RuntimeError(f"multiple direct-v2 decisions for seed {seed}")
            by_seed[seed] = path.parent
    if set(by_seed) != set(args.expected_seeds):
        raise RuntimeError(f"missing direct-v2 seeds: {sorted(set(args.expected_seeds) - set(by_seed))}")

    seed_reports: dict[str, dict] = {}
    for offset, seed in enumerate(args.expected_seeds):
        run = by_seed[seed]
        decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        config = decision.get("configuration", {})
        contracts = decision.get("contracts", {})
        required_contracts = {
            "candidate_boundary_uses_molecule_max": True,
            "candidate_boundary_clean_primary_corrective_dose": True,
            "candidate_boundary_query_normalized_multi_action_dose": True,
            "candidate_boundary_query_shared_candidate_references": True,
            "candidate_boundary_noncorrective_action_safety_only": True,
            "pmt_all_routed_actions_exposed_before_recycling": True,
            "query_equal_action_weighting": True,
            "candidate_boundary_scalar_teacher_used": False,
        }
        if (
            decision.get("formal") is not True
            or not config.get("candidate_boundary_loss")
            or config.get("candidate_boundary_version") != "v2_molecule_max"
            or config.get("pmt_arm") != "paired_target"
            or any(contracts.get(key) != value for key, value in required_contracts.items())
        ):
            raise RuntimeError(f"seed {seed} is not a formal direct-boundary-v2 run")
        ledger = pd.read_csv(run / "held_per_query.csv.gz").sort_values(
            "query_index", kind="stable",
        )
        required_columns = {"query_formula", "baseline_rank", "initialization_rank", "final_rank"}
        if missing := required_columns - set(ledger.columns):
            raise RuntimeError(f"seed {seed} held ledger lacks {sorted(missing)}")
        total_ci = corrected_formula_ci(
            ledger["baseline_rank"].to_numpy(int), ledger["final_rank"].to_numpy(int),
            ledger["query_formula"].astype(str).to_numpy(),
            resamples=args.bootstrap_resamples, seed=args.seed + 10 * offset,
            family_size=len(args.expected_seeds),
        )
        incremental_ci = corrected_formula_ci(
            ledger["initialization_rank"].to_numpy(int), ledger["final_rank"].to_numpy(int),
            ledger["query_formula"].astype(str).to_numpy(),
            resamples=args.bootstrap_resamples, seed=args.seed + 10 * offset + 1,
            family_size=len(args.expected_seeds),
        )
        held = decision["held_clean"]
        panel = held["complete_candidate_metrics"]["student_minus_official"]
        missing_metrics = set(STABILITY_METRICS) - set(panel)
        if missing_metrics:
            raise RuntimeError(f"seed {seed} metric panel lacks {sorted(missing_metrics)}")
        gates = {
            "total_delta_ge4pp": total_ci["mean"] >= 0.04,
            "bonferroni_formula_ci_positive": total_ci["ci_low"] > 0,
            "incremental_over_mature_e4_positive": incremental_ci["mean"] > 0,
            "incremental_formula_ci_nonnegative": incremental_ci["ci_low"] >= 0,
            "corrected_exceeds_introduced": int(held["corrected"]) > int(held["introduced"]),
            "near_recall1_nonnegative": float(held["delta_near_recall1"]) >= 0,
            "complete_candidate_panel_nonnegative": all(
                float(panel[key]) >= 0 for key in STABILITY_METRICS
            ),
        }
        seed_reports[str(seed)] = {
            "run": str(run),
            "total_formula_cluster_ci": total_ci,
            "incremental_formula_cluster_ci": incremental_ci,
            "held_clean": held,
            "gates": gates,
            "passed": bool(all(gates.values())),
            "decision_sha256": sha256_file(run / "decision.json"),
            "checkpoint_sha256": sha256_file(run / "final_shared_encoder.pt"),
            "held_ledger_sha256": sha256_file(run / "held_per_query.csv.gz"),
        }
    passed = all(report["passed"] for report in seed_reports.values())
    body = {
        "status": "noise_direct_boundary_v2_summary_complete",
        "formal": True,
        "seeds": seed_reports,
        "pass_ge4pp_multiseed": passed,
        "decision": (
            "promote direct-boundary v2 to multifold confirmation"
            if passed else "do not claim >=4 pp and do not expand this configuration"
        ),
        "contracts": {
            "query_weighted_primary_metric": True,
            "formula_cluster_bootstrap": True,
            "bonferroni_corrected_across_seeds": True,
            "complete_candidate_metric_panel_checked": True,
            "mature_e4_increment_checked": True,
            "P_actions_used": False,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "claim_limit": "Fold 0 multi-seed gate; multifold confirmation is still required after passing.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_direct_v2_summary_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
