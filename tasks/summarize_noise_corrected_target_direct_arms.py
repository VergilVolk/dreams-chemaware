"""Paired causal summary for clean-control, fixed-P, and combined N+P arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file


ARMS = ("clean_control", "p_direct", "np_direct")
CONTRASTS = (
    ("p_direct_vs_clean_control", "p_direct", "clean_control"),
    ("np_direct_vs_p_direct", "np_direct", "p_direct"),
    ("np_direct_vs_clean_control", "np_direct", "clean_control"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def formula_bootstrap(
    formula: np.ndarray, value: np.ndarray, repeats: int, seed: int, family_size: int,
) -> dict[str, float | int | bool]:
    grouped = pd.DataFrame({"formula": formula, "value": value}).groupby("formula").value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), len(grouped))
        boot[index] = 100.0 * sums[take].sum() / counts[take].sum()
    tail = 0.05 / (2.0 * family_size)
    low, high = np.quantile(boot, [tail, 1.0 - tail])
    return {
        "queries": int(len(value)),
        "formulas": int(len(grouped)),
        "delta_recall1_pp": float(100.0 * np.mean(value)),
        "multiplicity_method": "Bonferroni simultaneous confidence interval",
        "family_size": family_size,
        "familywise_alpha": 0.05,
        "ci_low_pp": float(low),
        "ci_high_pp": float(high),
        "strict_positive": bool(low > 0),
    }


def paired_rank_summary(
    better: pd.DataFrame, baseline: pd.DataFrame, repeats: int, seed: int, family_size: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    keys = ["query_index", "query_row", "query_ik14", "query_formula", "near"]
    if better[keys].to_dict("list") != baseline[keys].to_dict("list"):
        raise RuntimeError("arm held-query ledgers are not exactly aligned")
    out = better[keys].copy()
    out["baseline_rank"] = baseline.candidate_rank.to_numpy(np.int64)
    out["better_rank"] = better.candidate_rank.to_numpy(np.int64)
    corrected = out.baseline_rank.gt(1) & out.better_rank.eq(1)
    introduced = out.baseline_rank.eq(1) & out.better_rank.gt(1)
    delta = (out.better_rank.eq(1).astype(float) - out.baseline_rank.eq(1).astype(float)).to_numpy()
    out["corrected"] = corrected
    out["introduced"] = introduced
    out["delta_recall1"] = delta
    near = out.near.to_numpy(bool)
    report = {
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "net_corrected_minus_introduced": int(corrected.sum() - introduced.sum()),
        "risk_net_lambda2": int(corrected.sum() - 2 * introduced.sum()),
        "formula_cluster_delta_recall1": formula_bootstrap(
            out.query_formula.to_numpy(str), delta, repeats, seed, family_size,
        ),
        "near_formula_cluster_delta_recall1": formula_bootstrap(
            out.loc[near, "query_formula"].to_numpy(str), delta[near], repeats, seed + 1000, family_size,
        ),
    }
    return report, out


def _numeric_deltas(candidate: object, baseline: object, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(candidate, dict) and isinstance(baseline, dict):
        for key in candidate.keys() & baseline.keys():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_numeric_deltas(candidate[key], baseline[key], name))
    elif (
        isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
        and isinstance(baseline, (int, float)) and not isinstance(baseline, bool)
    ):
        output[prefix] = float(candidate) - float(baseline)
    return output


def _no_regression_gate(delta: dict[str, float]) -> dict[str, object]:
    positive = (
        "retrieval.mrr", "retrieval.macro_query_auroc", "retrieval.macro_query_auprc",
        "retrieval.mean_positive_vs_best_negative_margin", "retrieval.mean_top1_top2_gap",
        "near_subset.mrr", "near_subset.macro_query_auroc", "near_subset.macro_query_auprc",
        "near_subset.mean_positive_vs_best_negative_margin",
        "micro_candidate.auroc", "micro_candidate.auprc",
        "massspecgym_10ppm_pooled_pairwise.auroc", "massspecgym_10ppm_pooled_pairwise.auprc",
        "massspecgym_mh_10ppm_pooled_pairwise.auroc", "massspecgym_mh_10ppm_pooled_pairwise.auprc",
    )
    nonnegative = tuple(
        f"{scope}.recall@{cutoff}" for scope in ("retrieval", "near_subset")
        for cutoff in (2, 3, 5, 10, 20)
    )
    nonpositive = ("retrieval.mean_rank", "retrieval.median_rank", "near_subset.mean_rank", "near_subset.median_rank")
    checks = ({key: delta[key] > 0 for key in positive}
              | {key: delta[key] >= -1e-12 for key in nonnegative}
              | {key: delta[key] <= 1e-12 for key in nonpositive})
    return {"passed": bool(all(checks.values())), "checks": checks}


def summarize(arms_root: Path, output_dir: Path, repeats: int, seed: int) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    decisions: dict[str, dict[str, object]] = {}
    tables: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        decision_path = arms_root / arm / "decision.json"
        table_path = arms_root / arm / "held_per_query.csv.gz"
        if not decision_path.is_file() or not table_path.is_file():
            raise FileNotFoundError(f"formal arm is incomplete: {arm}")
        decisions[arm] = json.loads(decision_path.read_text(encoding="utf-8"))
        tables[arm] = pd.read_csv(table_path, low_memory=False)
        if decisions[arm].get("formal") is not True:
            raise RuntimeError(f"arm is not formal: {arm}")
    initialization = [decisions[arm].get("initialization") for arm in ARMS]
    if initialization[1:] != initialization[:-1]:
        raise RuntimeError("arms do not use the same E4 initialization")
    initial_metrics = [decisions[arm]["evaluation"]["initial_e4"] for arm in ARMS]
    if initial_metrics[1:] != initial_metrics[:-1]:
        raise RuntimeError("arms do not reproduce the same initial E4 metrics")

    contrasts: dict[str, object] = {}
    paired_tables: dict[str, pd.DataFrame] = {}
    for index, (label, better, baseline) in enumerate(CONTRASTS):
        rank_report, table = paired_rank_summary(
            tables[better], tables[baseline], repeats, seed + index, len(CONTRASTS),
        )
        rank_report["candidate_metric_delta"] = _numeric_deltas(
            decisions[better]["evaluation"]["candidate"],
            decisions[baseline]["evaluation"]["candidate"],
        )
        contrasts[label] = rank_report
        paired_tables[label] = table

    initial = initial_metrics[0]
    arm_reports: dict[str, object] = {}
    for arm in ARMS:
        candidate = decisions[arm]["evaluation"]["candidate"]
        arm_reports[arm] = {
            "candidate": candidate,
            "candidate_minus_initial_e4": _numeric_deltas(candidate, initial),
            "paired_vs_initial_e4": decisions[arm]["evaluation"]["candidate_vs_initial_e4"],
        }
    np_delta = arm_reports["np_direct"]["candidate_minus_initial_e4"]
    causal = contrasts["np_direct_vs_clean_control"]
    metric_gate = _no_regression_gate(np_delta)
    pass_to_second_seed = bool(
        100.0 * np_delta["retrieval.recall@1"] >= 4.0
        and 100.0 * np_delta["near_subset.recall@1"] >= 4.0
        and causal["formula_cluster_delta_recall1"]["strict_positive"]
        and causal["near_formula_cluster_delta_recall1"]["strict_positive"]
        and causal["risk_net_lambda2"] > 0
        and metric_gate["passed"]
    )
    report: dict[str, object] = {
        "status": "noise_corrected_target_direct_three_arm_summary_complete",
        "formal": True,
        "arms": arm_reports,
        "contrasts": contrasts,
        "promotion_gate": {
            "pass_to_second_seed": pass_to_second_seed,
            "requires_np_vs_initial_e4_recall1_delta_pp_at_least": 4.0,
            "requires_np_vs_initial_e4_near_recall1_delta_pp_at_least": 4.0,
            "requires_np_vs_clean_multiplicity_corrected_formula_ci_strict_positive": True,
            "requires_np_vs_clean_near_formula_ci_strict_positive": True,
            "requires_risk_net_lambda2_positive": True,
            "all_directional_metric_checks": metric_gate,
            "is_final_multiseed_claim": False,
        },
        "contracts": {
            "same_e4_initialization": True,
            "same_held_query_ledger": True,
            "clean_input_evaluation": True,
            "three_planned_contrasts_multiplicity_corrected": True,
            "exact_nist20_paper_replication": False,
        },
        "provenance": {
            arm: {
                "decision_sha256": sha256_file(arms_root / arm / "decision.json"),
                "held_per_query_sha256": sha256_file(arms_root / arm / "held_per_query.csv.gz"),
            } for arm in ARMS
        },
        "claim_limit": "Single-seed fold-0 development result; promotion is not a final multiseed claim.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for label, table in paired_tables.items():
            table.to_csv(staging / f"{label}.csv.gz", index=False, compression="gzip")
        (staging / "decision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    args = arguments()
    print(json.dumps(summarize(
        args.arms_root, args.output_dir, args.bootstrap_resamples, args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
