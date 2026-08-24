"""Combine S1c single-peak and S2 sequential headroom under locked gates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_noise_v3_candidate_gradient import (  # noqa: E402
    cluster_ci,
    query_candidate_block,
    strict_metrics,
)
from build_g8r_real_error_atlas import Cache, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s1c-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s1c_topk_matrix"),
    )
    parser.add_argument(
        "--s2-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s2_sequential"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/g8r_noise_v3_s2_headroom.json"),
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/validation/g8r_error_atlas_listwise_cache.npz"),
    )
    parser.add_argument("--minimum-headroom-pp", type=float, default=2.0)
    parser.add_argument("--minimum-recoverable-queries", type=int, default=600)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def corrected_set(frame: pd.DataFrame, base_wrong: set[int]) -> set[int]:
    return set(map(int, frame.loc[
        frame["query_index"].isin(base_wrong) & frame["target_rank"].eq(1), "query_index",
    ]))


def reconstruct_locked_baseline(cache_path: Path) -> pd.DataFrame:
    """Rebuild the full strict candidate-graph baseline used by S1c redecision."""
    cache = Cache(cache_path)
    score_column = cache.feature_names.index("dreams_similarity")
    rows = []
    for query in range(cache.n_queries):
        scores, _, ptr = query_candidate_block(cache, query, score_column)
        rank, _, margin = strict_metrics(scores, ptr)
        rows.append((query, rank, margin, bool(cache.query_has_near[query])))
    return pd.DataFrame(
        rows,
        columns=["query_index", "baseline_rank", "baseline_margin", "has_near"],
    )


def main() -> None:
    args = parse_args()
    s1c_paired_path = args.s1c_dir / "paired_interventions.csv.gz"
    s1c_validation_path = args.s1c_dir / "matrix_validation.json"
    s2_paired_path = args.s2_dir / "paired_interventions.csv.gz"
    s2_validation_path = args.s2_dir / "matrix_validation.json"
    s2_report_path = args.s2_dir / "report.json"
    for path in (
        s1c_paired_path, s1c_validation_path, s2_paired_path,
        s2_validation_path, s2_report_path, args.cache,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    s1c_validation = json.loads(s1c_validation_path.read_text(encoding="utf-8"))
    s2_validation = json.loads(s2_validation_path.read_text(encoding="utf-8"))
    report = json.loads(s2_report_path.read_text(encoding="utf-8"))
    if not str(s1c_validation.get("status", "")).endswith("matrix_validation_passed"):
        raise RuntimeError("S1c validation did not pass")
    if s2_validation.get("status") != "noise_v3_s2_sequential_matrix_validation_passed":
        raise RuntimeError("S2 validation did not pass")
    total = int(report["queries"])
    s1c = pd.read_csv(s1c_paired_path)
    s2 = pd.read_csv(s2_paired_path)
    baseline = reconstruct_locked_baseline(args.cache)
    if len(baseline) != total:
        raise RuntimeError(
            f"candidate-graph/query mismatch: cache={len(baseline)} report={total}"
        )
    base_wrong = set(map(int, baseline.loc[
        baseline["baseline_rank"].gt(1), "query_index",
    ]))
    base_correct = set(map(int, baseline.loc[
        baseline["baseline_rank"].eq(1), "query_index",
    ]))
    near_wrong = set(map(int, baseline.loc[
        baseline["baseline_rank"].gt(1) & baseline["has_near"], "query_index",
    ]))
    official_error_count = len(base_wrong)
    official_near_error_count = len(near_wrong)
    if official_error_count < 1:
        raise RuntimeError("locked full candidate graph contains no official errors")
    if int(report.get("official_errors", -1)) != official_error_count:
        raise RuntimeError(
            "S2 report and reconstructed full baseline disagree on official errors"
        )
    s1c_recoverable = corrected_set(s1c.loc[
        s1c["selector"].str.startswith("candidate_gradient")
        | s1c["selector"].str.startswith("role_confounder")
    ], base_wrong)

    action_results = {}
    s2_recoverable: set[int] = set()
    by_step: dict[int, set[int]] = {}
    specific_safe_actions = []
    for position, (key, group) in enumerate(s2.groupby(
        ["selector", "attenuation", "step"], sort=True,
    )):
        selector, attenuation, step = key
        corrected = corrected_set(group, base_wrong)
        introduced = set(map(int, group.loc[
            group["query_index"].isin(base_correct) & group["target_rank"].gt(1), "query_index",
        ]))
        s2_recoverable |= corrected
        by_step.setdefault(int(step), set()).update(corrected)
        wrong = group.loc[group["query_index"].isin(base_wrong)].copy()
        identity_ci = cluster_ci(
            wrong, "target_minus_random_top1", "query_ik14", args.bootstrap,
            args.seed + position,
        )
        formula_ci = cluster_ci(
            wrong, "target_minus_random_top1", "query_formula", args.bootstrap,
            args.seed + 10_000 + position,
        )
        specific = bool(
            identity_ci is not None and formula_ci is not None
            and identity_ci[0] > 0 and formula_ci[0] > 0
        )
        safe = len(corrected) > len(introduced)
        name = f"{selector}|a={float(attenuation):.2f}|step={int(step)}"
        action_results[name] = {
            "queries": int(len(group)),
            "identities": int(group["query_ik14"].nunique()),
            "formulas": int(group["query_formula"].nunique()),
            "corrected": int(len(corrected)),
            "introduced": int(len(introduced)),
            "net": int(len(corrected) - len(introduced)),
            "identity_target_minus_random_top1_95ci": identity_ci,
            "formula_target_minus_random_top1_95ci": formula_ci,
            "specificity_gate": specific,
            "safety_gate": safe,
        }
        if int(step) >= 2 and specific and safe:
            specific_safe_actions.append(name)

    combined = s1c_recoverable | s2_recoverable
    cumulative_previous = set(s1c_recoverable)
    step_headroom = {}
    for step in sorted(by_step):
        cumulative = cumulative_previous | by_step[step]
        step_headroom[str(step)] = {
            "sequence_recoverable_at_this_step": int(len(by_step[step])),
            "new_beyond_s1c_and_earlier_steps": int(len(cumulative - cumulative_previous)),
            "combined_recoverable": int(len(cumulative)),
            "combined_delta_recall1": float(len(cumulative) / total),
        }
        cumulative_previous = cumulative

    observed_near_wrong = near_wrong
    targets = {}
    for points in (1.0, 2.0, 4.0):
        required = math.ceil(total * points / 100.0)
        targets[f"{points:.1f}pp"] = {
            "required_net_corrections": int(required),
            "within_combined_oracle_headroom": bool(len(combined) >= required),
        }
    output = {
        "status": "noise_v3_s2_sequential_headroom_corrected",
        "queries": total,
        "official_errors": official_error_count,
        "s1c_single_peak_recoverable": int(len(s1c_recoverable)),
        "s2_sequence_recoverable": int(len(s2_recoverable)),
        "s2_unique_beyond_s1c": int(len(s2_recoverable - s1c_recoverable)),
        "combined_no_op_aware_oracle": {
            "recoverable_errors": int(len(combined)),
            "recall1": float((total - official_error_count + len(combined)) / total),
            "delta_recall1": float(len(combined) / total),
            "introduced": 0,
            "claim_limit": "Outcome-selected action/stop upper bound; not a policy result.",
        },
        "stepwise_incremental_headroom": step_headroom,
        "near_headroom": {
            "official_near_errors": official_near_error_count,
            "combined_recoverable_near_errors": int(
                len(combined & observed_near_wrong)
            ),
            "fraction_recoverable": float(
                len(combined & observed_near_wrong) / official_near_error_count
            ) if official_near_error_count else None,
            "note": (
                "The denominator is the full candidate graph; the recoverable numerator "
                "is limited to queries with complete paired paths."
            ),
        },
        "action_results": action_results,
        "specific_and_safe_step2_or_step3_actions": specific_safe_actions,
        "performance_targets": targets,
        "preregistered_thresholds": {
            "minimum_headroom_pp": float(args.minimum_headroom_pp),
            "minimum_recoverable_queries": int(args.minimum_recoverable_queries),
        },
        "decision_rule": (
            "Only proceed to cross-fitted policy learning when the combined no-op-aware "
            "oracle passes both locked headroom/query thresholds and at least one step-2/3 "
            "action is safer than no intervention and specifically outperforms matched random paths."
        ),
        "provenance": {
            "s1c_paired_sha256": sha256_file(s1c_paired_path),
            "s1c_validation_sha256": sha256_file(s1c_validation_path),
            "s2_paired_sha256": sha256_file(s2_paired_path),
            "s2_validation_sha256": sha256_file(s2_validation_path),
            "s2_report_sha256": sha256_file(s2_report_path),
            "candidate_graph_sha256": sha256_file(args.cache),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "baseline_rank_audit": {
            "definition": "full strict candidate graph, per-molecule max aggregation",
            "s1c_stored_rank_disagreements": int(sum(
                1 for row in s1c[["query_index", "baseline_rank"]]
                .drop_duplicates("query_index").itertuples(index=False)
                if (int(row.query_index) in base_wrong) != (int(row.baseline_rank) > 1)
            )),
            "s2_stored_rank_disagreements": int(sum(
                1 for row in s2[["query_index", "baseline_rank"]]
                .drop_duplicates("query_index").itertuples(index=False)
                if (int(row.query_index) in base_wrong) != (int(row.baseline_rank) > 1)
            )),
        },
    }
    output["gates"] = {
        "combined_oracle_headroom_ge_preregistered_target": bool(
            100.0 * len(combined) / total >= args.minimum_headroom_pp
        ),
        "combined_recoverable_queries_ge_preregistered_minimum": bool(
            len(combined) >= args.minimum_recoverable_queries
        ),
        "sequential_paths_add_new_recoveries": bool(len(s2_recoverable - s1c_recoverable) > 0),
        "specific_safe_step2_or_step3_exists": bool(specific_safe_actions),
    }
    output["gates"]["pass_to_policy_design"] = all(output["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
