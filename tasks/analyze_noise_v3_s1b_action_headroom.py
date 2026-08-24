"""Quantify the no-op-aware oracle and complementarity of the S1a action matrix."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from build_g8r_real_error_atlas import Cache, sha256_file  # noqa: E402
from audit_noise_v3_candidate_gradient import query_candidate_block, strict_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_s1a_single_peak_matrix",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_s1b_action_headroom.json",
    )
    parser.add_argument("--status", default="noise_v3_s1b_action_headroom_complete")
    parser.add_argument("--minimum-headroom-pp", type=float, default=1.0)
    parser.add_argument("--minimum-recoverable-queries", type=int, default=0)
    return parser.parse_args()


def set_summary(values: set[int], base_wrong: set[int]) -> dict:
    return {
        "recoverable_errors": int(len(values)),
        "fraction_of_all_official_errors": float(len(values) / len(base_wrong)),
    }


def main() -> None:
    args = parse_args()
    if args.minimum_headroom_pp <= 0 or args.minimum_recoverable_queries < 0:
        raise ValueError("invalid preregistered decision thresholds")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report_path = args.matrix_dir / "report.json"
    validation_path = args.matrix_dir / "matrix_validation.json"
    paired_path = args.matrix_dir / "paired_interventions.csv.gz"
    for path in (report_path, validation_path, paired_path, args.cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not str(validation.get("status", "")).endswith("matrix_validation_passed"):
        raise RuntimeError("matrix integrity gate did not pass")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path)
    cache = Cache(args.cache)
    score_column = cache.feature_names.index("dreams_similarity")

    baseline_rows = []
    for query in range(cache.n_queries):
        scores, _, ptr = query_candidate_block(cache, query, score_column)
        rank, _, margin = strict_metrics(scores, ptr)
        baseline_rows.append((query, rank, margin, bool(cache.query_has_near[query])))
    baseline = pd.DataFrame(
        baseline_rows, columns=["query_index", "baseline_rank", "baseline_margin", "has_near"],
    )
    base_wrong = set(map(int, baseline.loc[baseline["baseline_rank"] > 1, "query_index"]))
    base_correct = set(map(int, baseline.loc[baseline["baseline_rank"] == 1, "query_index"]))
    if len(base_wrong) < 1000 or len(baseline) != int(report["queries"]):
        raise RuntimeError("baseline reconstruction mismatch")

    paired["action"] = paired.apply(
        lambda row: f"{row['selector']}|a={float(row['attenuation']):.2f}", axis=1,
    )
    paired["target_correct"] = paired["target_rank"].eq(1)
    paired["corrected"] = paired["query_index"].isin(base_wrong) & paired["target_correct"]
    paired["introduced"] = paired["query_index"].isin(base_correct) & ~paired["target_correct"]

    action_results = {}
    recovered_by_action: dict[str, set[int]] = {}
    harmed_by_action: dict[str, set[int]] = {}
    for action, frame in paired.groupby("action", sort=True):
        corrected = set(map(int, frame.loc[frame["corrected"], "query_index"]))
        introduced = set(map(int, frame.loc[frame["introduced"], "query_index"]))
        recovered_by_action[action] = corrected
        harmed_by_action[action] = introduced
        action_results[action] = {
            "eligible_queries": int(frame["query_index"].nunique()),
            "corrected": int(len(corrected)),
            "introduced": int(len(introduced)),
            "net": int(len(corrected) - len(introduced)),
        }

    gradient = set().union(*(
        values for key, values in recovered_by_action.items()
        if key.startswith("candidate_gradient")
    ))
    confounder = set().union(*(
        values for key, values in recovered_by_action.items()
        if key.startswith("role_confounder")
    ))
    identity = set().union(*(
        values for key, values in recovered_by_action.items()
        if key.startswith("role_identity|")
    ))
    any_noncontrol = gradient | confounder
    any_action = any_noncontrol | identity
    near_wrong = set(map(int, baseline.loc[
        (baseline["baseline_rank"] > 1) & baseline["has_near"], "query_index"
    ]))

    # The no-op-aware oracle never acts on a baseline-correct query and chooses
    # an action only when that action truly corrects a baseline error. It is an
    # upper bound, not a deployable policy.
    total = len(baseline)
    oracle_recall = (len(base_correct) + len(any_noncontrol)) / total
    targets = {}
    for points in (0.5, 1.0, 2.0, 4.0):
        required = math.ceil(total * points / 100.0)
        targets[f"{points:.1f}pp"] = {
            "required_net_corrections": int(required),
            "within_noncontrol_oracle_headroom": bool(len(any_noncontrol) >= required),
        }

    output = {
        "status": args.status,
        "queries": total,
        "official_errors": len(base_wrong),
        "official_recall1": float(len(base_correct) / total),
        "action_results": action_results,
        "no_op_aware_oracle": {
            "noncontrol_recoverable_errors": int(len(any_noncontrol)),
            "recall1": float(oracle_recall),
            "delta_recall1": float(len(any_noncontrol) / total),
            "introduced": 0,
            "claim_limit": "Uses action outcomes to select the action; headroom only.",
        },
        "selector_complementarity": {
            "candidate_gradient": set_summary(gradient, base_wrong),
            "role_confounder": set_summary(confounder, base_wrong),
            "intersection": set_summary(gradient & confounder, base_wrong),
            "gradient_only": set_summary(gradient - confounder, base_wrong),
            "confounder_only": set_summary(confounder - gradient, base_wrong),
            "either_noncontrol": set_summary(any_noncontrol, base_wrong),
            "identity_negative_control_recoveries": int(len(identity)),
        },
        "near_headroom": {
            "official_near_errors": int(len(near_wrong)),
            "noncontrol_recoverable_near_errors": int(len(any_noncontrol & near_wrong)),
            "fraction_recoverable": float(
                len(any_noncontrol & near_wrong) / len(near_wrong)
            ) if near_wrong else None,
        },
        "performance_targets": targets,
        "policy_dataset": {
            "query_action_rows": int(len(paired)),
            "queries_with_any_noncontrol_rescue": int(len(any_noncontrol)),
            "positive_action_rows": int(paired.loc[
                paired["corrected"] & ~paired["selector"].eq("role_identity")
            ].shape[0]),
            "harmful_action_rows": int(paired.loc[paired["introduced"]].shape[0]),
        },
        "decision_rule": (
            "Proceed to nonlinear action-policy cross-fitting only if the no-op-aware "
            "noncontrol oracle reaches the explicitly supplied headroom and recoverable-query "
            "thresholds and both selectors contribute nonredundant recoveries."
        ),
        "preregistered_thresholds": {
            "minimum_headroom_pp": float(args.minimum_headroom_pp),
            "minimum_recoverable_queries": int(args.minimum_recoverable_queries),
        },
        "provenance": {
            "matrix_report_sha256": sha256_file(report_path),
            "matrix_validation_sha256": sha256_file(validation_path),
            "paired_interventions_sha256": sha256_file(paired_path),
            "candidate_graph_sha256": sha256_file(args.cache),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    output["gates"] = {
        "oracle_headroom_ge_preregistered_target": bool(
            100.0 * len(any_noncontrol) / total >= args.minimum_headroom_pp
        ),
        "recoverable_queries_ge_preregistered_minimum": bool(
            len(any_noncontrol) >= args.minimum_recoverable_queries
        ),
        "gradient_has_unique_recoveries": bool(len(gradient - confounder) >= 25),
        "confounder_has_unique_recoveries": bool(len(confounder - gradient) >= 10),
        "policy_rows_ge_1000": bool(output["policy_dataset"]["positive_action_rows"] >= 1000),
    }
    output["gates"]["pass_to_policy_design"] = bool(
        output["gates"]["oracle_headroom_ge_preregistered_target"]
        and output["gates"]["recoverable_queries_ge_preregistered_minimum"]
        and output["gates"]["gradient_has_unique_recoveries"]
        and output["gates"]["confounder_has_unique_recoveries"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
