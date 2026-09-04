"""Analyze exact A4 peak actions without confusing oracle headroom with policy gain."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_A4 = ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan"
DEFAULT_S1C = ROOT / "data/validation/g8r_noise_v3_s1c_topk_matrix"
DEFAULT_S2 = ROOT / "data/validation/g8r_noise_v3_s2_sequential"
DEFAULT_S3A = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix"
ROLE_NAME = {-1: "invalid", 0: "identity_only", 1: "confounder_only", 2: "shared", 3: "unmatched"}
GRADE_NAME = {-2: "identity", -1: "unknown", 0: "near", 1: "mid", 2: "far"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a4-dir", type=Path, default=DEFAULT_A4)
    parser.add_argument("--s1c-dir", type=Path, default=DEFAULT_S1C)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2)
    parser.add_argument("--s3a-dir", type=Path, default=DEFAULT_S3A)
    parser.add_argument("--four-point-target", type=float, default=0.04)
    parser.add_argument("--minimum-policy-headroom", type=int, default=1000)
    parser.add_argument("--top-actions-per-query", type=int, default=64)
    return parser.parse_args()


def previous_recoverable(directory: Path) -> set[int]:
    path = directory / "paired_interventions.csv.gz"
    if not path.is_file():
        return set()
    use = ["query_index", "baseline_rank", "target_rank"]
    frame = pd.read_csv(path, usecols=lambda column: column in use)
    if not set(use).issubset(frame.columns):
        return set()
    return set(map(int, frame.loc[
        (frame["baseline_rank"] > 1) & (frame["target_rank"] == 1), "query_index"
    ]))


def action_frame(handle: h5py.File) -> pd.DataFrame:
    return pd.DataFrame({
        "action_index": np.arange(len(handle["action_query"]), dtype=np.int64),
        "scan_position": handle["action_query"][:],
        "token": handle["action_token"][:],
        "role_code": handle["action_role"][:],
        "mz": handle["action_mz"][:],
        "intensity": handle["action_intensity"][:],
        "gradient": handle["action_gradient"][:],
        "predicted_gain": handle["action_predicted_gain"][:],
        "gradient_rank": handle["action_gradient_rank"][:],
        "policy_eligible": handle["action_policy_eligible"][:].astype(bool),
    })


def bin_gradient_rank(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, [-np.inf, 1, 3, 6, 12, 25, 50, np.inf],
        labels=["1", "2-3", "4-6", "7-12", "13-25", "26-50", "51+"],
    ).astype(str)


def main() -> None:
    args = parse_args()
    for name in ("report.json", "scan_queries.csv.gz", "exact_peak_scan.h5"):
        if not (args.a4_dir / name).is_file():
            raise FileNotFoundError(args.a4_dir / name)
    report = json.loads((args.a4_dir / "report.json").read_text(encoding="utf-8"))
    queries = pd.read_csv(args.a4_dir / "scan_queries.csv.gz")
    if queries["scan_position"].tolist() != list(range(len(queries))):
        raise RuntimeError("scan query positions are not contiguous")

    with h5py.File(args.a4_dir / "exact_peak_scan.h5", "r") as handle:
        doses = np.asarray(json.loads(handle.attrs["attenuations_json"]), dtype=float)
        actions = action_frame(handle)
        n_actions = len(actions)
        if len(handle["result_rank"]) != n_actions * len(doses):
            raise RuntimeError("action/result cardinality mismatch")
        rank = handle["result_rank"][:].reshape(n_actions, len(doses))
        margin = handle["result_margin"][:].reshape(n_actions, len(doses))
        adversarial_molecule = handle["result_adversarial_molecule_local"][:].reshape(
            n_actions, len(doses)
        )

    actions = actions.merge(
        queries[[
            "scan_position", "query_index", "query_ik14", "query_formula", "scan_kind",
            "baseline_rank", "baseline_margin", "has_near",
            "baseline_adversarial_mces_grade",
        ] + [column for column in (
            "score_error_family", "positive_deficit", "negative_excess",
            "rules_favor_positive", "rules_favor_wrong",
        ) if column in queries.columns]],
        on="scan_position", validate="many_to_one",
    )
    actions["role"] = actions["role_code"].map(ROLE_NAME)
    actions["gradient_rank_bin"] = bin_gradient_rank(actions["gradient_rank"])
    actions["intensity_bin"] = pd.qcut(
        actions["intensity"].rank(method="first"), 5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    ).astype(str)

    dose_summaries = []
    candidate_rows = []
    recoverable_by_dose: dict[str, set[int]] = {}
    at_risk_by_dose: dict[str, set[int]] = {}
    for dose_index, dose in enumerate(doses):
        local = actions.copy()
        local["result_rank"] = rank[:, dose_index]
        local["result_margin"] = margin[:, dose_index]
        local["margin_change"] = local["result_margin"] - local["baseline_margin"]
        local["corrected"] = (
            local["scan_kind"].eq("official_error") & local["result_rank"].eq(1)
            & local["policy_eligible"]
        )
        local["introduced"] = (
            local["scan_kind"].eq("safety_control") & local["result_rank"].gt(1)
            & local["policy_eligible"]
        )
        recoverable = set(map(int, local.loc[local["corrected"], "query_index"]))
        at_risk = set(map(int, local.loc[local["introduced"], "query_index"]))
        recoverable_by_dose[f"{dose:.2f}"] = recoverable
        at_risk_by_dose[f"{dose:.2f}"] = at_risk
        error_actions = local.loc[
            local["scan_kind"].eq("official_error") & local["policy_eligible"]
        ]
        control_actions = local.loc[
            local["scan_kind"].eq("safety_control") & local["policy_eligible"]
        ]
        correlation = spearmanr(
            error_actions["predicted_gain"], error_actions["margin_change"], nan_policy="omit",
        )
        statistic = float(correlation.statistic)
        pvalue = float(correlation.pvalue)
        dose_summaries.append({
            "attenuation": float(dose),
            "unique_recoverable_errors": len(recoverable),
            "unique_at_risk_controls": len(at_risk),
            "correcting_action_rows": int(local["corrected"].sum()),
            "introducing_action_rows": int(local["introduced"].sum()),
            "gradient_exact_spearman": statistic if np.isfinite(statistic) else None,
            "gradient_exact_spearman_p": pvalue if np.isfinite(pvalue) else None,
            "median_best_error_margin_change": float(
                error_actions.groupby("query_index")["margin_change"].max().median()
            ),
            "median_worst_control_margin_change": float(
                control_actions.groupby("query_index")["margin_change"].min().median()
            ),
        })
        for query_index, group in local.groupby("query_index", sort=False):
            ordered = group.sort_values(
                ["corrected", "margin_change", "predicted_gain"],
                ascending=[False, False, False], kind="mergesort",
            ).head(args.top_actions_per_query)
            candidate_rows.append(ordered.assign(attenuation=float(dose)))

    candidate = pd.concat(candidate_rows, ignore_index=True)
    all_recoverable = set().union(*recoverable_by_dose.values())
    all_at_risk = set().union(*at_risk_by_dose.values())
    previous_sets = {
        "s1c": previous_recoverable(args.s1c_dir),
        "s2": previous_recoverable(args.s2_dir),
        "s3a": previous_recoverable(args.s3a_dir),
    }
    previous_union = set().union(*previous_sets.values())
    a4_new = all_recoverable - previous_union
    combined = previous_union | all_recoverable
    total_queries = 23876
    requirement = int(math.ceil(total_queries * args.four_point_target))

    matrix_rows = []
    for dose_index, dose in enumerate(doses):
        local = actions.copy()
        local["result_rank"] = rank[:, dose_index]
        local["result_margin"] = margin[:, dose_index]
        local["margin_change"] = local["result_margin"] - local["baseline_margin"]
        local["corrected"] = (
            local["scan_kind"].eq("official_error") & local["result_rank"].eq(1)
            & local["policy_eligible"]
        )
        local["introduced"] = (
            local["scan_kind"].eq("safety_control") & local["result_rank"].gt(1)
            & local["policy_eligible"]
        )
        for key, group in local.groupby(
            ["role", "gradient_rank_bin", "intensity_bin"], observed=True, sort=True,
        ):
            role, gradient_bin, intensity_bin = key
            errors = group.loc[group["scan_kind"].eq("official_error")]
            controls = group.loc[group["scan_kind"].eq("safety_control")]
            matrix_rows.append({
                "attenuation": float(dose), "role": role,
                "gradient_rank_bin": gradient_bin, "intensity_bin": intensity_bin,
                "error_action_rows": int(len(errors)),
                "control_action_rows": int(len(controls)),
                "correcting_action_rows": int(group["corrected"].sum()),
                "introducing_action_rows": int(group["introduced"].sum()),
                "error_action_correction_rate": float(errors["corrected"].mean()) if len(errors) else None,
                "control_action_introduction_rate": float(controls["introduced"].mean()) if len(controls) else None,
                "mean_error_margin_change": float(errors["margin_change"].mean()) if len(errors) else None,
                "mean_control_margin_change": float(controls["margin_change"].mean()) if len(controls) else None,
            })
    matrix = pd.DataFrame(matrix_rows)

    # How many exact rescues would have been missed by a gradient top-k policy?
    gradient_coverage = {}
    for top_k in (1, 3, 6, 12, 25, 50, 100):
        rescued: set[int] = set()
        for dose_index in range(len(doses)):
            mask = (
                actions["scan_kind"].eq("official_error") & actions["policy_eligible"]
                & actions["gradient_rank"].between(1, top_k)
                & (rank[:, dose_index] == 1)
            )
            rescued.update(map(int, actions.loc[mask, "query_index"]))
        gradient_coverage[str(top_k)] = {
            "recoverable_errors": len(rescued),
            "fraction_of_exact_oracle": len(rescued) / max(len(all_recoverable), 1),
        }

    error_family = {}
    if "score_error_family" in actions.columns:
        error_query = queries.loc[queries["scan_kind"].eq("official_error")].set_index("query_index")
        for family, group in error_query.groupby("score_error_family", dropna=False):
            family_queries = set(map(int, group.index))
            error_family[str(family)] = {
                "errors": len(family_queries),
                "a4_recoverable": len(family_queries & all_recoverable),
                "a4_recoverable_fraction": len(family_queries & all_recoverable) / max(len(family_queries), 1),
                "new_beyond_previous": len(family_queries & a4_new),
            }

    decision = {
        "status": "noise_v3_a4_exact_peak_scan_decision",
        "integrity": {
            "formal": bool(report["formal"]),
            "official_errors_scanned": int(report["official_errors_scanned"]),
            "safety_controls": int(report["unique_safety_controls"]),
            "fragment_actions": int(report["fragment_actions"]),
            "exact_variants": int(report["exact_variants"]),
        },
        "dose_results": dose_summaries,
        "exact_action_oracle": {
            "unique_recoverable_errors": len(all_recoverable),
            "unique_at_risk_controls": len(all_at_risk),
            "new_recoverable_beyond_s1c_s2_s3a": len(a4_new),
            "previous_union_recomputed": len(previous_union),
            "combined_recoverable": len(combined),
            "combined_delta_recall1_upper_bound": len(combined) / total_queries,
            "four_point_requirement": requirement,
            "headroom_surplus_or_shortfall": len(combined) - requirement,
            "minimum_policy_headroom": args.minimum_policy_headroom,
        },
        "gradient_rank_coverage_of_exact_oracle": gradient_coverage,
        "error_family_coverage": error_family,
        "gates": {
            "a4_adds_at_least_157_new_errors": len(a4_new) >= 157,
            "combined_headroom_reaches_four_points": len(combined) >= requirement,
            "combined_headroom_reaches_policy_floor": len(combined) >= args.minimum_policy_headroom,
        },
        "decision": (
            "Advance to formula-group OOF action-policy fitting only when exact action headroom "
            "reaches the preregistered floor. Otherwise expand the positive-deficit branch first."
        ),
        "claim_limit": (
            "Every oracle quantity chooses actions after observing outcomes. It is action-space "
            "headroom, not a learned policy and not model performance."
        ),
    }

    staging = Path(tempfile.mkdtemp(prefix="noise_v3_a4_analysis_", dir=args.a4_dir.parent))
    try:
        candidate.to_csv(staging / "policy_candidate_actions.csv.gz", index=False, compression="gzip")
        matrix.to_csv(staging / "exact_action_matrix.csv", index=False)
        (staging / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        for name in ("policy_candidate_actions.csv.gz", "exact_action_matrix.csv", "decision.json"):
            shutil.move(str(staging / name), str(args.a4_dir / name))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
