"""Matched-query refinement of the A4 all-peak exact intervention scan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ROLE = {-1: "invalid", 0: "identity_only", 1: "confounder_only", 2: "shared", 3: "unmatched"}


def rank_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, [-np.inf, 1, 3, 6, 12, 25, 50, np.inf],
        labels=["1", "2-3", "4-6", "7-12", "13-25", "26-50", "51+"],
    ).astype(str)


def formula_cluster_ci(
    effects: pd.DataFrame, bootstrap: int, seed: int,
) -> tuple[float, list[float]]:
    values = effects.groupby("query_formula", sort=False)["effect"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(values, len(values), replace=True).mean() for _ in range(bootstrap)
    ])
    return float(values.mean()), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a4-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    queries = pd.read_csv(args.a4_dir / "scan_queries.csv.gz")
    matches = pd.read_csv(args.a4_dir / "safety_control_matches.csv.gz")
    decision = json.loads((args.a4_dir / "decision.json").read_text(encoding="utf-8"))
    with h5py.File(args.a4_dir / "exact_peak_scan.h5", "r") as handle:
        doses = np.asarray(json.loads(handle.attrs["attenuations_json"]), float)
        n_actions = len(handle["action_query"])
        action = pd.DataFrame({
            "action_index": np.arange(n_actions),
            "scan_position": handle["action_query"][:],
            "role_code": handle["action_role"][:],
            "gradient_rank": handle["action_gradient_rank"][:],
            "intensity": handle["action_intensity"][:],
            "predicted_gain": handle["action_predicted_gain"][:],
            "policy_eligible": handle["action_policy_eligible"][:].astype(bool),
        })
        rank = handle["result_rank"][:].reshape(n_actions, len(doses))
        margin = handle["result_margin"][:].reshape(n_actions, len(doses))
    action = action.merge(
        queries[[
            "scan_position", "query_index", "query_formula", "scan_kind", "baseline_rank",
            "baseline_margin", "has_near", "score_error_family", "rules_favor_positive",
            "rules_favor_wrong",
        ]], on="scan_position", validate="many_to_one",
    )
    action["role"] = action["role_code"].map(ROLE)
    action["gradient_rank_bin"] = rank_bin(action["gradient_rank"])
    action["intensity_bin"] = pd.qcut(
        action["intensity"].rank(method="first"), 5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    ).astype(str)
    expanded = []
    for dose_index, dose in enumerate(doses):
        local = action.copy()
        local["attenuation"] = float(dose)
        local["result_rank"] = rank[:, dose_index]
        local["result_margin"] = margin[:, dose_index]
        local["margin_change"] = local["result_margin"] - local["baseline_margin"]
        local["corrected"] = (
            local["scan_kind"].eq("official_error") & local["policy_eligible"]
            & local["result_rank"].eq(1)
        )
        local["introduced"] = (
            local["scan_kind"].eq("safety_control") & local["policy_eligible"]
            & local["result_rank"].gt(1)
        )
        expanded.append(local)
    frame = pd.concat(expanded, ignore_index=True)
    error_meta = queries.loc[queries["scan_kind"].eq("official_error")].set_index("query_index")
    matched = matches.merge(
        queries[["query_index", "query_formula"]].rename(
            columns={"query_index": "error_query_index"}
        ), on="error_query_index", validate="many_to_one",
    )

    coarse = []
    for position, (key, group) in enumerate(frame.groupby(
        ["attenuation", "role", "gradient_rank_bin"], observed=True, sort=True,
    )):
        dose, role, gradient_bin = key
        error_success = set(map(int, group.loc[group["corrected"], "query_index"]))
        control_harm = set(map(int, group.loc[group["introduced"], "query_index"]))
        effect = matched[["error_query_index", "control_query_index", "query_formula"]].copy()
        effect["error_success"] = effect["error_query_index"].isin(error_success).astype(float)
        effect["control_harm"] = effect["control_query_index"].isin(control_harm).astype(float)
        effect = effect.groupby(
            ["error_query_index", "query_formula"], as_index=False,
        ).agg(error_success=("error_success", "first"), control_harm=("control_harm", "mean"))
        effect["effect"] = effect["error_success"] - effect["control_harm"]
        macro, ci = formula_cluster_ci(effect, args.bootstrap, args.seed + position)
        coarse.append({
            "attenuation": float(dose), "role": role, "gradient_rank_bin": gradient_bin,
            "corrected_error_queries": len(error_success),
            "introduced_control_queries": len(control_harm),
            "matched_query_mean_effect": float(effect["effect"].mean()),
            "formula_macro_effect": macro, "formula_cluster_ci_low": ci[0],
            "formula_cluster_ci_high": ci[1],
            "matched_safe_cell": bool(ci[0] > 0 and len(error_success) >= 20),
        })
    coarse_frame = pd.DataFrame(coarse)

    eligible = frame.loc[frame["policy_eligible"]].copy()
    errors = eligible.loc[eligible["scan_kind"].eq("official_error")]
    controls = eligible.loc[eligible["scan_kind"].eq("safety_control")]
    best_error = errors.sort_values(
        ["query_index", "result_margin", "predicted_gain"],
        ascending=[True, False, False], kind="mergesort",
    ).drop_duplicates("query_index")
    worst_control = controls.sort_values(
        ["query_index", "result_margin"], ascending=[True, True], kind="mergesort",
    ).drop_duplicates("query_index")
    best_error["recoverable"] = best_error["result_rank"].eq(1)
    worst_control["at_risk"] = worst_control["result_rank"].gt(1)

    best_distributions = {
        "recoverable_best_action_role": best_error.loc[best_error["recoverable"], "role"].value_counts().to_dict(),
        "recoverable_best_action_attenuation": {
            str(key): int(value) for key, value in best_error.loc[
                best_error["recoverable"], "attenuation"
            ].value_counts().sort_index().items()
        },
        "recoverable_best_action_gradient_rank": {
            "median": float(best_error.loc[best_error["recoverable"], "gradient_rank"].median()),
            "p90": float(best_error.loc[best_error["recoverable"], "gradient_rank"].quantile(0.9)),
        },
        "at_risk_worst_action_role": worst_control.loc[worst_control["at_risk"], "role"].value_counts().to_dict(),
        "at_risk_worst_action_attenuation": {
            str(key): int(value) for key, value in worst_control.loc[
                worst_control["at_risk"], "attenuation"
            ].value_counts().sort_index().items()
        },
    }
    # Convert numpy values from value_counts.
    for name in ("recoverable_best_action_role", "at_risk_worst_action_role"):
        best_distributions[name] = {
            str(key): int(value) for key, value in best_distributions[name].items()
        }

    family_rows = []
    recoverable_set = set(map(int, best_error.loc[best_error["recoverable"], "query_index"]))
    for family, group in error_meta.groupby("score_error_family", dropna=False):
        query_set = set(map(int, group.index))
        family_rows.append({
            "score_error_family": str(family), "errors": len(query_set),
            "recoverable": len(query_set & recoverable_set),
            "unrecoverable": len(query_set - recoverable_set),
            "recoverable_fraction": len(query_set & recoverable_set) / len(query_set),
        })

    rule_rows = []
    for column in ("rules_favor_positive", "rules_favor_wrong"):
        for value, group in error_meta.groupby(column, dropna=False):
            query_set = set(map(int, group.index))
            rule_rows.append({
                "rule_screen": column, "screen_value": str(value), "errors": len(query_set),
                "recoverable": len(query_set & recoverable_set),
                "recoverable_fraction": len(query_set & recoverable_set) / len(query_set),
            })

    refined = {
        "status": "noise_v3_a4_refined_audit_complete",
        "integrity": {
            "errors": int((queries["scan_kind"] == "official_error").sum()),
            "controls": int((queries["scan_kind"] == "safety_control").sum()),
            "actions": int(len(action)), "variants": int(len(frame)),
            "matched_edges": int(len(matches)),
        },
        "headroom": decision["exact_action_oracle"],
        "best_and_worst_action_distributions": best_distributions,
        "matched_safe_cells": int(coarse_frame["matched_safe_cell"].sum()),
        "top_matched_cells": coarse_frame.sort_values(
            ["matched_safe_cell", "formula_cluster_ci_low", "corrected_error_queries"],
            ascending=[False, False, False],
        ).head(20).to_dict(orient="records"),
        "error_family_exact_coverage": family_rows,
        "rule_screen_descriptive_coverage": rule_rows,
        "decision": {
            "fit_final_policy_now": False,
            "reason": (
                "Combined outcome-oracle headroom is 920, below both the 956 four-point "
                "requirement and the 1000 policy floor. Build the positive-deficit action branch "
                "before policy fitting; retain A4 exact actions as supervised action examples."
            ),
            "retain_for_next_stage": (
                "candidate-conditioned exact actions in matched-safe cells; identity-only actions "
                "remain negative controls; rules remain covariates pending formula-OOF ablation"
            ),
        },
        "claim_limit": (
            "Best/worst actions and matched-cell success use observed outcomes and remain oracle "
            "analyses. They cannot be reported as policy or fine-tuning performance."
        ),
    }
    (args.a4_dir / "refined_audit.json").write_text(
        json.dumps(refined, indent=2), encoding="utf-8",
    )
    coarse_frame.to_csv(args.a4_dir / "matched_action_cells.csv", index=False)
    pd.DataFrame(family_rows).to_csv(args.a4_dir / "error_family_exact_coverage.csv", index=False)
    print(json.dumps(refined, indent=2), flush=True)


if __name__ == "__main__":
    main()
