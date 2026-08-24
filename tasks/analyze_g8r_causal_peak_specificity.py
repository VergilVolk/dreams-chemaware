"""Test whether causal peak effects are specific to the screened DreaMS errors.

The first-stage occlusion audit establishes a generic geometric effect.  This
second stage asks the harder question: after matching protected-correct cases
on direction, clean similarity and deletion size, is the effect larger in the
corresponding error arm?  Only this difference-in-differences style residual is
eligible to define a fine-tuning pool.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data/validation/g8r_causal_peak_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_causal_peak_specificity")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--min-controls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def cluster_ci(frame: pd.DataFrame, cluster: str, column: str, n: int, seed: int) -> list[float] | None:
    values = frame.groupby(cluster, sort=False)[column].mean().dropna().to_numpy(float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for index in range(n):
        draws[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def add_matching_strata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["removed_bin"] = pd.cut(
        output["removed_count"], bins=[0, 3, 6, 9, np.inf], labels=False,
        include_lowest=True,
    ).astype(int)
    output["similarity_bin"] = -1
    for (_, direction), index in output.groupby(["arm", "direction"], sort=False).groups.items():
        values = output.loc[index, "clean_similarity"]
        # Rank first so duplicated cosine values cannot make qcut drop bins in
        # a data-dependent fashion.
        bins = pd.qcut(values.rank(method="first"), q=5, labels=False)
        output.loc[index, "similarity_bin"] = bins.astype(int)
    output["similarity_bin"] = output["similarity_bin"].astype(int)
    return output


def attach_control_expectation(frame: pd.DataFrame, min_controls: int) -> pd.DataFrame:
    controls = frame.loc[frame["transition"] == "protected_correct"].copy()
    if controls.empty:
        raise RuntimeError("no protected-correct controls in causal audit")
    fine = ["arm", "direction", "removed_bin", "similarity_bin"]
    coarse = ["arm", "direction", "similarity_bin"]
    global_keys = ["arm", "direction"]

    def table(keys: list[str], suffix: str) -> pd.DataFrame:
        return controls.groupby(keys, as_index=False).agg(**{
            f"control_mean_{suffix}": ("directional_support", "mean"),
            f"control_n_{suffix}": ("directional_support", "size"),
        })

    output = frame.merge(table(fine, "fine"), on=fine, how="left", validate="many_to_one")
    output = output.merge(table(coarse, "coarse"), on=coarse, how="left", validate="many_to_one")
    output = output.merge(table(global_keys, "global"), on=global_keys, how="left", validate="many_to_one")
    use_fine = output["control_n_fine"].fillna(0) >= min_controls
    use_coarse = (~use_fine) & (output["control_n_coarse"].fillna(0) >= min_controls)
    output["control_expected_support"] = output["control_mean_global"]
    output["control_match_level"] = "arm_direction"
    output.loc[use_coarse, "control_expected_support"] = output.loc[use_coarse, "control_mean_coarse"]
    output.loc[use_coarse, "control_match_level"] = "similarity"
    output.loc[use_fine, "control_expected_support"] = output.loc[use_fine, "control_mean_fine"]
    output.loc[use_fine, "control_match_level"] = "similarity_and_deletion_size"
    if output["control_expected_support"].isna().any():
        raise RuntimeError("some interventions lack any protected-correct control expectation")
    output["specific_excess_support"] = output["directional_support"] - output["control_expected_support"]
    return output


def summarize_group(group: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    identity_ci = cluster_ci(group, "query_ik14", "specific_excess_support", bootstrap, seed)
    formula_ci = cluster_ci(group, "query_formula", "specific_excess_support", bootstrap, seed + 10_000)
    return {
        "directed_cases": int(len(group)),
        "query_identities": int(group["query_ik14"].nunique()),
        "query_formulas": int(group["query_formula"].nunique()),
        "mean_raw_directional_support": float(group["directional_support"].mean()),
        "mean_matched_control_support": float(group["control_expected_support"].mean()),
        "mean_specific_excess_support": float(group["specific_excess_support"].mean()),
        "median_specific_excess_support": float(group["specific_excess_support"].median()),
        "specific_supportive_fraction": float((group["specific_excess_support"] > 0).mean()),
        "identity_cluster_bootstrap_95ci": identity_ci,
        "formula_cluster_bootstrap_95ci": formula_ci,
        "specificity_gate": bool(
            identity_ci is not None and formula_ci is not None
            and identity_ci[0] > 0 and formula_ci[0] > 0
            and group["query_ik14"].nunique() >= 100
        ),
        "control_match_levels": {
            str(key): int(value) for key, value in group["control_match_level"].value_counts().items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    selected_path = args.audit_dir / "selected_queries.csv.gz"
    paired_path = args.audit_dir / "paired_effects.csv.gz"
    for path in (selected_path, paired_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected = pd.read_csv(selected_path)
    paired = pd.read_csv(paired_path)
    flags = ["positive_deficit", "negative_excess", "shared_major_peak_screen", "neutral_loss_convergence_screen"]
    missing = set(flags + ["query_index"]) - set(selected.columns)
    if missing:
        raise RuntimeError(f"selected-query table missing columns: {sorted(missing)}")
    for column in flags:
        selected[column] = as_bool(selected[column])
    lookup = selected[["query_index"] + flags].drop_duplicates("query_index")
    if len(lookup) != selected["query_index"].nunique():
        raise RuntimeError("screening flags are not unique per query index")
    frame = paired.merge(lookup, on="query_index", how="left", validate="many_to_one")
    if frame[flags].isna().any().any():
        raise RuntimeError("causal effects could not be joined to all screening flags")
    frame = attach_control_expectation(add_matching_strata(frame), args.min_controls)

    arm_flag = {"positive_deficit": "positive_deficit", "negative_excess": "negative_excess"}
    report_groups: dict[str, dict] = {}
    eligible_parts = []
    position = 0
    for arm, flag in arm_flag.items():
        cases = frame.loc[(frame["arm"] == arm) & frame[flag] & (frame["transition"] != "protected_correct")]
        for transition, group in cases.groupby("transition", sort=True):
            key = f"{arm}|{transition}|screen_matched"
            report_groups[key] = summarize_group(group, args.bootstrap, args.seed + position)
            if report_groups[key]["specificity_gate"]:
                eligible_parts.append(group.assign(training_arm=arm))
            position += 1

    eligible = pd.concat(eligible_parts, ignore_index=True) if eligible_parts else frame.iloc[0:0].copy()
    report = {
        "status": "g8r_causal_peak_specificity_complete",
        "matching": {
            "primary": "arm + direction + clean-similarity quintile + removed-peak bin",
            "fallback": "arm + direction + clean-similarity quintile, then arm + direction",
            "minimum_controls_per_stratum": int(args.min_controls),
        },
        "screen_specific_results": report_groups,
        "eligible_directed_interventions": int(len(eligible)),
        "eligible_query_identities": int(eligible["query_ik14"].nunique()) if len(eligible) else 0,
        "decision_rule": "An arm/transition enters the candidate training pool only when both identity- and formula-cluster CI lower bounds exceed zero with at least 100 identities.",
        "claim_limit": "This establishes intervention specificity relative to matched protected-correct queries; it does not yet establish that training on the intervention improves retrieval.",
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "matched_interventions.csv.gz", index=False, compression="gzip")
        eligible.to_csv(staging / "training_candidate_interventions.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
