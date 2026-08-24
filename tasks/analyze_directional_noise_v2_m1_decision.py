"""Final decision audit for directional-noise V2 M1.

Separates two claims that the first report intentionally did not conflate:
robustness of already-correct queries and correction of baseline-wrong queries.
Top-1 transitions under the targeted view are compared with the mean of the
three matched-random views on the same query.
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


def as_bool(series: pd.Series) -> pd.Series:
    """Parse booleans fail-closed instead of treating non-empty strings as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unknown = sorted(set(normalized.dropna()) - allowed)
    if unknown:
        raise ValueError(f"unrecognized boolean values: {unknown[:10]}")
    return normalized.map({"true": True, "1": True, "false": False, "0": False}).astype(bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m1_decision")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def cluster_ci(frame: pd.DataFrame, cluster: str, column: str, n: int, seed: int) -> list[float] | None:
    values = frame.groupby(cluster, sort=False)[column].mean().dropna().to_numpy(float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for index in range(n):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(group: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    identity_ci = cluster_ci(group, "ik14", "target_minus_random_top1_delta", bootstrap, seed)
    formula_ci = cluster_ci(group, "formula", "target_minus_random_top1_delta", bootstrap, seed + 10_000)
    return {
        "queries": int(len(group)),
        "identities": int(group["ik14"].nunique()),
        "baseline_accuracy": float(group["baseline_top1"].mean()),
        "target_accuracy": float(group["target_top1"].mean()),
        "mean_random_accuracy": float(group["random_top1_mean"].mean()),
        "target_corrected": int(((group["baseline_top1"] == 0) & (group["target_top1"] == 1)).sum()),
        "target_introduced": int(((group["baseline_top1"] == 1) & (group["target_top1"] == 0)).sum()),
        "expected_random_corrected": float(((group["baseline_top1"] == 0) * group["random_top1_mean"]).sum()),
        "expected_random_introduced": float(((group["baseline_top1"] == 1) * (1.0 - group["random_top1_mean"])).sum()),
        "mean_target_minus_random_top1_delta": float(group["target_minus_random_top1_delta"].mean()),
        "identity_cluster_top1_delta_95ci": identity_ci,
        "formula_cluster_top1_delta_95ci": formula_ci,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    variant_path = args.m1_dir / "variant_results.csv.gz"
    paired_path = args.m1_dir / "paired_margin_effects.csv.gz"
    selected_path = args.m1_dir / "selected_triples.csv.gz"
    for path in (variant_path, paired_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    variants = pd.read_csv(variant_path)
    paired = pd.read_csv(paired_path)
    selected = pd.read_csv(selected_path)

    key = ["query_row", "positive_row", "negative_row", "ik14", "formula", "adduct"]
    target = variants.loc[variants["condition"] == "targeted", key + ["perturbed_margin"]].copy()
    target = target.rename(columns={"perturbed_margin": "target_margin"})
    random = variants.loc[variants["condition"] == "matched_random"].groupby(key, as_index=False).agg(
        random_top1_mean=("perturbed_margin", lambda values: float((values > 0).mean())),
        random_margin_mean=("perturbed_margin", "mean"),
        random_repeats_observed=("repeat", "count"),
    )
    columns = key + ["baseline_margin", "target_margin_change", "random_margin_change", "target_minus_random_margin_change", "cross_condition_positive"]
    decision = paired[columns].merge(target, on=key, validate="one_to_one")
    decision = decision.merge(random, on=key, validate="one_to_one")
    decision = decision.loc[decision["random_repeats_observed"] == 3].copy()
    decision["cross_condition_positive"] = as_bool(decision["cross_condition_positive"])
    decision["baseline_top1"] = (decision["baseline_margin"] > 0).astype(int)
    decision["target_top1"] = (decision["target_margin"] > 0).astype(int)
    decision["target_top1_delta"] = decision["target_top1"] - decision["baseline_top1"]
    decision["random_top1_delta"] = decision["random_top1_mean"] - decision["baseline_top1"]
    decision["target_minus_random_top1_delta"] = decision["target_top1_delta"] - decision["random_top1_delta"]
    decision["baseline_status"] = np.where(decision["baseline_top1"] == 1, "baseline_correct", "baseline_wrong")

    groups = {
        "overall": decision,
        "baseline_correct": decision.loc[decision["baseline_status"] == "baseline_correct"],
        "baseline_wrong": decision.loc[decision["baseline_status"] == "baseline_wrong"],
        "cross_condition": decision.loc[decision["cross_condition_positive"]],
    }
    summaries = {
        name: summarize(group, args.bootstrap, args.seed + position)
        for position, (name, group) in enumerate(groups.items()) if len(group)
    }
    correct = summaries["baseline_correct"]
    wrong = summaries["baseline_wrong"]
    margin_wrong = decision.loc[decision["baseline_status"] == "baseline_wrong"]
    wrong_margin_identity_ci = cluster_ci(
        margin_wrong, "ik14", "target_minus_random_margin_change", args.bootstrap, args.seed + 20_000,
    )
    wrong_margin_formula_ci = cluster_ci(
        margin_wrong, "formula", "target_minus_random_margin_change", args.bootstrap, args.seed + 30_000,
    )
    gates = {
        "robustness_target_introduced_no_more_than_random": correct["target_introduced"] <= correct["expected_random_introduced"],
        "robustness_identity_top1_ci_nonnegative": correct["identity_cluster_top1_delta_95ci"][0] >= 0,
        "robustness_formula_top1_ci_nonnegative": correct["formula_cluster_top1_delta_95ci"][0] >= 0,
        "error_correction_margin_identity_ci_positive": wrong_margin_identity_ci[0] > 0,
        "error_correction_margin_formula_ci_positive": wrong_margin_formula_ci[0] > 0,
        "error_correction_identity_top1_ci_positive": wrong["identity_cluster_top1_delta_95ci"][0] > 0,
        "error_correction_formula_top1_ci_positive": wrong["formula_cluster_top1_delta_95ci"][0] > 0,
    }
    report = {
        "status": "directional_noise_v2_m1_final_decision",
        "selection_retention": {
            "selected_before_token_filter": int(len(selected)),
            "paired_after_complete_controls": int(len(decision)),
            "fraction_retained": float(len(decision) / len(selected)),
        },
        "top1_results": summaries,
        "baseline_wrong_margin_specificity": {
            "identity_cluster_95ci": wrong_margin_identity_ci,
            "formula_cluster_95ci": wrong_margin_formula_ci,
        },
        "gates": gates,
        "robustness_augmentation_pass": bool(all(value for key, value in gates.items() if key.startswith("robustness_"))),
        "error_correction_pass": bool(all(value for key, value in gates.items() if key.startswith("error_correction_"))),
        "decision": "Robustness and error-correction are separate claims. Only the corresponding passing branch may enter M2; a global M1 pass cannot override a failed error-correction branch.",
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent))
    try:
        decision.to_csv(staging / "query_top1_decisions.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
