"""Stratified analysis for discovery residual-pair peak occlusion.

The causal unit is a molecule pair. Directed perturbations are averaged within
the pair before any confidence interval is computed, preventing two spectrum
directions (or repeated spectra) from being treated as independent evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EFFECTS = (
    ROOT / "data/validation/dreams_residual_pair_occlusion_discovery"
    / "paired_effects.csv"
)
DEFAULT_SOURCE = (
    ROOT / "data/validation/dreams_residual_peak_mechanisms_large"
    / "discovery_peak_mechanisms.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "data/validation/dreams_residual_pair_occlusion_discovery"
    / "stratified"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", choices=["discovery", "confirmation"], default="discovery")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize(group: pd.DataFrame, seed: int, bootstrap: int) -> dict[str, object]:
    values = group["directional_support"].to_numpy(float)
    if not len(values):
        return {}
    low, high = bootstrap_ci(values, bootstrap, seed)
    return {
        "molecule_pairs": int(len(group)),
        "mean_directional_support": float(values.mean()),
        "median_directional_support": float(np.median(values)),
        "fraction_supportive": float((values > 0).mean()),
        "bootstrap_95ci_low": low,
        "bootstrap_95ci_high": high,
        "median_removed_peaks": float(group["removed_count"].median()),
        "mean_clean_pair_cosine": float(group["clean_pair_cosine"].mean()),
    }


def add_stratum(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    column: str,
    bootstrap: int,
    seed: int,
) -> None:
    for position, (value, group) in enumerate(frame.groupby(column, dropna=False, sort=True)):
        if len(group) < 10:
            continue
        rows.append({
            "stratum": column,
            "level": str(value),
            **summarize(group, seed + position, bootstrap),
        })


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(args.effects)
    source = pd.read_csv(args.source)
    effects["pair_row_low"] = effects[["source_index", "target_index"]].min(axis=1)
    effects["pair_row_high"] = effects[["source_index", "target_index"]].max(axis=1)
    source["pair_row_low"] = source[["row_a", "row_b"]].min(axis=1)
    source["pair_row_high"] = source[["row_a", "row_b"]].max(axis=1)

    pair_effects = effects.groupby(
        ["pair_position", "pair_key", "mechanism_screen", "pair_row_low", "pair_row_high"],
        as_index=False,
    ).agg(
        directional_support=("directional_support", "mean"),
        target_minus_random_cosine_change=(
            "target_minus_random_cosine_change", "mean"
        ),
        removed_count=("removed_count", "mean"),
        target_class_peak_count=("target_class_peak_count", "mean"),
        clean_pair_cosine=("clean_pair_cosine", "mean"),
        directions_observed=("source_side", "nunique"),
    )
    metadata_columns = [
        "pair_row_low", "pair_row_high", "mechanism_screen", "pair_type",
        "same_formula", "tanimoto", "tanimoto_bin",
        "same_instrument", "instrument_a", "instrument_b", "ce_delta",
        "residual_official_finetuned", "absolute_residual",
        "fragment_min_shared_intensity", "fragment_top10_match_fraction",
        "neutral_loss_min_shared_intensity",
    ]
    metadata = source[metadata_columns].drop_duplicates(
        ["pair_row_low", "pair_row_high", "mechanism_screen"]
    )
    merged = pair_effects.merge(
        metadata,
        on=["pair_row_low", "pair_row_high", "mechanism_screen"],
        how="left",
        validate="one_to_one",
    )
    valid_mechanism_pair = (
        merged["mechanism_screen"].eq("shared_major_peak_overaggregation_candidate")
        & merged["pair_type"].eq("different_identity")
    ) | (
        merged["mechanism_screen"].str.endswith("identity_instability")
        & merged["pair_type"].eq("same_identity")
    )
    invalid_pair_type_rows = int((~valid_mechanism_pair).sum())
    merged = merged.loc[valid_mechanism_pair].copy()
    merged["instrument_relation"] = np.where(
        merged["same_instrument"].fillna(False), "same_instrument", "cross_instrument"
    )
    merged["removed_peak_bin"] = pd.cut(
        merged["removed_count"], bins=[0, 2, 5, 8, 12], include_lowest=True
    ).astype(str)
    merged.to_csv(args.output_dir / "pair_level_effects.csv", index=False)

    rows: list[dict[str, object]] = []
    for mechanism_position, (mechanism, mechanism_group) in enumerate(
        merged.groupby("mechanism_screen", sort=True)
    ):
        base_seed = args.seed + mechanism_position * 100
        rows.append({
            "mechanism_screen": mechanism,
            "stratum": "all",
            "level": "all",
            **summarize(mechanism_group, base_seed, args.bootstrap),
        })
        strata = ["same_formula", "instrument_relation", "removed_peak_bin"]
        if mechanism == "shared_major_peak_overaggregation_candidate":
            strata.append("tanimoto_bin")
        for stratum_position, column in enumerate(strata, start=1):
            local: list[dict[str, object]] = []
            add_stratum(
                local, mechanism_group, column, args.bootstrap,
                base_seed + stratum_position * 10,
            )
            rows.extend({"mechanism_screen": mechanism, **item} for item in local)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "stratified_summary.csv", index=False)
    report = {
        "status": f"{args.cohort}_occlusion_stratified_analysis",
        "causal_unit": "molecule_pair; two intervention directions averaged",
        "pairs_analyzed": int(len(merged)),
        "invalid_pair_type_rows_excluded": invalid_pair_type_rows,
        "pairs_with_both_directions": int((merged["directions_observed"] == 2).sum()),
        "mechanisms": {
            mechanism: summarize(group, args.seed + index, args.bootstrap)
            for index, (mechanism, group) in enumerate(
                merged.groupby("mechanism_screen", sort=True)
            )
        },
        "claim_boundary": (
            "Discovery-only causal screening. Mechanisms and thresholds must be frozen "
            "before independent confirmation."
            if args.cohort == "discovery"
            else "Independent confirmation using discovery-frozen thresholds and interventions."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
