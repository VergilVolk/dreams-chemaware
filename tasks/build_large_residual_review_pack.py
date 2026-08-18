"""Build a large, stratified human-review pack from discovery residuals.

The source must be the discovery/development cohort. Confirmation and test
cases are deliberately excluded so that chemical review cannot leak into model
selection. All source cases remain available for machine analysis; this script
only selects a feasible but substantially sized expert-review subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = (
    ROOT / "data/validation/dreams_structure_residual_atlas_large"
    / "discovery_residual_priority_cases.csv"
)
DEFAULT_OUTPUT = ROOT / "data/validation/dreams_structure_residual_atlas_large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def round_robin_stratified(
    frame: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if len(frame) <= n:
        return frame.copy()
    strata_columns = ["tanimoto_bin", "same_formula", "same_instrument"]
    groups = []
    for _, group in frame.groupby(strata_columns, dropna=False, sort=False):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        groups.append(list(indices))
    selected = []
    cursor = 0
    while len(selected) < n and groups:
        group_idx = cursor % len(groups)
        if groups[group_idx]:
            selected.append(groups[group_idx].pop())
        if not groups[group_idx]:
            groups.pop(group_idx)
            if not groups:
                break
            cursor %= len(groups)
        else:
            cursor += 1
    return frame.loc[selected].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source)
    if "allowed_use" in source and not source["allowed_use"].eq(
        "method_development_or_training"
    ).all():
        raise ValueError("Review source contains locked validation cases.")
    if "cohort" in source and not source["cohort"].eq("discovery").all():
        raise ValueError("Only discovery cases may enter the development review pack.")

    rng = np.random.default_rng(args.seed)
    identity = source.loc[
        source["residual_family_official_finetuned"] == "same_identity_instability"
    ].copy()
    remaining_target = max(0, args.target - len(identity))
    high_target = remaining_target // 2
    low_target = remaining_target - high_target
    high = round_robin_stratified(
        source.loc[source["residual_family_official_finetuned"] == "higher_than_structure_expected"],
        high_target, rng,
    )
    low = round_robin_stratified(
        source.loc[source["residual_family_official_finetuned"] == "lower_than_structure_expected"],
        low_target, rng,
    )
    review = pd.concat([identity, high, low], ignore_index=True)
    review = review.sort_values(
        ["residual_family_official_finetuned", "absolute_residual"],
        ascending=[True, False],
    ).reset_index(drop=True)
    review.insert(0, "review_id", [f"DREAMS-RES-{idx + 1:05d}" for idx in range(len(review))])
    review.insert(1, "review_status", "pending")
    for column in (
        "spectrum_quality_verdict",
        "structure_label_verdict",
        "shared_peak_pattern",
        "true_candidate_diagnostic_peaks",
        "wrong_candidate_unique_peaks",
        "suspected_error_mechanism",
        "usable_for_training",
        "reviewer",
        "review_notes",
    ):
        review[column] = ""

    output_csv = args.output_dir / "discovery_expert_review_2000.csv"
    review.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "status": "large_discovery_expert_review_pack",
        "source": str(args.source),
        "source_cases": int(len(source)),
        "selected_cases": int(len(review)),
        "selection_seed": int(args.seed),
        "selection_policy": (
            "all same-identity-instability cases plus round-robin sampling across "
            "Tanimoto bin, same-formula status, and instrument agreement for high/low residuals"
        ),
        "cohort_policy": "discovery only; confirmation and test excluded",
        "by_error_family": {
            str(k): int(v)
            for k, v in review["residual_family_official_finetuned"].value_counts().items()
        },
        "same_formula_cases": int(review["same_formula"].fillna(False).sum()),
        "cross_or_unknown_instrument_cases": int((~review["same_instrument"].fillna(False)).sum()),
        "tanimoto_bins": {
            str(k): int(v) for k, v in review["tanimoto_bin"].value_counts().sort_index().items()
        },
    }
    (args.output_dir / "discovery_expert_review_2000_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
