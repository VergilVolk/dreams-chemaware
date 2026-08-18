"""Combine relabelled v1 effects with the v2 supplemental occlusion run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/dreams_residual_pair_occlusion_discovery_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reused_path = args.output_dir / "paired_effects_reused.csv"
    supplement_path = args.output_dir / "supplement/paired_effects.csv"
    reused = pd.read_csv(reused_path)
    supplement = pd.read_csv(supplement_path)
    for column in ("pair_row_low", "pair_row_high"):
        if column in reused:
            reused = reused.drop(columns=column)
    combined = pd.concat([reused, supplement], ignore_index=True, sort=False)
    combined["pair_row_low"] = combined[["source_index", "target_index"]].min(axis=1).astype(int)
    combined["pair_row_high"] = combined[["source_index", "target_index"]].max(axis=1).astype(int)
    duplicate_directions = int(combined.duplicated(
        ["pair_row_low", "pair_row_high", "source_side"], keep=False
    ).sum())
    if duplicate_directions:
        raise RuntimeError(f"Duplicate directed interventions: {duplicate_directions}")
    combined.to_csv(args.output_dir / "paired_effects.csv", index=False)
    report = {
        "status": "v2_discovery_occlusion_finalized",
        "reused_directed_effects": int(len(reused)),
        "supplemental_directed_effects": int(len(supplement)),
        "combined_directed_effects": int(len(combined)),
        "combined_unique_pairs": int(
            combined[["pair_row_low", "pair_row_high"]].drop_duplicates().shape[0]
        ),
        "duplicate_directions": duplicate_directions,
    }
    (args.output_dir / "finalize_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
