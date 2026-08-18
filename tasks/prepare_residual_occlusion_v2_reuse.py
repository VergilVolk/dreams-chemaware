"""Reuse valid v1 occlusion effects under corrected v2 residual labels.

Peak interventions depend on the spectrum pair and on whether shared or unique
peaks are removed. They do not depend on the descriptive mechanism subtype.
Therefore existing effects can be relabelled exactly, while genuinely new v2
pairs are emitted for supplemental encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_residual_pair_peak_occlusion import SUPPORTED


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-effects", type=Path,
        default=ROOT / "data/validation/dreams_residual_pair_occlusion_discovery/paired_effects.csv",
    )
    parser.add_argument(
        "--v2-source", type=Path,
        default=ROOT / "data/validation/dreams_residual_peak_mechanisms_large_v2/discovery_peak_mechanisms.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/dreams_residual_pair_occlusion_discovery_v2",
    )
    return parser.parse_args()


def supported_v2(frame: pd.DataFrame) -> pd.DataFrame:
    supported = frame.loc[frame["mechanism_screen"].isin(SUPPORTED)].copy()
    valid = (
        supported["mechanism_screen"].eq("shared_major_peak_overaggregation_candidate")
        & supported["pair_type"].eq("different_identity")
    ) | (
        supported["mechanism_screen"].str.endswith("identity_instability")
        & supported["pair_type"].eq("same_identity")
    )
    return supported.loc[valid].copy()


def add_pair_rows(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["pair_row_low"] = frame[[left, right]].min(axis=1).astype(int)
    frame["pair_row_high"] = frame[[left, right]].max(axis=1).astype(int)
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old = add_pair_rows(pd.read_csv(args.old_effects), "source_index", "target_index")
    new = add_pair_rows(supported_v2(pd.read_csv(args.v2_source)), "row_a", "row_b")
    label_map = new[["pair_row_low", "pair_row_high", "mechanism_screen"]].rename(
        columns={"mechanism_screen": "mechanism_screen_v2"}
    )
    reused = old.drop(columns=["mechanism_screen"]).merge(
        label_map, on=["pair_row_low", "pair_row_high"], how="inner", validate="many_to_one"
    ).rename(columns={"mechanism_screen_v2": "mechanism_screen"})

    reused_keys = set(zip(reused["pair_row_low"], reused["pair_row_high"]))
    new_keys = list(zip(new["pair_row_low"], new["pair_row_high"]))
    missing_mask = [key not in reused_keys for key in new_keys]
    missing = new.loc[missing_mask].drop(columns=["pair_row_low", "pair_row_high"])
    reused.to_csv(args.output_dir / "paired_effects_reused.csv", index=False)
    missing.to_csv(args.output_dir / "missing_pairs_for_occlusion.csv", index=False)

    report = {
        "status": "v2_occlusion_reuse_prepared",
        "v2_supported_pairs": int(len(new)),
        "v2_supported_mechanism_counts": {
            str(key): int(value) for key, value in new["mechanism_screen"].value_counts().items()
        },
        "reused_unique_pairs": int(
            reused[["pair_row_low", "pair_row_high"]].drop_duplicates().shape[0]
        ),
        "reused_directed_effects": int(len(reused)),
        "missing_pairs_to_encode": int(len(missing)),
        "missing_mechanism_counts": {
            str(key): int(value) for key, value in missing["mechanism_screen"].value_counts().items()
        },
        "reuse_justification": (
            "The intervention indices are determined by shared-vs-unique peaks for the same "
            "spectrum pair. V2 changes residual-family membership and subtype thresholds only."
        ),
    }
    (args.output_dir / "reuse_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
