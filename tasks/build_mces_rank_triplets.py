"""Build leakage-safe local MCES ranking triplets from the semantic cache.

Only exact MCES distances can define a local positive.  A negative must either
have an exact distance separated by ``min_exact_gap`` or be independently
marked as proven beyond the MCES threshold.  Bound-only values are never used
as continuous regression targets.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data/e2/mces_local_rank"


def directed_edges(frame: pd.DataFrame) -> pd.DataFrame:
    forward = frame.rename(columns={
        "ik14_a": "anchor_ik14", "ik14_b": "neighbor_ik14",
        "representative_spectrum_a": "anchor_spectrum",
        "representative_spectrum_b": "neighbor_spectrum",
        "formula_a": "anchor_formula", "formula_b": "neighbor_formula",
    })
    reverse = frame.rename(columns={
        "ik14_b": "anchor_ik14", "ik14_a": "neighbor_ik14",
        "representative_spectrum_b": "anchor_spectrum",
        "representative_spectrum_a": "neighbor_spectrum",
        "formula_b": "anchor_formula", "formula_a": "neighbor_formula",
    })
    columns = [
        "anchor_ik14", "neighbor_ik14", "anchor_spectrum", "neighbor_spectrum",
        "anchor_formula", "neighbor_formula", "same_formula", "precursor_ppm",
        "distance", "distance_kind", "usable_exact_local", "usable_proven_far",
    ]
    return pd.concat([forward[columns], reverse[columns]], ignore_index=True)


def build_split(
    manifest: pd.DataFrame,
    cache: pd.DataFrame,
    local_max: float,
    min_exact_gap: float,
) -> tuple[pd.DataFrame, dict]:
    merged = manifest.merge(
        cache[[
            "pair_key", "distance", "distance_kind", "usable_exact_local",
            "usable_proven_far", "status",
        ]],
        on="pair_key", how="left", validate="one_to_one",
    )
    computed = merged.loc[merged["status"].eq("ok")].copy()
    edges = directed_edges(computed)
    rows = []
    for anchor, group in edges.groupby("anchor_ik14", sort=True):
        positives = group.loc[
            group["usable_exact_local"].eq(1) & group["distance"].le(local_max)
        ].sort_values(["distance", "precursor_ppm", "neighbor_ik14"])
        if positives.empty:
            continue
        positive = positives.iloc[0]
        exact_far = group.loc[
            group["distance_kind"].eq("exact")
            & group["distance"].ge(float(positive["distance"]) + min_exact_gap)
        ]
        proven_far = group.loc[group["usable_proven_far"].eq(1)]
        negatives = pd.concat([exact_far, proven_far], ignore_index=True).drop_duplicates(
            subset=["neighbor_ik14"]
        )
        if negatives.empty:
            continue
        # Same-formula and closest-precursor negatives are chemically hardest.
        negatives = negatives.assign(
            same_formula_priority=negatives["same_formula"].astype(int)
        ).sort_values(
            ["same_formula_priority", "precursor_ppm", "neighbor_ik14"],
            ascending=[False, True, True],
        )
        negative = negatives.iloc[0]
        rows.append({
            "anchor_ik14": anchor,
            "positive_ik14": positive["neighbor_ik14"],
            "negative_ik14": negative["neighbor_ik14"],
            "anchor_spectrum": int(positive["anchor_spectrum"]),
            "positive_spectrum": int(positive["neighbor_spectrum"]),
            "negative_spectrum": int(negative["neighbor_spectrum"]),
            "positive_mces": float(positive["distance"]),
            "negative_mces": (
                float(negative["distance"])
                if negative["distance_kind"] == "exact" else np.nan
            ),
            "negative_distance_kind": negative["distance_kind"],
            "positive_same_formula": bool(positive["same_formula"]),
            "negative_same_formula": bool(negative["same_formula"]),
            "positive_precursor_ppm": float(positive["precursor_ppm"]),
            "negative_precursor_ppm": float(negative["precursor_ppm"]),
        })
    triplets = pd.DataFrame(rows)
    report = {
        "manifest_pairs": int(len(manifest)),
        "computed_pairs": int(len(computed)),
        "exact_local_pairs_at_or_below_cutoff": int(
            (computed["usable_exact_local"].eq(1) & computed["distance"].le(local_max)).sum()
        ),
        "proven_far_pairs": int(computed["usable_proven_far"].eq(1).sum()),
        "rank_triplets": int(len(triplets)),
        "unique_anchors": int(triplets["anchor_ik14"].nunique()) if len(triplets) else 0,
        "same_formula_positive_fraction": (
            float(triplets["positive_same_formula"].mean()) if len(triplets) else None
        ),
        "same_formula_negative_fraction": (
            float(triplets["negative_same_formula"].mean()) if len(triplets) else None
        ),
    }
    return triplets, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_DIR / "mces_cache.sqlite")
    parser.add_argument("--local-max", type=float, default=8.0)
    parser.add_argument("--min-exact-gap", type=float, default=2.0)
    args = parser.parse_args()
    with sqlite3.connect(args.cache) as connection:
        cache = pd.read_sql_query("SELECT * FROM mces_pair", connection)
    reports = {
        "local_positive_definition": f"exact MCES <= {args.local_max:g}",
        "negative_definition": (
            f"exact MCES at least {args.min_exact_gap:g} farther, or proven above threshold"
        ),
        "bound_values_used_as_regression_targets": False,
        "splits": {},
    }
    for split in ("train", "val"):
        manifest = pd.read_csv(args.manifest_dir / f"{split}_unique_molecule_pairs.csv")
        manifest["pair_key"] = manifest["ik14_a"].astype(str) + "|" + manifest["ik14_b"].astype(str)
        triplets, split_report = build_split(
            manifest, cache.loc[cache["split"].eq(split)].copy(),
            args.local_max, args.min_exact_gap,
        )
        triplets.to_csv(args.manifest_dir / f"{split}_mces_rank_triplets.csv", index=False)
        reports["splits"][split] = split_report
    (args.manifest_dir / "mces_rank_triplet_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
