"""Check replicated peak-evidence directions across instruments and CE bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def query_difference(hits: pd.DataFrame, feature_id: str) -> pd.DataFrame:
    subset = hits.loc[hits["feature_id"] == feature_id].drop_duplicates(
        ["split", "query_index", "evidence"]
    )
    subset["hit"] = 1
    pivot = subset.pivot_table(
        index=["split", "query_index"], columns="evidence", values="hit",
        aggfunc="max", fill_value=0,
    ).reset_index()
    for column in ("identity", "confounder"):
        if column not in pivot:
            pivot[column] = 0
    pivot["difference"] = pivot["identity"] - pivot["confounder"]
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-dir", type=Path, default=Path("data/validation/large_failure_peak_chemical_attribution_v2"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_failure_peak_evidence_strata"))
    parser.add_argument("--min-queries", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(args.attribution_dir / "peak_feature_hits.csv")
    panel = pd.read_csv(args.attribution_dir / "replicated_evidence_panel.csv")
    manifests = []
    for split in ("discovery", "confirmation"):
        manifest = pd.read_csv(args.embedding_root / f"large_observability_embeddings_{split}" / "manifest.csv")
        manifest["split"] = split
        manifest["query_index"] = manifest.index
        manifests.append(manifest[[
            "split", "query_index", "instrument", "collision_energy", "ring_class", "precursor_mz"
        ]])
    metadata = pd.concat(manifests, ignore_index=True)
    metadata["collision_energy_bin"] = pd.cut(
        metadata["collision_energy"], [-1e-9, 20, 40, 70, float("inf")],
        labels=["CE<=20", "CE20-40", "CE40-70", "CE>70"], include_lowest=True,
    ).astype("object").fillna("CE_missing")
    metadata["precursor_mass_bin"] = pd.cut(
        metadata["precursor_mz"], [0, 200, 350, 500, float("inf")],
        labels=["mz<=200", "mz200-350", "mz350-500", "mz>500"], include_lowest=True,
    ).astype(str)

    rows = []
    for feature_id in panel["feature_id"]:
        values = query_difference(hits, feature_id).merge(
            metadata, on=["split", "query_index"], how="left", validate="one_to_one"
        )
        for dimension in ("instrument", "collision_energy_bin", "ring_class", "precursor_mass_bin"):
            for (split, stratum), group in values.groupby(["split", dimension], dropna=False):
                rows.append({
                    "feature_id": feature_id, "dimension": dimension,
                    "split": split, "stratum": str(stratum), "queries": len(group),
                    "identity_minus_confounder": float(group["difference"].mean()),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "panel_stratum_effects.csv", index=False)
    eligible = frame.loc[frame["queries"] >= args.min_queries].copy()
    stability = eligible.groupby(["feature_id", "dimension"]).agg(
        eligible_strata=("stratum", "size"),
        negative_strata=("identity_minus_confounder", lambda x: int((x < 0).sum())),
        min_effect=("identity_minus_confounder", "min"),
        max_effect=("identity_minus_confounder", "max"),
    ).reset_index()
    stability["all_eligible_strata_confounder_enriched"] = (
        stability["eligible_strata"] == stability["negative_strata"]
    )
    stability.to_csv(args.output_dir / "panel_stability_summary.csv", index=False)
    overall = stability.groupby("feature_id").agg(
        dimensions_checked=("dimension", "size"),
        dimensions_fully_stable=("all_eligible_strata_confounder_enriched", "sum"),
        eligible_strata=("eligible_strata", "sum"),
        negative_strata=("negative_strata", "sum"),
    ).reset_index()
    overall["all_checked_strata_confounder_enriched"] = overall["eligible_strata"] == overall["negative_strata"]
    overall.to_csv(args.output_dir / "panel_overall_stability.csv", index=False)
    frozen = panel.merge(
        overall[["feature_id", "all_checked_strata_confounder_enriched"]],
        on="feature_id", how="inner", validate="one_to_one",
    )
    frozen = frozen.loc[frozen["all_checked_strata_confounder_enriched"]].copy()
    frozen["frozen_expected_direction"] = "confounder_enriched"
    frozen["frozen_before_test"] = True
    frozen.to_csv(args.output_dir / "frozen_test_panel.csv", index=False)
    report = {
        "panel_features": len(panel), "minimum_queries_per_stratum": args.min_queries,
        "features_confounder_enriched_in_all_eligible_strata": int(
            overall["all_checked_strata_confounder_enriched"].sum()
        ),
        "features_with_any_direction_reversal": int(
            (~overall["all_checked_strata_confounder_enriched"]).sum()
        ),
        "frozen_test_panel_size": len(frozen),
        "dimensions": ["instrument", "collision_energy_bin", "ring_class", "precursor_mass_bin"],
        "claim_limit": "descriptive stability audit; sparse strata are excluded by the prespecified minimum",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
