#!/usr/bin/env python3
"""Join processed quant/MS2 pairs to contrast and biological-unit readiness.

This is an outcome-blind admission ledger.  It never estimates an effect and
never upgrades repeated metadata labels to cross-study biological truth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def sample_stem(value: object) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    text = re.sub(r"\s+(?:filtered\s+)?Peak area$", "", text, flags=re.I)
    return Path(text).stem.casefold()


def quant_columns(path: Path) -> set[str]:
    columns = pd.read_csv(path, nrows=0).columns
    return {
        sample_stem(column)
        for column in columns
        if re.search(r"\.mz(?:ml|xml)\s+(?:filtered\s+)?Peak area$", str(column), flags=re.I)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--contrast-files", required=True, type=Path)
    parser.add_argument("--native-readiness", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str, keep_default_na=False)
    pairs = pairs.loc[pairs["schema_qualified"].str.lower().eq("true")].copy()
    files = pd.read_csv(args.contrast_files, sep="\t", dtype=str, keep_default_na=False)
    native = pd.read_csv(args.native_readiness, sep="\t", dtype=str, keep_default_na=False)
    native_by_contrast = native.set_index("contrast_id").to_dict("index")
    rows: list[dict[str, object]] = []

    for pair in pairs.to_dict("records"):
        stems = quant_columns(Path(pair["quant_local_path"]))
        accession = pair["dataset_accession"]
        block = files.loc[files["dataset_accession"].eq(accession)].copy()
        block["_stem"] = block["filename"].map(sample_stem)
        block = block.loc[block["_stem"].isin(stems)].copy()
        for contrast_id, panel in block.groupby("contrast_id", sort=True):
            panel = panel.loc[panel["reported_unique_subject_id"].ne("")].copy()
            reported_crossing = int(
                (panel.groupby("reported_unique_subject_id")["level"].nunique() > 1).sum()
            )
            reported_counts = panel.groupby("level")["reported_unique_subject_id"].nunique().to_dict()
            native_row = native_by_contrast.get(str(contrast_id))
            if native_row is None:
                unit_status = "reported_identifier_only_unadjudicated"
                n_units = sum(reported_counts.values())
                unit_counts = reported_counts
                crossing = reported_crossing
                source_family = f"accession_only_unadjudicated:{accession}"
                model_requirement = "unknown_until_design_reconstruction"
            else:
                native_ready = str(native_row["native_unit_map_ready"]).lower() == "true"
                unit_status = "native_unit_topology_complete" if native_ready else "native_unit_topology_failed"
                n_units = int(native_row["n_biological_units"])
                unit_counts = json.loads(native_row["units_per_level"])
                crossing = int(native_row["units_crossing_contrast_levels"])
                source_family = native_row["preliminary_source_family"]
                model_requirement = native_row["effect_model_requirement"]
            min_level_units = min(unit_counts.values()) if unit_counts else 0
            response_ready = unit_status == "native_unit_topology_complete" and crossing == 0 and min_level_units >= 5
            rows.append({
                "contrast_id": contrast_id,
                "dataset_accession": accession,
                "source_family": source_family,
                "comparison_stratum": panel["comparison_stratum"].iloc[0] if len(panel) else "",
                "contrast_field": panel["contrast_field"].iloc[0] if len(panel) else "",
                "joined_quantitative_file_appearances": len(panel),
                "reported_unit_identifiers": sum(reported_counts.values()),
                "reported_units_per_level": json.dumps(reported_counts, sort_keys=True),
                "reported_units_crossing_levels": reported_crossing,
                "unit_evidence_status": unit_status,
                "adjudicated_biological_units": n_units if unit_status.startswith("native_") else "",
                "adjudicated_units_per_level": json.dumps(unit_counts, sort_keys=True) if unit_status.startswith("native_") else "",
                "adjudicated_units_crossing_levels": crossing if unit_status.startswith("native_") else "",
                "effect_model_requirement": model_requirement,
                "ms2_linked_features": int(pair["shared_feature_ids"]),
                "response_topology_ready": response_ready,
                "cross_study_truth_status": "metadata_candidate_only_not_adjudicated_truth",
            })

    ledger = pd.DataFrame(rows)
    eligible = ledger.loc[ledger["response_topology_ready"]].copy()
    repeated = (
        eligible.groupby("comparison_stratum")
        .agg(candidate_panels=("contrast_id", "nunique"), candidate_sources=("source_family", "nunique"))
        .reset_index()
    ) if len(eligible) else pd.DataFrame(columns=["comparison_stratum", "candidate_panels", "candidate_sources"])
    repeated["has_at_least_two_candidate_sources"] = repeated["candidate_sources"] >= 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output_dir / "processed_contrast_readiness.tsv", sep="\t", index=False)
    repeated.to_csv(args.output_dir / "candidate_repeated_strata.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_processed_contrast_readiness_v1",
        "denominators": {
            "schema_qualified_processed_accessions": int(pairs["dataset_accession"].nunique()),
            "ms2_linked_features_across_best_pairs": int(pd.to_numeric(pairs["shared_feature_ids"]).sum()),
            "quantitative_sample_columns_across_best_pairs": int(pd.to_numeric(pairs["quant_sample_columns"]).sum()),
            "joined_contrast_panels": len(ledger),
            "panels_with_native_unit_topology": int(ledger["unit_evidence_status"].eq("native_unit_topology_complete").sum()),
            "panels_response_topology_ready": int(ledger["response_topology_ready"].sum()),
            "candidate_repeated_strata_with_two_sources": int(repeated["has_at_least_two_candidate_sources"].sum()),
        },
        "leakage_status": "schema_design_and_unit_topology_only_no_feature_effects_or_retrieval_scores_opened",
        "claim_boundary": [
            "Topology-ready is not analysis-ready; covariates, batch, contrast semantics, and source independence still require review.",
            "Repeated metadata strata are candidate positives, not biological truth.",
            "Reported identifiers never substitute for a native biological-unit hierarchy.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
