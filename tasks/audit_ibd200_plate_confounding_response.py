#!/usr/bin/env python3
"""Re-estimate the frozen IBD200 response within the only plate containing controls.

All healthy controls in the consumed IBD200 table occur on Plate3.  The primary
158-vs-19 response is therefore diagnosis/plate confounded.  This audit keeps the
same feature QC, spectra and identity partition, but estimates CD/UC-vs-HC using
Plate3 only (36 vs 19) and writes a drop-in seven-source sensitivity collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_ibd_contrast_atlas_source_artifacts import (
    MIN_BIO_PREVALENCE,
    MIN_BLANK_FOLD,
    _adult_sample_map,
    parse_mgf,
    robust_clr_by_sample,
    welch_effect,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("data/reference/contrast_atlas_ibd_cross_study_20261014"),
    )
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v3_ibd_20261019"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_ibd200_plate3_v2_20261020"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    quant_path = args.source_dir / "adult_ibd200_gnps_quant.csv"
    metadata_path = args.source_dir / "adult_ibd200_metadata.tsv"
    mgf_path = args.source_dir / "adult_ibd200_consensus.mgf"
    quant = pd.read_csv(quant_path)
    metadata_raw = pd.read_csv(metadata_path, sep="\t", dtype=str)
    metadata, biological_columns, blank_columns = _adult_sample_map(quant, metadata_raw)

    diagnosis_by_plate = pd.crosstab(metadata["Metabolomics-Plate"], metadata["Diagnosis"])
    if int(diagnosis_by_plate.drop(index="Plate3", errors="ignore").get("HC", pd.Series()).sum()) != 0:
        raise ValueError("healthy controls unexpectedly occur outside Plate3")

    spectra = parse_mgf(mgf_path)
    raw_bio = quant[biological_columns].to_numpy(float)
    raw_blank = (
        quant[blank_columns].to_numpy(float)
        if blank_columns else np.zeros((len(quant), 0), dtype=float)
    )
    prevalence = np.mean(raw_bio > 0, axis=1)
    mean_bio = np.mean(raw_bio, axis=1)
    mean_blank = np.mean(raw_blank, axis=1) if raw_blank.shape[1] else np.zeros(len(quant))
    feature_ids = quant["row ID"].astype(int).to_numpy()
    quant_mz = quant["row m/z"].astype(float).to_numpy()
    qc = (
        np.isfinite(quant_mz)
        & (prevalence >= MIN_BIO_PREVALENCE)
        & (mean_bio > 0)
        & (mean_bio >= MIN_BLANK_FOLD * mean_blank)
        & np.asarray([int(value) in spectra for value in feature_ids])
    )
    values = robust_clr_by_sample(raw_bio[qc])
    eligible_ids = feature_ids[qc]
    plate3 = metadata["Metabolomics-Plate"].eq("Plate3").to_numpy()
    positive = plate3 & metadata["Diagnosis"].isin(["CD", "UC"]).to_numpy()
    negative = plate3 & metadata["Diagnosis"].eq("HC").to_numpy()
    if int(positive.sum()) != 36 or int(negative.sum()) != 19:
        raise ValueError("Plate3 case/control denominator drift")
    beta, se = welch_effect(values, positive, negative)
    estimable = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    if int(estimable.sum()) == 0:
        raise ValueError("no estimable Plate3 effects")

    response_path = args.collection / "response_entities.tsv.gz"
    response = pd.read_csv(response_path, sep="\t", dtype={"raw_entity_id": str})
    adult = response["study_id"].astype(str).eq("IBD200_GROUP")
    if int(adult.sum()) != len(eligible_ids):
        raise ValueError("frozen adult response and recomputed QC feature counts differ")
    replacement = pd.DataFrame(
        {
            "raw_entity_id": eligible_ids[estimable].astype(str),
            "plate3_beta": beta[estimable],
            "plate3_se": se[estimable],
        }
    )
    adult_rows = response.loc[adult].merge(
        replacement, on="raw_entity_id", how="inner", validate="one_to_one"
    )
    if len(adult_rows) != int(estimable.sum()) or adult_rows[["plate3_beta", "plate3_se"]].isna().any().any():
        raise ValueError("Plate3 effect join incomplete")

    original_beta = adult_rows["beta"].to_numpy(float)
    original_se = adult_rows["se"].to_numpy(float)
    plate3_beta = adult_rows["plate3_beta"].to_numpy(float)
    plate3_se = adult_rows["plate3_se"].to_numpy(float)
    original_weight = original_beta / np.sqrt(original_beta**2 + original_se**2)
    plate3_weight = plate3_beta / np.sqrt(plate3_beta**2 + plate3_se**2)

    adult_rows["beta"] = adult_rows.pop("plate3_beta")
    adult_rows["se"] = adult_rows.pop("plate3_se")
    adult_rows["n_positive"] = int(positive.sum())
    adult_rows["n_negative"] = int(negative.sum())
    adult_rows["effect_model"] = (
        "Plate3-restricted Welch difference after robust CLR; diagnosis/plate sensitivity"
    )
    updated = pd.concat([response.loc[~adult], adult_rows[response.columns]], ignore_index=True)
    if len(response) - len(updated) != int((~estimable).sum()):
        raise ValueError("unexpected response row-count change")

    args.output_dir.mkdir(parents=True)
    updated_path = args.output_dir / "response_entities.tsv.gz"
    updated.to_csv(updated_path, sep="\t", index=False, compression="gzip")
    contrast_manifest = pd.read_csv(args.collection / "contrast_manifest.tsv", sep="\t")
    contrast_row = contrast_manifest["study_id"].astype(str).eq("IBD200_GROUP")
    adult_known = int(adult_rows["identity_status"].eq("library_supported").sum())
    adult_unresolved = int(adult_rows["identity_status"].eq("unresolved_entity").sum())
    contrast_manifest.loc[contrast_row, "independent_units"] = int(positive.sum() + negative.sum())
    contrast_manifest.loc[contrast_row, "eligible_response_entities"] = len(adult_rows)
    contrast_manifest.loc[contrast_row, "library_supported_response_entities"] = adult_known
    contrast_manifest.loc[contrast_row, "unresolved_response_entities"] = adult_unresolved
    contrast_path = args.output_dir / "contrast_manifest.tsv"
    contrast_manifest.to_csv(contrast_path, sep="\t", index=False)

    source_manifest = pd.read_csv(args.collection / "source_manifest.tsv", sep="\t")
    source_row = source_manifest["study_id"].astype(str).eq("IBD200_GROUP")
    source_manifest.loc[source_row, "biological_units"] = int(positive.sum() + negative.sum())
    source_manifest.loc[source_row, "response_eligible_entity_union"] = len(adult_rows)
    source_manifest.loc[source_row, "response_eligible_library_supported_union"] = adult_known
    source_manifest.loc[source_row, "response_eligible_unresolved_union"] = adult_unresolved
    source_path = args.output_dir / "source_manifest.tsv"
    source_manifest.to_csv(source_path, sep="\t", index=False)

    relation_path = args.output_dir / "registered_relations.tsv"
    relations = pd.read_csv(args.collection / "registered_relations.tsv", sep="\t")
    relations["signed_truth_claimed"] = relations["signed_truth_claimed"].map(
        lambda value: False if pd.isna(value) else bool(value)
    )
    relations.to_csv(relation_path, sep="\t", index=False)
    copied = [contrast_path, source_path, relation_path]

    report = {
        "schema": "contrast_atlas_ibd200_plate3_response_sensitivity_v1",
        "status": "DEVELOPMENT_CONSUMED_PLATE_CONFOUNDING_SENSITIVITY_COMPLETE",
        "confounding_audit": {
            "diagnosis_by_plate": {
                str(index): {str(k): int(v) for k, v in row.items()}
                for index, row in diagnosis_by_plate.to_dict("index").items()
            },
            "all_healthy_controls_on_plate3": True,
            "primary_effect_denominators": {"ibd": 158, "healthy": 19},
            "plate3_effect_denominators": {"ibd": int(positive.sum()), "healthy": int(negative.sum())},
            "fixed_qc_ms2_entities": int(len(eligible_ids)),
            "plate3_response_estimable_entities": int(estimable.sum()),
            "plate3_response_unestimable_abstained": int((~estimable).sum()),
            "plate3_library_supported_entities": adult_known,
            "plate3_unresolved_entities": adult_unresolved,
        },
        "effect_change": {
            "pearson_beta": float(np.corrcoef(original_beta, plate3_beta)[0, 1]),
            "pearson_uncertainty_shrunk_weight": float(
                np.corrcoef(original_weight, plate3_weight)[0, 1]
            ),
            "sign_concordance": float(np.mean(np.sign(original_beta) == np.sign(plate3_beta))),
            "median_absolute_beta_change": float(np.median(np.abs(original_beta - plate3_beta))),
        },
        "claim_boundary": [
            "Plate3 restriction removes perfect healthy-control/plate confounding but is not an independent cohort.",
            "The 746 QC+MS2 candidate boundary and frozen identity labels are retained; 84 entities with unestimable Plate3 response are explicitly excluded.",
            "This sensitivity can invalidate a biological interpretation but cannot externally confirm one.",
            "All inputs are development/consumed.",
        ],
        "input_sha256": {
            str(path): sha256(path)
            for path in (quant_path, metadata_path, mgf_path, response_path)
        },
        "output_sha256": {
            updated_path.name: sha256(updated_path),
            **{path.name: sha256(path) for path in copied},
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
