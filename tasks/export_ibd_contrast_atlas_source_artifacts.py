#!/usr/bin/env python3
"""Export complete MS2-linked IBD response artifacts for Contrast Atlas.

This is a thin, deterministic adapter over the already-consumed VEO-IBD and
IBD200 inputs.  It deliberately reuses the response estimation and MS2 parsing
logic from ``run_ibd_cross_study_contrast_query.py``.  It does not rebuild
features, download data, score cross-study relations, or read retrieval truth.

The registered cross-study relation is *near exact* at the condition level.
Its expected signed direction is unspecified because pediatric VEO-IBD and
adult CD/UC differ in age, subtype, and treatment exposure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_ibd_cross_study_contrast_query import (
    ADULT_CONTRASTS,
    MIN_BIO_PREVALENCE,
    MIN_BLANK_FOLD,
    library_hits,
    parse_mgf,
    robust_clr_by_sample,
    sha256_file,
    welch_effect,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/reference/contrast_atlas_ibd_cross_study_20261014"
DEFAULT_OUTPUT = ROOT / "data/validation/contrast_atlas_ibd_source_artifacts_20261019_v1"

PEDIATRIC_STUDY_ID = "MSV000097610"
PEDIATRIC_SOURCE_FAMILY = "MSV000097610_veo_ibd_case_control"
ADULT_STUDY_ID = "IBD200_GROUP"
ADULT_SOURCE_FAMILY = "IBD200_MSV000084775_MSV000084908_MSV000086509"
ADULT_ACCESSIONS = ("MSV000084775", "MSV000084908", "MSV000086509")
ADULT_RELATION_CONTRAST = "adult_all_ibd_vs_healthy"

ARTIFACT_COLUMNS = [
    "study_id",
    "source_family",
    "source_accessions",
    "contrast_id",
    "entity_id",
    "raw_feature_id",
    "ms2_join_id",
    "ms2_mgf_path",
    "ms2_precursor_mz",
    "quant_precursor_mz",
    "beta",
    "se",
    "n_positive",
    "n_negative",
    "identity_status",
    "identity_partition_contract",
    "effect_model",
]


def _finite_positive_se(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    return np.isfinite(beta) & np.isfinite(se) & (se > 0)


def validate_artifact(frame: pd.DataFrame, expected_study: str, contrasts: set[str]) -> None:
    missing = set(ARTIFACT_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"artifact missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError(f"empty artifact for {expected_study}")
    if set(frame["study_id"].astype(str)) != {expected_study}:
        raise RuntimeError(f"study drift in artifact for {expected_study}")
    if set(frame["contrast_id"].astype(str)) != contrasts:
        raise RuntimeError(f"contrast drift in artifact for {expected_study}")
    if frame.duplicated(["contrast_id", "entity_id"]).any():
        raise RuntimeError(f"duplicate contrast/entity rows for {expected_study}")
    numeric = frame[["ms2_precursor_mz", "beta", "se"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (frame["ms2_precursor_mz"] <= 0).any() or (frame["se"] <= 0).any():
        raise RuntimeError(f"non-finite response or invalid MS2 precursor for {expected_study}")
    if frame["ms2_join_id"].isna().any() or frame["raw_feature_id"].isna().any():
        raise RuntimeError(f"missing MS2 join for {expected_study}")
    if not set(frame["identity_status"]).issubset({"library_supported", "unresolved_entity"}):
        raise RuntimeError(f"invalid identity partition for {expected_study}")
    denominators = frame.groupby("contrast_id")[["n_positive", "n_negative"]].nunique()
    if (denominators != 1).any().any() or (frame[["n_positive", "n_negative"]] <= 0).any().any():
        raise RuntimeError(f"sample-denominator drift for {expected_study}")
    per_contrast_entities = frame.groupby("contrast_id")["entity_id"].apply(set)
    first = per_contrast_entities.iloc[0]
    if any(values != first for values in per_contrast_entities.iloc[1:]):
        raise RuntimeError(f"candidate entity universe changes across contrasts for {expected_study}")


def pediatric_artifact(source: Path) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    rclr_path = source / "pediatric_metabolomics_rclr.csv"
    metadata_path = source / "pediatric_metadata.csv"
    mgf_path = source / "pediatric_veoibd_consensus.mgf"
    annotations_path = source / "pediatric_paper_annotations.tsv"
    spectra = parse_mgf(mgf_path)
    hits = library_hits(annotations_path)

    table = pd.read_csv(rclr_path)
    metadata = pd.read_csv(metadata_path)
    table["SampleID"] = table["SampleID"].astype(str)
    metadata["sample_ID"] = metadata["sample_ID"].astype(str)
    table = table.merge(
        metadata[["sample_ID", "Cohort"]],
        left_on="SampleID",
        right_on="sample_ID",
        how="left",
        validate="one_to_one",
    )
    if table["Cohort"].isna().any():
        raise RuntimeError("pediatric response samples are not fully mapped")
    feature_columns = [column for column in table.columns if str(column).isdigit()]
    feature_ids = np.asarray([int(value) for value in feature_columns], dtype=int)
    positive = table["Cohort"].eq("VEOIBD").to_numpy()
    negative = table["Cohort"].eq("Healthy").to_numpy()
    beta, se = welch_effect(table[feature_columns].to_numpy(float).T, positive, negative)
    has_ms2 = np.asarray([int(value) in spectra for value in feature_ids], dtype=bool)
    finite = _finite_positive_se(beta, se)
    eligible = has_ms2 & finite

    rows: list[dict[str, Any]] = []
    mgf_relative = mgf_path.relative_to(ROOT).as_posix()
    for feature_id, effect, standard_error in zip(feature_ids[eligible], beta[eligible], se[eligible]):
        spectrum = spectra[int(feature_id)]
        supported = int(feature_id) in hits
        rows.append({
            "study_id": PEDIATRIC_STUDY_ID,
            "source_family": PEDIATRIC_SOURCE_FAMILY,
            "source_accessions": PEDIATRIC_STUDY_ID,
            "contrast_id": "pediatric_veoibd_vs_healthy",
            "entity_id": f"{PEDIATRIC_STUDY_ID}:feature:{int(feature_id)}",
            "raw_feature_id": str(int(feature_id)),
            "ms2_join_id": str(int(feature_id)),
            "ms2_mgf_path": mgf_relative,
            "ms2_precursor_mz": float(spectrum.precursor_mz),
            "quant_precursor_mz": np.nan,
            "beta": float(effect),
            "se": float(standard_error),
            "n_positive": int(positive.sum()),
            "n_negative": int(negative.sum()),
            "identity_status": "library_supported" if supported else "unresolved_entity",
            "identity_partition_contract": "legacy_ibd_local_library_hit_wse_ge_0.70_shared_peaks_ge_6",
            "effect_model": "Welch difference of author rCLR feature values",
        })
    artifact = pd.DataFrame(rows, columns=ARTIFACT_COLUMNS)
    audit = {
        "samples_after_author_qc": int(len(table)),
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
        "total_response_features": int(len(feature_ids)),
        "nonempty_mgf_spectra": int(len(spectra)),
        "response_features_with_ms2": int(has_ms2.sum()),
        "response_features_with_finite_beta_se": int(finite.sum()),
        "eligible_ms2_finite_beta_se": int(eligible.sum()),
        "excluded_without_ms2": int((~has_ms2).sum()),
        "excluded_invalid_beta_se_among_ms2": int((has_ms2 & ~finite).sum()),
        "eligible_library_supported": int(sum(int(value) in hits for value in feature_ids[eligible])),
        "eligible_unresolved": int(sum(int(value) not in hits for value in feature_ids[eligible])),
    }
    return artifact, audit, [rclr_path, metadata_path, mgf_path, annotations_path]


def _adult_sample_map(quant: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    metadata = metadata[
        metadata["Diagnosis"].isin(["CD", "UC", "HC"])
        & metadata["Metabolomics-FileName-Run1"].ne("not applicable")
    ].copy()
    metadata["quant_stem"] = metadata["Metabolomics-FileName-Run1"].str.replace(
        r"\.(mzML|mzXML)$", "", regex=True
    )
    sample_columns = [column for column in quant.columns if column.endswith(" Peak area")]
    stem_to_column = {
        column.removesuffix(" Peak area").removesuffix(".mzML").removesuffix(".mzXML"): column
        for column in sample_columns
    }
    metadata = metadata[metadata["quant_stem"].isin(stem_to_column)].copy()
    metadata["quant_column"] = metadata["quant_stem"].map(stem_to_column)
    metadata = metadata.drop_duplicates("quant_column", keep="first").reset_index(drop=True)
    biological = metadata["quant_column"].tolist()
    blanks = [column for column in sample_columns if "blank" in column.lower() or "blk_" in column.lower()]
    return metadata, biological, blanks


def adult_artifact(source: Path) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    quant_path = source / "adult_ibd200_gnps_quant.csv"
    metadata_path = source / "adult_ibd200_metadata.tsv"
    mgf_path = source / "adult_ibd200_consensus.mgf"
    annotations_path = source / "adult_ibd200_annotations.tsv"
    spectra = parse_mgf(mgf_path)
    hits = library_hits(annotations_path)
    quant = pd.read_csv(quant_path)
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    metadata, biological_columns, blank_columns = _adult_sample_map(quant, metadata)

    raw_bio = quant[biological_columns].to_numpy(float)
    raw_blank = quant[blank_columns].to_numpy(float) if blank_columns else np.zeros((len(quant), 0))
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
    eligible_mz = quant_mz[qc]
    mgf_relative = mgf_path.relative_to(ROOT).as_posix()
    rows: list[dict[str, Any]] = []
    contrast_audit: dict[str, Any] = {}
    export_contrasts = [
        contrast for contrast in ADULT_CONTRASTS if contrast.name == ADULT_RELATION_CONTRAST
    ]
    if len(export_contrasts) != 1:
        raise RuntimeError("adult near-exact relation contrast registry drift")
    for contrast in export_contrasts:
        positive = contrast.positive(metadata).to_numpy()
        negative = contrast.negative(metadata).to_numpy()
        beta, se = welch_effect(values, positive, negative)
        finite = _finite_positive_se(beta, se)
        if not finite.all():
            raise RuntimeError(f"adult contrast has invalid beta/SE: {contrast.name}")
        contrast_audit[contrast.name] = {
            "n_positive": int(positive.sum()),
            "n_negative": int(negative.sum()),
            "eligible_entities": int(len(eligible_ids)),
        }
        for feature_id, q_mz, effect, standard_error in zip(eligible_ids, eligible_mz, beta, se):
            spectrum = spectra[int(feature_id)]
            supported = int(feature_id) in hits
            rows.append({
                "study_id": ADULT_STUDY_ID,
                "source_family": ADULT_SOURCE_FAMILY,
                "source_accessions": ";".join(ADULT_ACCESSIONS),
                "contrast_id": contrast.name,
                "entity_id": f"{ADULT_STUDY_ID}:feature:{int(feature_id)}",
                "raw_feature_id": str(int(feature_id)),
                "ms2_join_id": str(int(feature_id)),
                "ms2_mgf_path": mgf_relative,
                "ms2_precursor_mz": float(spectrum.precursor_mz),
                "quant_precursor_mz": float(q_mz),
                "beta": float(effect),
                "se": float(standard_error),
                "n_positive": int(positive.sum()),
                "n_negative": int(negative.sum()),
                "identity_status": "library_supported" if supported else "unresolved_entity",
                "identity_partition_contract": "legacy_ibd_local_library_hit_wse_ge_0.70_shared_peaks_ge_6",
                "effect_model": "Welch difference after robust CLR per biological sample",
            })
    artifact = pd.DataFrame(rows, columns=ARTIFACT_COLUMNS)
    eligible_known = int(sum(int(value) in hits for value in eligible_ids))
    audit = {
        "source_accessions_merged_as_one_family": list(ADULT_ACCESSIONS),
        "mapped_biological_samples": int(len(metadata)),
        "diagnosis_counts": {str(key): int(value) for key, value in metadata["Diagnosis"].value_counts().items()},
        "total_quant_features": int(len(quant)),
        "nonempty_mgf_spectra": int(len(spectra)),
        "qc_ms2_entities": int(len(eligible_ids)),
        "eligible_library_supported": eligible_known,
        "eligible_unresolved": int(len(eligible_ids) - eligible_known),
        "contrasts": contrast_audit,
    }
    return artifact, audit, [quant_path, metadata_path, mgf_path, annotations_path]


def run(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    pediatric, pediatric_audit, pediatric_inputs = pediatric_artifact(source)
    adult, adult_audit, adult_inputs = adult_artifact(source)
    validate_artifact(pediatric, PEDIATRIC_STUDY_ID, {"pediatric_veoibd_vs_healthy"})
    validate_artifact(adult, ADULT_STUDY_ID, {ADULT_RELATION_CONTRAST})
    if PEDIATRIC_SOURCE_FAMILY == ADULT_SOURCE_FAMILY:
        raise RuntimeError("source families are not independent")

    output.mkdir(parents=True)
    pediatric_path = output / "msv000097610_veoibd_response_artifact.tsv.gz"
    adult_path = output / "ibd200_merged_source_response_artifact.tsv.gz"
    relation_path = output / "relation_registry.json"
    pediatric.to_csv(pediatric_path, sep="\t", index=False, compression="gzip")
    adult.to_csv(adult_path, sep="\t", index=False, compression="gzip")
    relation = {
        "schema": "contrast_atlas_prespecified_relation_v1",
        "relation_class": "prespecified_near_exact_condition_family",
        "left": {"study_id": PEDIATRIC_STUDY_ID, "contrast_id": "pediatric_veoibd_vs_healthy"},
        "right": {"study_id": ADULT_STUDY_ID, "contrast_id": "adult_all_ibd_vs_healthy"},
        "expected_signed_direction": "unspecified",
        "truth_status": "development_consumed_relevance_not_signed_truth",
        "independence": {
            "independent_source_families": True,
            "adult_accessions_count_as_one_family": list(ADULT_ACCESSIONS),
            "shared_subject_or_sample_ids_observed": 0,
        },
        "boundary": (
            "Same species, fecal biospecimen and IBD-versus-healthy condition family, but pediatric "
            "VEO-IBD and adult CD/UC differ in age, subtype and treatment exposure. Unregistered "
            "relations are unadjudicated, not negatives."
        ),
    }
    relation_path.write_text(json.dumps(relation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    input_paths = pediatric_inputs + adult_inputs
    report = {
        "schema": "contrast_atlas_ibd_complete_source_artifacts_v1",
        "status": "DEVELOPMENT_CONSUMED_IBD_SOURCE_ARTIFACTS_COMPLETE",
        "feature_extraction_recomputed": False,
        "data_downloaded": False,
        "cross_study_scores_computed": False,
        "signed_truth_claimed": False,
        "source_family_count": 2,
        "relation_count": 1,
        "pediatric": pediatric_audit,
        "adult": adult_audit,
        "outputs": {
            pediatric_path.name: {"rows": int(len(pediatric)), "sha256": sha256_file(pediatric_path)},
            adult_path.name: {"rows": int(len(adult)), "sha256": sha256_file(adult_path)},
            relation_path.name: {"sha256": sha256_file(relation_path)},
        },
        "input_provenance": {
            str(path.relative_to(ROOT).as_posix()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        },
        "identity_boundary": (
            "library_supported versus unresolved_entity uses the legacy IBD local library-hit rule. "
            "A downstream collection must reapply its frozen global identity partition before a "
            "cross-source unknown-information claim."
        ),
        "entity_boundary": "Rows are feature entities, not ion-family-collapsed unique analytes.",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
