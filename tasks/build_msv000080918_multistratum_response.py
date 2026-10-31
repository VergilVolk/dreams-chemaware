#!/usr/bin/env python3
"""Build cage-clustered multistratum response objects for MSV000080918.

The public abundance matrix is joined to GNPS consensus spectra only through
three observable fields: precursor m/z, retention time, and at least one source
sample in which the abundance feature is non-zero.  Biological effects are
estimated from cage-by-arm cells with cage-clustered standard errors;
individual mice sharing a cage are never treated as independent replicates.
This also handles cages containing mice from both arms without averaging the
two arms together.

This is a development/consumed response source.  In particular, X-Align used
the B6 6-month subset.  That subset remains in the panel ledger but is not a
primary response because its HFD and NC age ranges do not overlap.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


META_COLUMNS = [
    "#OTU ID",
    "Sample MS name",
    "cage",
    "DESCRIPTION",
    "COLLECTION_TIMESTAMP",
    "age weeks",
    "host_subject_id",
    "Strain",
    "diet",
    "Antibiotics",
    "phenotype ",
    "timepoint",
    "immune_status",
]

CONTRAST_SPECS = (
    {
        "contrast_id": "b6_6m_hfd_vs_nc_xalign_consumed",
        "control": "B6-NC (6M)",
        "treatment": "B6-HFD (6M)",
        "contrast_family": "high_fat_diet",
        "analysis_role": "direct_competitor_only",
        "minimum_cages": 2,
        "minimum_unique_cages": 4,
        "minimum_shared_dates": 1,
        "require_within_date_age_range_overlap": False,
    },
    {
        "contrast_id": "b6_10_11m_hfd_vs_nc_date_matched",
        "control": "B6-NC (10-11M)",
        "treatment": "B6-HFD (10-11M)",
        "contrast_family": "high_fat_diet",
        "analysis_role": "exploratory_development",
        "minimum_cages": 3,
        "minimum_unique_cages": 6,
        "minimum_shared_dates": 2,
        "require_within_date_age_range_overlap": False,
    },
    {
        "contrast_id": "mup_hfd_10_11m_iga_ko_vs_intact_date_matched",
        "control": "MupHe-HFD (10-11M)",
        "treatment": "IgAkMup-HFD (10-11M)",
        "contrast_family": "iga_genetic_perturbation",
        "analysis_role": "primary_development_positive_control",
        "minimum_cages": 3,
        "minimum_unique_cages": 6,
        "minimum_shared_dates": 2,
        "require_within_date_age_range_overlap": True,
    },
    {
        "contrast_id": "mup_hfd_10_11m_cd8_ko_vs_intact_date_matched",
        "control": "MupHe-HFD (10-11M)",
        "treatment": "CD8kMup-HFD (10-11M)",
        "contrast_family": "cd8_genetic_perturbation",
        "analysis_role": "sensitivity_development_positive_control",
        "minimum_cages": 3,
        "minimum_unique_cages": 6,
        "minimum_shared_dates": 2,
        "require_within_date_age_range_overlap": False,
    },
    {
        "contrast_id": "cd8ko_mup_hfd_10_11m_antibiotic_vs_no_date_matched",
        "control": "CD8kMup-HFD (10-11M)",
        "treatment": "Antibiotic-CD8kMup-HFD (10-11M)",
        "contrast_family": "antibiotic_perturbation",
        "analysis_role": "candidate_failed_design",
        "minimum_cages": 3,
        "minimum_unique_cages": 6,
        "minimum_shared_dates": 2,
        "require_within_date_age_range_overlap": False,
    },
    {
        "contrast_id": "mup_hfd_10_11m_oxaliplatin_vs_untreated_date_matched",
        "control": "MupHe-HFD (10-11M)",
        "treatment": "MupHe-HFD-Oxali (10-11M)",
        "contrast_family": "oxaliplatin_treatment",
        "analysis_role": "excluded_unverified_treatment_provenance",
        "minimum_cages": 3,
        "minimum_unique_cages": 6,
        "minimum_shared_dates": 2,
        "require_within_date_age_range_overlap": False,
    },
    {
        "contrast_id": "mup_hfd_10_11m_oxaliplatin_pdl1_vs_untreated_date_matched",
        "control": "MupHe-HFD (10-11M)",
        "treatment": "MupHe-HFD-Oxali+aPD-L1 (10-11M)",
        "contrast_family": "combination_immunotherapy",
        "analysis_role": "excluded_unverified_treatment_provenance",
        "minimum_cages": 3,
        "minimum_unique_cages": 6,
        "minimum_shared_dates": 2,
        "require_within_date_age_range_overlap": False,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abundance", type=Path, required=True)
    parser.add_argument("--cluster-members", type=Path, required=True)
    parser.add_argument("--cluster-summary", type=Path, required=True)
    parser.add_argument("--params-xml", type=Path, required=True)
    parser.add_argument("--library-clusters", type=Path, required=True)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-ms2-peaks", type=int, default=3)
    parser.add_argument("--minimum-detected-cages", type=int, default=3)
    parser.add_argument("--minimum-pooled-cage-prevalence", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    observed = values[finite]
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[finite] = restored
    return out


def parse_feature_name(value: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([+-]?[0-9.]+)_\s*([+-]?[0-9.]+)\s*", str(value))
    if match is None:
        raise ValueError(f"unparseable feature column: {value!r}")
    return float(match.group(1)), float(match.group(2))


def parse_mgf_scans(path: Path) -> dict[int, int]:
    peak_counts: dict[int, int] = {}
    current_scan: int | None = None
    current_peaks = 0
    inside = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "BEGIN IONS":
                inside = True
                current_scan = None
                current_peaks = 0
            elif inside and stripped.startswith("SCANS="):
                current_scan = int(stripped.split("=", 1)[1])
            elif inside and stripped == "END IONS":
                if current_scan is None:
                    raise ValueError("MGF block lacks SCANS")
                if current_scan in peak_counts:
                    raise ValueError(f"duplicate MGF SCANS={current_scan}")
                peak_counts[current_scan] = current_peaks
                inside = False
            elif inside and stripped and "=" not in stripped:
                fields = stripped.split()
                if len(fields) >= 2:
                    try:
                        float(fields[0])
                        float(fields[1])
                    except ValueError:
                        continue
                    current_peaks += 1
    if not peak_counts:
        raise ValueError(f"no SCANS entries found in {path}")
    return peak_counts


def sample_ids_from_allfiles(value: object) -> frozenset[str]:
    if pd.isna(value):
        return frozenset()
    found: set[str] = set()
    for token in str(value).split("###"):
        filename = token.split(":", 1)[0]
        filename = Path(filename).name
        filename = re.sub(r"\.(?:mzXML|mzML)$", "", filename, flags=re.I)
        if filename:
            found.add(filename)
    return frozenset(found)


def normalize_library_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "n/a", "na", "no match"}:
        return ""
    return text


def build_full_cluster_table(
    members_path: Path,
    summary_path: Path,
    params_path: Path,
    library_path: Path,
) -> pd.DataFrame:
    root = ET.parse(params_path).getroot()
    file_mapping: dict[str, str] = {}
    for element in root.findall(".//parameter[@name='upload_file_mapping']"):
        text = element.text or ""
        if "|" not in text:
            continue
        staged, original = text.split("|", 1)
        original_name = Path(original).name
        original_name = re.sub(r"\.(?:mzXML|mzML)$", "", original_name, flags=re.I)
        file_mapping[Path(staged).name] = original_name
    if not file_mapping:
        raise ValueError("params XML contains no upload_file_mapping records")

    members = pd.read_csv(members_path, sep="\t", low_memory=False)
    required_members = {"#ClusterIdx", "#Filename", "#RetTime"}
    if not required_members.issubset(members.columns):
        raise ValueError(f"cluster member table lacks {sorted(required_members - set(members.columns))}")
    members["cluster index"] = pd.to_numeric(members["#ClusterIdx"], errors="raise").astype(int)
    members["member_rt"] = pd.to_numeric(members["#RetTime"], errors="coerce")
    members["staged_filename"] = members["#Filename"].map(lambda value: Path(str(value)).name)
    members["source_sample"] = members["staged_filename"].map(file_mapping)
    if members["source_sample"].isna().any():
        missing = members.loc[members["source_sample"].isna(), "staged_filename"].unique()[:5]
        raise ValueError(f"unmapped staged filenames: {missing.tolist()}")
    aggregate = members.groupby("cluster index", sort=False).agg(
        member_rt_mean=("member_rt", "mean"),
        source_samples=("source_sample", lambda values: frozenset(map(str, values))),
        constituent_spectra=("source_sample", "size"),
        unique_source_samples=("source_sample", "nunique"),
    ).reset_index()

    summary = pd.read_csv(summary_path, sep="\t", low_memory=False)
    required_summary = {"cluster index", "parent mass"}
    if not required_summary.issubset(summary.columns):
        raise ValueError(f"cluster summary lacks {sorted(required_summary - set(summary.columns))}")
    summary["cluster index"] = pd.to_numeric(summary["cluster index"], errors="raise").astype(int)
    summary["parent mass"] = pd.to_numeric(summary["parent mass"], errors="coerce")
    if "RTMean" in summary.columns:
        summary = summary.rename(columns={"RTMean": "summary_rt_mean"})

    library = pd.read_csv(library_path, sep="\t", low_memory=False)
    if not {"cluster index", "LibraryID"}.issubset(library.columns):
        raise ValueError("library cluster table lacks cluster index or LibraryID")
    library["cluster index"] = pd.to_numeric(library["cluster index"], errors="raise").astype(int)
    library = library[["cluster index", "LibraryID"]].drop_duplicates("cluster index")

    cluster = summary.merge(aggregate, on="cluster index", how="left", validate="one_to_one")
    cluster = cluster.merge(library, on="cluster index", how="left", validate="one_to_one")
    if cluster["source_samples"].isna().any():
        raise ValueError("some consensus clusters lack constituent spectra")
    cluster["RTMean"] = pd.to_numeric(cluster["member_rt_mean"], errors="coerce")
    return cluster


def build_feature_cluster_join(
    abundance: pd.DataFrame,
    clusters: pd.DataFrame,
    mgf_peak_counts: dict[int, int],
    minimum_ms2_peaks: int,
    ppm: float,
    rt_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        column for column in abundance.columns
        if column not in META_COLUMNS and column != "age_numeric"
    ]
    parsed = [parse_feature_name(column) for column in feature_columns]
    feature_mass = np.asarray([value[0] for value in parsed], dtype=float)
    feature_rt = np.asarray([value[1] for value in parsed], dtype=float)

    cluster = clusters.copy()
    required = {"cluster index", "parent mass", "RTMean", "source_samples", "LibraryID"}
    missing = required - set(cluster.columns)
    if missing:
        raise ValueError(f"cluster table lacks columns: {sorted(missing)}")
    cluster["cluster index"] = pd.to_numeric(cluster["cluster index"], errors="raise").astype(int)
    cluster["parent mass"] = pd.to_numeric(cluster["parent mass"], errors="coerce")
    cluster["RTMean"] = pd.to_numeric(cluster["RTMean"], errors="coerce")
    cluster["ms2_peak_count"] = cluster["cluster index"].map(mgf_peak_counts)
    cluster = cluster.loc[
        cluster["ms2_peak_count"].ge(minimum_ms2_peaks)
        & cluster["parent mass"].notna()
        & cluster["RTMean"].notna()
    ].copy()
    cluster["library_id_clean"] = cluster["LibraryID"].map(normalize_library_id)
    cluster = cluster.sort_values("parent mass").reset_index(drop=True)
    masses = cluster["parent mass"].to_numpy(float)

    sample_ids = abundance["#OTU ID"].astype(str).to_numpy()
    intensity = abundance[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    observed_samples = [
        frozenset(sample_ids[intensity[column].to_numpy(float) > 0.0])
        for column in feature_columns
    ]

    rows: list[dict[str, object]] = []
    for index, feature_id in enumerate(feature_columns):
        mass = feature_mass[index]
        tolerance = mass * ppm * 1e-6
        lo = int(np.searchsorted(masses, mass - tolerance, side="left"))
        hi = int(np.searchsorted(masses, mass + tolerance, side="right"))
        if hi <= lo:
            continue
        candidate = cluster.iloc[lo:hi]
        candidate = candidate.loc[(candidate["RTMean"] - feature_rt[index]).abs() <= rt_seconds]
        for _, record in candidate.iterrows():
            overlap = observed_samples[index].intersection(record["source_samples"])
            if not overlap:
                continue
            rows.append(
                {
                    "feature_id": str(feature_id).strip(),
                    "feature_mz": mass,
                    "feature_rt_seconds": feature_rt[index],
                    "cluster_index": int(record["cluster index"]),
                    "cluster_parent_mz": float(record["parent mass"]),
                    "cluster_rt_seconds": float(record["RTMean"]),
                    "ppm_error": 1e6 * (float(record["parent mass"]) - mass) / mass,
                    "rt_error_seconds": float(record["RTMean"]) - feature_rt[index],
                    "ms2_peak_count": int(record["ms2_peak_count"]),
                    "overlap_sample_count": len(overlap),
                    "library_id": record["library_id_clean"],
                    "library_supported": bool(record["library_id_clean"]),
                }
            )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("no feature-to-cluster candidates passed the fixed join")
    candidates["cluster_feature_count"] = candidates.groupby(
        "cluster_index", sort=False
    )["feature_id"].transform("nunique")
    counts = candidates.groupby("feature_id", sort=False).agg(
        cluster_candidate_count=("cluster_index", "nunique"),
        library_candidate_count=("library_supported", "sum"),
        maximum_sample_overlap=("overlap_sample_count", "max"),
        minimum_ms2_peak_count=("ms2_peak_count", "min"),
        maximum_ms2_peak_count=("ms2_peak_count", "max"),
        minimum_cluster_feature_count=("cluster_feature_count", "min"),
        maximum_cluster_feature_count=("cluster_feature_count", "max"),
    )
    counts["spectral_evidence_status"] = np.select(
        [
            counts["cluster_candidate_count"].eq(1)
            & counts["maximum_cluster_feature_count"].eq(1)
            & counts["library_candidate_count"].eq(1),
            counts["cluster_candidate_count"].eq(1)
            & counts["maximum_cluster_feature_count"].eq(1)
            & counts["library_candidate_count"].eq(0),
            counts["cluster_candidate_count"].eq(1)
            & counts["library_candidate_count"].eq(0),
            counts["cluster_candidate_count"].eq(1)
            & counts["library_candidate_count"].gt(0),
            counts["cluster_candidate_count"].gt(1) & counts["library_candidate_count"].eq(0),
        ],
        [
            "reciprocal_unique_library_supported",
            "reciprocal_unique_unresolved",
            "single_candidate_nonreciprocal_unresolved",
            "single_candidate_nonreciprocal_with_library",
            "ambiguous_unresolved",
        ],
        default="ambiguous_with_library_candidate",
    )
    return candidates.sort_values(["feature_id", "cluster_index"]), counts.reset_index()


def build_contrast_ledger(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for specification in CONTRAST_SPECS:
        control_all = metadata.loc[metadata["DESCRIPTION"].eq(specification["control"])].copy()
        treatment_all = metadata.loc[
            metadata["DESCRIPTION"].eq(specification["treatment"])
        ].copy()
        raw_common_dates = sorted(
            set(control_all["COLLECTION_TIMESTAMP"].astype(str))
            & set(treatment_all["COLLECTION_TIMESTAMP"].astype(str))
        )
        raw_date_age_mean_differences: list[float] = []
        raw_date_age_range_overlap: list[bool] = []
        for date in raw_common_dates:
            control_age = pd.to_numeric(
                control_all.loc[
                    control_all["COLLECTION_TIMESTAMP"].astype(str).eq(date),
                    "age_numeric",
                ],
                errors="coerce",
            ).dropna()
            treatment_age = pd.to_numeric(
                treatment_all.loc[
                    treatment_all["COLLECTION_TIMESTAMP"].astype(str).eq(date),
                    "age_numeric",
                ],
                errors="coerce",
            ).dropna()
            if control_age.empty or treatment_age.empty:
                raw_date_age_mean_differences.append(float("nan"))
                raw_date_age_range_overlap.append(False)
            else:
                raw_date_age_mean_differences.append(
                    abs(float(control_age.mean()) - float(treatment_age.mean()))
                )
                raw_date_age_range_overlap.append(
                    max(float(control_age.min()), float(treatment_age.min()))
                    <= min(float(control_age.max()), float(treatment_age.max()))
                )
        if specification["require_within_date_age_range_overlap"]:
            common_dates = [
                date
                for date, supported in zip(
                    raw_common_dates, raw_date_age_range_overlap, strict=True
                )
                if supported
            ]
        else:
            common_dates = raw_common_dates
        control = control_all.loc[
            control_all["COLLECTION_TIMESTAMP"].astype(str).isin(common_dates)
        ]
        treatment = treatment_all.loc[
            treatment_all["COLLECTION_TIMESTAMP"].astype(str).isin(common_dates)
        ]
        included_date_indices = [raw_common_dates.index(date) for date in common_dates]
        date_age_mean_differences = [
            raw_date_age_mean_differences[index] for index in included_date_indices
        ]
        date_age_range_overlap = [
            raw_date_age_range_overlap[index] for index in included_date_indices
        ]
        control_cages = int(control["cage"].nunique())
        treatment_cages = int(treatment["cage"].nunique())
        control_cage_ids = set(control["cage"].astype(str))
        treatment_cage_ids = set(treatment["cage"].astype(str))
        shared_cage_count = len(control_cage_ids & treatment_cage_ids)
        unique_cage_count = len(control_cage_ids | treatment_cage_ids)
        cage_arm_cell_count = control_cages + treatment_cages
        anticipated_model_parameters = len(common_dates) + 2
        enough_model_support = bool(
            cage_arm_cell_count - anticipated_model_parameters >= 2
            and unique_cage_count >= anticipated_model_parameters + 2
        )
        control_min, control_max = control["age_numeric"].min(), control["age_numeric"].max()
        treatment_min = treatment["age_numeric"].min()
        treatment_max = treatment["age_numeric"].max()
        age_overlap = bool(
            len(control)
            and len(treatment)
            and max(control_min, treatment_min) <= min(control_max, treatment_max)
        )
        enough_cages = (
            control_cages >= int(specification["minimum_cages"])
            and treatment_cages >= int(specification["minimum_cages"])
            and unique_cage_count >= int(specification["minimum_unique_cages"])
        )
        enough_dates = len(common_dates) >= int(specification["minimum_shared_dates"])
        included_roles = {
            "exploratory_development",
            "primary_development_positive_control",
            "sensitivity_development_positive_control",
        }
        included = specification["analysis_role"] in included_roles
        eligible = bool(
            included
            and enough_cages
            and enough_dates
            and age_overlap
            and enough_model_support
        )
        primary = specification["analysis_role"] == "primary_development_positive_control"
        reasons: list[str] = []
        if not included:
            reasons.append(str(specification["analysis_role"]))
        if not enough_dates:
            reasons.append("insufficient_shared_collection_dates")
        if not enough_cages:
            reasons.append("insufficient_independent_date_matched_cages")
        if not age_overlap:
            reasons.append("nonoverlapping_date_matched_age_ranges")
        if not enough_model_support:
            reasons.append("insufficient_cages_for_date_and_age_adjusted_model")
        rows.append(
            {
                "contrast_id": specification["contrast_id"],
                "contrast_family": specification["contrast_family"],
                "analysis_role": specification["analysis_role"],
                "control_description": specification["control"],
                "treatment_description": specification["treatment"],
                "all_control_samples": len(control_all),
                "all_treatment_samples": len(treatment_all),
                "raw_shared_collection_dates": "|".join(raw_common_dates),
                "raw_shared_collection_date_count": len(raw_common_dates),
                "requires_within_date_age_range_overlap": bool(
                    specification["require_within_date_age_range_overlap"]
                ),
                "shared_collection_dates": "|".join(common_dates),
                "shared_collection_date_count": len(common_dates),
                "minimum_required_shared_dates": int(
                    specification["minimum_shared_dates"]
                ),
                "date_matched_control_samples": len(control),
                "date_matched_treatment_samples": len(treatment),
                "date_matched_control_cages": control_cages,
                "date_matched_treatment_cages": treatment_cages,
                "date_matched_shared_cages": shared_cage_count,
                "date_matched_unique_cages": unique_cage_count,
                "date_matched_cage_arm_cells": cage_arm_cell_count,
                "anticipated_model_parameters": anticipated_model_parameters,
                "anticipated_model_residual_df": (
                    cage_arm_cell_count - anticipated_model_parameters
                ),
                "control_age_min": control_min,
                "control_age_max": control_max,
                "treatment_age_min": treatment_min,
                "treatment_age_max": treatment_max,
                "age_ranges_overlap": age_overlap,
                "maximum_within_date_mean_age_difference_weeks": (
                    float(np.nanmax(date_age_mean_differences))
                    if date_age_mean_differences
                    else float("nan")
                ),
                "all_shared_dates_have_arm_age_range_overlap": bool(
                    date_age_range_overlap and all(date_age_range_overlap)
                ),
                "all_raw_shared_dates_have_arm_age_range_overlap": bool(
                    raw_date_age_range_overlap and all(raw_date_age_range_overlap)
                ),
                "response_analysis_eligible": eligible,
                "primary_response_eligible": bool(eligible and primary),
                "qualification_notes": "pass" if not reasons else ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def median_ratio_normalize(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    array = values.to_numpy(float)
    positive = array > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(positive, np.log(array), np.nan)
    geometric_mean = np.full(array.shape[1], np.nan, dtype=float)
    positive_columns = positive.any(axis=0)
    geometric_mean[positive_columns] = np.exp(
        np.nansum(logged[:, positive_columns], axis=0)
        / positive[:, positive_columns].sum(axis=0)
    )
    usable = np.isfinite(geometric_mean) & (geometric_mean > 0) & (positive.mean(axis=0) >= 0.2)
    if usable.sum() < 100:
        raise ValueError("too few features for median-ratio normalization")
    ratios = np.where(array[:, usable] > 0, array[:, usable] / geometric_mean[usable], np.nan)
    with np.errstate(invalid="ignore"):
        size_factors = np.nanmedian(ratios, axis=1)
    valid = np.isfinite(size_factors) & (size_factors > 0)
    if not valid.all():
        raise ValueError("non-finite sample size factor")
    size_factors = size_factors / np.median(size_factors)
    normalized = array / size_factors[:, None]
    return pd.DataFrame(normalized, index=values.index, columns=values.columns), pd.Series(
        size_factors, index=values.index, name="size_factor"
    )


def fit_contrast_effects(
    abundance: pd.DataFrame,
    normalized: pd.DataFrame,
    contrast: pd.Series,
    evidence: pd.DataFrame,
    minimum_detected_cages: int,
    minimum_pooled_cage_prevalence: float,
) -> pd.DataFrame:
    descriptions = [contrast["control_description"], contrast["treatment_description"]]
    common_dates = str(contrast["shared_collection_dates"]).split("|")
    selected = abundance["DESCRIPTION"].isin(descriptions) & abundance[
        "COLLECTION_TIMESTAMP"
    ].astype(str).isin(common_dates)
    meta = abundance.loc[selected, META_COLUMNS].copy()
    data = normalized.loc[selected].copy()
    meta["treatment_binary"] = meta["DESCRIPTION"].eq(
        contrast["treatment_description"]
    ).astype(float)
    meta["age_numeric"] = pd.to_numeric(meta["age weeks"], errors="coerce")
    meta["cage_id"] = meta["cage"].astype(str)

    log_data = np.log2(data + 1.0)
    cell_index = pd.MultiIndex.from_arrays(
        [meta["cage_id"].to_numpy(), meta["treatment_binary"].to_numpy()],
        names=["cage_id", "treatment_binary"],
    )
    log_data.index = cell_index
    cage_data = log_data.groupby(
        level=["cage_id", "treatment_binary"], sort=True
    ).mean()
    raw_selected = abundance.loc[selected, data.columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    raw_selected.index = cell_index
    cage_detected = raw_selected.gt(0).groupby(
        level=["cage_id", "treatment_binary"], sort=True
    ).max()
    meta_for_cells = meta.copy()
    meta_for_cells.index = cell_index
    cage_meta = meta_for_cells.groupby(
        level=["cage_id", "treatment_binary"], sort=True
    ).agg(
        age_numeric=("age_numeric", "mean"),
        collection_date=("COLLECTION_TIMESTAMP", "first"),
        mouse_count=("#OTU ID", "size"),
    )
    cage_meta["treatment_binary"] = cage_meta.index.get_level_values(
        "treatment_binary"
    ).astype(float)
    if not cage_data.index.equals(cage_meta.index):
        raise AssertionError("cage abundance and metadata indices differ")

    date_dummies = pd.get_dummies(
        cage_meta["collection_date"].astype(str), drop_first=True, dtype=float
    )
    if cage_meta["age_numeric"].isna().any():
        raise ValueError(f"missing age in contrast {contrast['contrast_id']}")
    centered_age = (
        cage_meta["age_numeric"].to_numpy(float)
        - cage_meta["age_numeric"].mean()
    )
    design = np.column_stack(
        [
            np.ones(len(cage_meta)),
            cage_meta["treatment_binary"].to_numpy(float),
            date_dummies.to_numpy(float),
            centered_age,
        ]
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(f"rank-deficient design in contrast {contrast['contrast_id']}")
    cluster_ids = cage_meta.index.get_level_values("cage_id").astype(str).to_numpy()
    cluster_count = int(pd.Series(cluster_ids).nunique())
    if cluster_count < design.shape[1] + 2:
        raise ValueError(
            "too few independent cages for clustered inference in contrast "
            f"{contrast['contrast_id']}: clusters={cluster_count}, "
            f"parameters={design.shape[1]}"
        )
    inverse = np.linalg.inv(design.T @ design)
    response = cage_data.to_numpy(float)
    coefficients = inverse @ design.T @ response
    fitted = design @ coefficients
    residual = response - fitted
    beta_cluster_influence: list[np.ndarray] = []
    for cluster_id in pd.unique(cluster_ids):
        rows = cluster_ids == cluster_id
        cluster_score = design[rows].T @ residual[rows]
        beta_cluster_influence.append((inverse @ cluster_score)[1])
    influence = np.vstack(beta_cluster_influence)
    finite_sample_correction = (
        cluster_count / (cluster_count - 1.0)
        * (len(cage_meta) - 1.0)
        / (len(cage_meta) - design.shape[1])
    )
    se = np.sqrt(finite_sample_correction * np.sum(influence**2, axis=0))
    beta = coefficients[1]
    statistic = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    model_residual_df = len(cage_meta) - design.shape[1]
    reference_df = min(cluster_count - 1, model_residual_df)
    p_value = 2.0 * stats.t.sf(np.abs(statistic), df=reference_df)

    date_loso_betas: list[np.ndarray] = []
    for omitted_date in sorted(cage_meta["collection_date"].astype(str).unique()):
        keep = ~cage_meta["collection_date"].astype(str).eq(omitted_date).to_numpy()
        fold_meta = cage_meta.iloc[np.flatnonzero(keep)]
        fold_response = response[keep]
        if fold_meta["treatment_binary"].nunique() < 2:
            continue
        fold_date_dummies = pd.get_dummies(
            fold_meta["collection_date"].astype(str), drop_first=True, dtype=float
        )
        fold_age = (
            fold_meta["age_numeric"].to_numpy(float)
            - fold_meta["age_numeric"].mean()
        )
        fold_design = np.column_stack(
            [
                np.ones(len(fold_meta)),
                fold_meta["treatment_binary"].to_numpy(float),
                fold_date_dummies.to_numpy(float),
                fold_age,
            ]
        )
        if (
            len(fold_meta) <= fold_design.shape[1]
            or np.linalg.matrix_rank(fold_design) < fold_design.shape[1]
        ):
            continue
        date_loso_betas.append(np.linalg.lstsq(fold_design, fold_response, rcond=None)[0][1])
    if date_loso_betas:
        date_loso = np.vstack(date_loso_betas)
        loso_direction_consistent = np.all(
            np.sign(date_loso) == np.sign(beta)[None, :], axis=0
        ) & np.isfinite(beta) & (beta != 0)
        loso_min_abs_beta = np.min(np.abs(date_loso), axis=0)
    else:
        loso_direction_consistent = np.zeros(beta.shape, dtype=bool)
        loso_min_abs_beta = np.full(beta.shape, np.nan, dtype=float)

    treatment_mask = cage_meta["treatment_binary"].eq(1.0)
    control_mask = ~treatment_mask
    treatment_prevalence = cage_detected.loc[treatment_mask].mean(axis=0).to_numpy(float)
    control_prevalence = cage_detected.loc[control_mask].mean(axis=0).to_numpy(float)
    pooled_cage_detected = cage_detected.groupby(level="cage_id", sort=True).max()
    pooled_detected_cages = pooled_cage_detected.sum(axis=0).to_numpy(int)
    pooled_cage_prevalence = pooled_cage_detected.mean(axis=0).to_numpy(float)
    required_detected_cages = max(
        int(minimum_detected_cages),
        int(math.ceil(minimum_pooled_cage_prevalence * cluster_count)),
    )
    eligible = pooled_detected_cages >= required_detected_cages
    result = pd.DataFrame(
        {
            "study_id": "MSV000080918",
            "source_family": "MSV000080918_MSV000080558",
            "contrast_id": contrast["contrast_id"],
            "contrast_family": contrast["contrast_family"],
            "analysis_role": contrast["analysis_role"],
            "feature_id": cage_data.columns.astype(str).str.strip(),
            "beta_log2_treatment_minus_control": beta,
            "se_cluster_cr1": se,
            "t_cluster_cr1": statistic,
            "p_cluster_cr1": p_value,
            "model_residual_df": model_residual_df,
            "cluster_reference_df": reference_df,
            "date_loso_estimable_folds": len(date_loso_betas),
            "date_loso_direction_consistent": loso_direction_consistent,
            "date_loso_min_abs_beta": loso_min_abs_beta,
            "independent_cage_clusters": cluster_count,
            "cage_arm_cells": int(len(cage_meta)),
            "treatment_cages": int(treatment_mask.sum()),
            "control_cages": int(control_mask.sum()),
            "treatment_prevalence_cage": treatment_prevalence,
            "control_prevalence_cage": control_prevalence,
            "pooled_detected_cages": pooled_detected_cages,
            "pooled_cage_prevalence": pooled_cage_prevalence,
            "minimum_required_detected_cages": required_detected_cages,
            "response_eligible": eligible,
            "collection_date_fixed_effects": int(date_dummies.shape[1]),
            "age_adjusted_weeks": True,
            "inference_unit": "cage_clustered_cage_by_arm_cell",
        }
    )
    result["bh_family_size"] = int(result["response_eligible"].sum())
    result["q_cluster_cr1_within_contrast"] = bh_adjust(
        np.where(result["response_eligible"], result["p_cluster_cr1"], np.nan)
    )
    result = result.merge(evidence, on="feature_id", how="inner", validate="one_to_one")
    return result


def build_contrast_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for contrast_id, block in effects.groupby("contrast_id", sort=False):
        eligible = block.loc[block["response_eligible"]]
        significant = eligible.loc[
            eligible["q_cluster_cr1_within_contrast"].le(0.05)
        ]
        robust_significant = significant.loc[
            significant["date_loso_direction_consistent"]
        ]
        rows.append(
            {
                "contrast_id": contrast_id,
                "contrast_family": block["contrast_family"].iloc[0],
                "analysis_role": block["analysis_role"].iloc[0],
                "effect_rows": len(block),
                "response_eligible_features": len(eligible),
                "bh_family_size_all_abundance_features": int(
                    block["bh_family_size"].iloc[0]
                ),
                "q05_features": len(significant),
                "q05_positive_features": int(
                    significant["beta_log2_treatment_minus_control"].gt(0).sum()
                ),
                "q05_negative_features": int(
                    significant["beta_log2_treatment_minus_control"].lt(0).sum()
                ),
                "q05_date_loso_direction_consistent": int(len(robust_significant)),
                "q05_date_loso_reciprocal_unique_unresolved": int(
                    robust_significant["spectral_evidence_status"]
                    .eq("reciprocal_unique_unresolved")
                    .sum()
                ),
                "q05_date_loso_reciprocal_unique_library_supported": int(
                    robust_significant["spectral_evidence_status"]
                    .eq("reciprocal_unique_library_supported")
                    .sum()
                ),
                "q05_reciprocal_unique_unresolved": int(
                    significant["spectral_evidence_status"]
                    .eq("reciprocal_unique_unresolved")
                    .sum()
                ),
                "q05_reciprocal_unique_library_supported": int(
                    significant["spectral_evidence_status"].eq(
                        "reciprocal_unique_library_supported"
                    ).sum()
                ),
                "control_cages": int(block["control_cages"].iloc[0]),
                "treatment_cages": int(block["treatment_cages"].iloc[0]),
                "model_residual_df": int(block["model_residual_df"].iloc[0]),
                "cluster_reference_df": int(block["cluster_reference_df"].iloc[0]),
                "independent_cage_clusters": int(
                    block["independent_cage_clusters"].iloc[0]
                ),
                "cage_arm_cells": int(block["cage_arm_cells"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_tsv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    abundance = pd.read_csv(args.abundance, low_memory=False)
    if abundance.columns[: len(META_COLUMNS)].tolist() != META_COLUMNS:
        raise ValueError("unexpected abundance metadata schema")
    abundance["age_numeric"] = pd.to_numeric(abundance["age weeks"], errors="coerce")
    biological = abundance.loc[abundance["diet"].isin(["HFD", "NC"])].copy()
    if len(biological) != 451:
        raise AssertionError(f"expected 451 biological samples, observed {len(biological)}")

    clusters = build_full_cluster_table(
        args.cluster_members,
        args.cluster_summary,
        args.params_xml,
        args.library_clusters,
    )
    mgf_peak_counts = parse_mgf_scans(args.mgf)
    candidates, evidence = build_feature_cluster_join(
        biological,
        clusters,
        mgf_peak_counts,
        args.minimum_ms2_peaks,
        args.ppm,
        args.rt_seconds,
    )

    panels = build_contrast_ledger(biological)
    feature_columns = [column for column in biological.columns if column not in META_COLUMNS + ["age_numeric"]]
    numeric = biological[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    normalized, size_factors = median_ratio_normalize(numeric)
    normalized.index = biological.index

    analysis_panels = panels.loc[panels["response_analysis_eligible"]]
    primary_panels = panels.loc[panels["primary_response_eligible"]]
    effect_blocks = [
        fit_contrast_effects(
            biological,
            normalized,
            panel,
            evidence,
            args.minimum_detected_cages,
            args.minimum_pooled_cage_prevalence,
        )
        for _, panel in analysis_panels.iterrows()
    ]
    effects = pd.concat(effect_blocks, ignore_index=True)
    contrast_summary = build_contrast_summary(effects)

    candidates_path = args.out_dir / "feature_cluster_candidates.tsv.gz"
    evidence_path = args.out_dir / "feature_spectral_evidence.tsv"
    panels_path = args.out_dir / "panel_qualification.tsv"
    effects_path = args.out_dir / "contrast_effects.tsv.gz"
    contrast_summary_path = args.out_dir / "contrast_summary.tsv"
    sample_norm_path = args.out_dir / "sample_normalization.tsv"
    write_tsv_gz(candidates, candidates_path)
    evidence.to_csv(evidence_path, sep="\t", index=False)
    panels.to_csv(panels_path, sep="\t", index=False)
    write_tsv_gz(effects, effects_path)
    contrast_summary.to_csv(contrast_summary_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": biological["#OTU ID"].astype(str),
            "cage": biological["cage"].astype(str),
            "description": biological["DESCRIPTION"].astype(str),
            "size_factor": size_factors.to_numpy(float),
        }
    ).to_csv(sample_norm_path, sep="\t", index=False)

    status_counts = evidence["spectral_evidence_status"].value_counts().to_dict()
    eligible_effects = effects.loc[effects["response_eligible"]]
    report = {
        "schema": "contrast_atlas_msv000080918_multistratum_response_v1",
        "status": "DEVELOPMENT_CONSUMED_RESPONSE_SOURCE_COMPLETE",
        "evidence_boundary": {
            "source_status": "development_consumed",
            "independent_source_count": 1,
            "accessions_are_same_parent_study": True,
            "xalign_overlap": (
                "X-Align used the B6-HFD (6M) 5-mouse versus B6-NC (6M) "
                "6-mouse subset; it is not an unopened external panel."
            ),
            "original_paper_consumed_positive_controls": [
                "mup_hfd_10_11m_iga_ko_vs_intact_date_matched",
                "mup_hfd_10_11m_cd8_ko_vs_intact_date_matched",
            ],
            "causal_claim": False,
            "identity_truth_claimed": False,
            "mechanism_claimed": False,
            "inference_unit": "cage cluster with cage-by-arm analysis cells",
        },
        "inputs": {
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "abundance": str(args.abundance),
            "abundance_sha256": sha256(args.abundance),
            "cluster_members": str(args.cluster_members),
            "cluster_members_sha256": sha256(args.cluster_members),
            "cluster_summary": str(args.cluster_summary),
            "cluster_summary_sha256": sha256(args.cluster_summary),
            "params_xml": str(args.params_xml),
            "params_xml_sha256": sha256(args.params_xml),
            "library_clusters": str(args.library_clusters),
            "library_clusters_sha256": sha256(args.library_clusters),
            "mgf": str(args.mgf),
            "mgf_sha256": sha256(args.mgf),
        },
        "join_contract": {
            "precursor_ppm": args.ppm,
            "retention_seconds": args.rt_seconds,
            "minimum_ms2_peaks": args.minimum_ms2_peaks,
            "requires_nonzero_source_sample_overlap": True,
            "uses_abundance_presence_for_join": True,
            "uses_arm_labels_or_effect_estimates_for_join": False,
        },
        "source_counts": {
            "biological_samples": int(len(biological)),
            "cages": int(biological["cage"].nunique()),
            "abundance_features": int(len(feature_columns)),
            "mgf_spectra_total": int(len(mgf_peak_counts)),
            "mgf_spectra_meeting_peak_qc": int(
                sum(value >= args.minimum_ms2_peaks for value in mgf_peak_counts.values())
            ),
            "mgf_spectra_below_peak_qc": int(
                sum(value < args.minimum_ms2_peaks for value in mgf_peak_counts.values())
            ),
            "cluster_rows_meeting_mgf_peak_qc": int(
                pd.to_numeric(clusters["cluster index"], errors="coerce")
                .map(mgf_peak_counts)
                .ge(args.minimum_ms2_peaks)
                .sum()
            ),
            "features_with_spectral_candidates": int(evidence["feature_id"].nunique()),
            "feature_cluster_candidate_rows": int(len(candidates)),
            "spectral_evidence_status_counts": {
                str(key): int(value) for key, value in status_counts.items()
            },
        },
        "response_counts": {
            "intended_panels": int(len(panels)),
            "analysis_eligible_contrasts": int(analysis_panels.shape[0]),
            "primary_eligible_contrasts": int(primary_panels.shape[0]),
            "effect_rows": int(len(effects)),
            "eligible_effect_rows": int(len(eligible_effects)),
            "eligible_features": int(eligible_effects["feature_id"].nunique()),
            "within_contrast_q05_rows": int(
                eligible_effects["q_cluster_cr1_within_contrast"].le(0.05).sum()
            ),
            "contrasts_with_q05_features": int(contrast_summary["q05_features"].gt(0).sum()),
        },
        "method": {
            "normalization": "median-ratio across biological samples",
            "transform": "log2(normalized abundance + 1)",
            "within_cage_arm": "mean of mouse-level log abundance",
            "effect_model": (
                "OLS treatment indicator plus shared-collection-date fixed effects and "
                "centered cage-cell mean age in weeks, "
                "CR1 finite-sample-corrected cage-clustered SE, cage-by-arm rows; "
                "t reference df=min(number of cages minus 1, model residual df)"
            ),
            "response_filter": (
                "label-blind pooled cage detection; at least max(minimum_detected_cages, "
                "ceil(minimum_pooled_cage_prevalence * independent cages))"
            ),
            "minimum_detected_cages": args.minimum_detected_cages,
            "minimum_pooled_cage_prevalence": args.minimum_pooled_cage_prevalence,
            "multiple_testing": (
                "BH within each contrast over all label-blind response-eligible abundance "
                "features before the spectral-evidence inner join"
            ),
            "date_robustness": (
                "leave one collection date out; report direction consistency separately "
                "from the full-model BH result"
            ),
            "primary_panel_gate": (
                "predefined analysis role; >=specification shared collection dates, cages "
                "per arm, and total unique cages; overlapping observed age ranges"
            ),
        },
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checksum_targets = [
        candidates_path,
        evidence_path,
        panels_path,
        effects_path,
        contrast_summary_path,
        sample_norm_path,
        report_path,
    ]
    checksum_path = args.out_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_targets),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
