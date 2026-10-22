#!/usr/bin/env python3
"""Estimate the frozen observational MTBLS1866 COVID response channels.

The phenotype-blind measurement report must already be complete.  Continuous
abundance and differential detection are kept as separate response channels;
only abundance is eligible for the current continuous-effect collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = {"COVID-19": 83, "HEALTHY": 26}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=1).astype(int)
    mean = np.divide(
        np.nansum(values, axis=1), count,
        out=np.full(values.shape[0], np.nan), where=count > 0,
    )
    centered = np.where(finite, values - mean[:, None], 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=1), count - 1,
        out=np.full(values.shape[0], np.nan), where=count > 1,
    )
    return count, mean, variance


def strict_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise RuntimeError(f"{label} is not a strict boolean column")
    return normalized.map(mapping).astype(bool)


def acquisition_order_audit(
    measurement_dir: Path, design: pd.DataFrame
) -> dict[str, object]:
    path = measurement_dir / "phenotype_blind_acquisition_times.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if len(frame) != 109 or frame.assay_name.nunique() != 109:
        raise RuntimeError("acquisition-time universe drift")
    merged = frame.merge(
        design[["Sample Name", "group"]],
        left_on="assay_name",
        right_on="Sample Name",
        how="inner",
        validate="one_to_one",
    )
    parsed = pd.to_datetime(merged.acquisition_start_time, errors="coerce", utc=True)
    result: dict[str, object] = {
        "files": len(merged),
        "parsed_start_times": int(parsed.notna().sum()),
        "complete": bool(parsed.notna().all()),
        "order_auc_covid_after_healthy": None,
        "group_transitions": None,
        "perfect_temporal_separation": None,
    }
    if not parsed.notna().all():
        return result
    ordered = merged.assign(_time=parsed).sort_values("_time").reset_index(drop=True)
    labels = ordered.group.eq("COVID-19").to_numpy(int)
    result["group_transitions"] = int(np.count_nonzero(labels[1:] != labels[:-1]))
    covid_ranks = np.flatnonzero(labels == 1) + 1
    healthy_ranks = np.flatnonzero(labels == 0) + 1
    rank_sum = float(covid_ranks.sum())
    u_value = rank_sum - len(covid_ranks) * (len(covid_ranks) + 1) / 2
    auc = u_value / (len(covid_ranks) * len(healthy_ranks))
    result["order_auc_covid_after_healthy"] = float(auc)
    result["perfect_temporal_separation"] = bool(auc in {0.0, 1.0})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_mtbls1866_measurement_v1_20261022"
        ),
    )
    parser.add_argument(
        "--design-ledger",
        type=Path,
        default=Path(
            "data/validation/mtbls1866_positive_manifest_v1_20261022/design_ledger.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_mtbls1866_covid_response_v1_20261022"
        ),
    )
    parser.add_argument("--minimum-group-prevalence", type=float, default=0.50)
    parser.add_argument("--minimum-detection-total", type=int, default=10)
    parser.add_argument("--minimum-nondetection-total", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    measurement_report_path = args.measurement_dir / "report.json"
    measurement = json.loads(measurement_report_path.read_text(encoding="utf-8"))
    if measurement.get("status") != "PHENOTYPE_BLIND_MEASUREMENT_OBJECT_FROZEN":
        raise RuntimeError("phenotype-blind measurement object is not frozen")
    if "not inputs" not in measurement.get("leakage_status", ""):
        raise RuntimeError("measurement report does not certify label separation")

    entities_path = args.measurement_dir / "consensus_entities.tsv.gz"
    matrix_path = args.measurement_dir / "discovery_intensity_matrix.tsv.gz"
    entities = pd.read_csv(entities_path, sep="\t")
    matrix = pd.read_csv(matrix_path, sep="\t")
    design = pd.read_csv(args.design_ledger, sep="\t", dtype=str, keep_default_na=False)
    observed = design.groupby("group").size().astype(int).to_dict()
    if observed != GROUPS or design["Sample Name"].nunique() != 109:
        raise RuntimeError(f"design topology drift: {observed}")
    order_audit = acquisition_order_audit(args.measurement_dir, design)
    sample_names = design["Sample Name"].astype(str).tolist()
    if set(sample_names) != set(matrix.columns) - {"feature_id"}:
        raise RuntimeError("design and phenotype-blind intensity matrix do not share one universe")
    if matrix.feature_id.duplicated().any() or entities.feature_id.duplicated().any():
        raise RuntimeError("duplicate consensus feature IDs")
    if set(matrix.feature_id) != set(entities.feature_id):
        raise RuntimeError("entity and intensity universes differ")

    # Column order follows the design ledger only after the frozen measurement is read.
    raw = matrix[sample_names].to_numpy(float)
    detected = np.isfinite(raw) & (raw > 0)
    log_values = np.full(raw.shape, np.nan, dtype=float)
    log_values[detected] = np.log2(raw[detected])
    offsets = np.nanmedian(log_values, axis=0)
    if np.any(~np.isfinite(offsets)):
        raise RuntimeError("at least one sample has no positive consensus intensity")
    normalized = log_values - offsets[None, :]

    groups = design.group.to_numpy(str)
    covid = groups == "COVID-19"
    healthy = groups == "HEALTHY"
    n_covid, mean_covid, var_covid = group_moments(normalized[:, covid])
    n_healthy, mean_healthy, var_healthy = group_moments(normalized[:, healthy])
    abundance_beta = mean_covid - mean_healthy
    abundance_se = np.sqrt(var_covid / n_covid + var_healthy / n_healthy)
    prevalence_covid = n_covid / int(covid.sum())
    prevalence_healthy = n_healthy / int(healthy.sum())
    min_covid = math.ceil(args.minimum_group_prevalence * int(covid.sum()))
    min_healthy = math.ceil(args.minimum_group_prevalence * int(healthy.sum()))
    abundance_qc = (
        (n_covid >= min_covid)
        & (n_healthy >= min_healthy)
        & np.isfinite(abundance_beta)
        & np.isfinite(abundance_se)
        & (abundance_se > 0)
    )

    detected_covid = detected[:, covid].sum(axis=1).astype(int)
    detected_healthy = detected[:, healthy].sum(axis=1).astype(int)
    missing_covid = int(covid.sum()) - detected_covid
    missing_healthy = int(healthy.sum()) - detected_healthy
    detection_beta = np.log(
        ((detected_covid + 0.5) / (missing_covid + 0.5))
        / ((detected_healthy + 0.5) / (missing_healthy + 0.5))
    )
    detection_se = np.sqrt(
        1.0 / (detected_covid + 0.5)
        + 1.0 / (missing_covid + 0.5)
        + 1.0 / (detected_healthy + 0.5)
        + 1.0 / (missing_healthy + 0.5)
    )
    total_detected = detected_covid + detected_healthy
    total_missing = missing_covid + missing_healthy
    detection_qc = (
        (total_detected >= args.minimum_detection_total)
        & (total_missing >= args.minimum_nondetection_total)
        & np.isfinite(detection_beta)
        & np.isfinite(detection_se)
        & (detection_se > 0)
    )

    base = entities[
        ["feature_id", "raw_entity_id", "mz", "rt_sec", "has_representative_ms2"]
    ].copy()
    base["has_representative_ms2"] = strict_bool(
        base.has_representative_ms2, "has_representative_ms2"
    )
    abundance = base.copy()
    abundance["contrast_id"] = "noncritical_covid_minus_healthy_plasma"
    abundance["response_channel"] = "continuous_abundance"
    abundance["beta"] = abundance_beta
    abundance["se"] = abundance_se
    abundance["n_positive"] = n_covid
    abundance["n_negative"] = n_healthy
    abundance["prevalence_positive"] = prevalence_covid
    abundance["prevalence_negative"] = prevalence_healthy
    abundance["qc_pass"] = abundance_qc
    abundance["identity_status"] = "unpartitioned"

    detection = base.copy()
    detection["contrast_id"] = "noncritical_covid_minus_healthy_plasma"
    detection["response_channel"] = "differential_detection_log_odds"
    detection["beta"] = detection_beta
    detection["se"] = detection_se
    detection["n_positive"] = int(covid.sum())
    detection["n_negative"] = int(healthy.sum())
    detection["detections_positive"] = detected_covid
    detection["detections_negative"] = detected_healthy
    detection["qc_pass"] = detection_qc
    detection["identity_status"] = "unpartitioned"

    args.output_dir.mkdir(parents=True)
    abundance_all = args.output_dir / "all_abundance_effects.tsv.gz"
    detection_all = args.output_dir / "all_detection_effects.tsv.gz"
    abundance_path = args.output_dir / "ms2_linked_abundance_response.tsv.gz"
    detection_path = args.output_dir / "ms2_linked_detection_response.tsv.gz"
    abundance.to_csv(abundance_all, sep="\t", index=False)
    detection.to_csv(detection_all, sep="\t", index=False)
    abundance_eligible = abundance.loc[
        abundance.qc_pass & abundance.has_representative_ms2
    ].copy()
    detection_eligible = detection.loc[
        detection.qc_pass & detection.has_representative_ms2
    ].copy()
    abundance_eligible.to_csv(abundance_path, sep="\t", index=False)
    detection_eligible.to_csv(detection_path, sep="\t", index=False)

    report = {
        "schema": "contrast_atlas_mtbls1866_covid_response_v1",
        "status": "DEVELOPMENT_CONSUMED_OBSERVATIONAL_RESPONSE_COMPLETE",
        "evidence_role": "development_consumed",
        "leakage_status": (
            "the 109-file chemical measurement object and hashes were frozen before "
            "the COVID/healthy design ledger entered response estimation"
        ),
        "denominators": {
            "participants": 109,
            "noncritical_covid": 83,
            "healthy": 26,
            "consensus_entities": len(entities),
            "entities_with_representative_ms2": int(
                entities.has_representative_ms2.astype(bool).sum()
            ),
            "abundance_qc_all_entities": int(abundance.qc_pass.sum()),
            "abundance_qc_ms2_linked": len(abundance_eligible),
            "detection_qc_all_entities": int(detection.qc_pass.sum()),
            "detection_qc_ms2_linked": len(detection_eligible),
            "eligible_channel_overlap": len(
                set(abundance_eligible.feature_id) & set(detection_eligible.feature_id)
            ),
        },
        "response_models": {
            "primary": (
                "median-normalized log2 intensity; unpaired COVID-minus-healthy mean "
                "difference with Welch standard error among detected observations"
            ),
            "secondary": (
                "Haldane-corrected unpaired differential-detection log odds; frozen "
                "separately and not collapsed into abundance"
            ),
            "minimum_group_prevalence": args.minimum_group_prevalence,
            "minimum_detected_covid": min_covid,
            "minimum_detected_healthy": min_healthy,
        },
        "acquisition_order_audit": order_audit,
        "collection_policy": {
            "current_open_collection_channel": "continuous_abundance_only",
            "detection_channel": "secondary_frozen_not_collapsed",
            "identity_partition": "unpartitioned_pending_frozen_library_search",
        },
        "provenance": {
            "measurement_report_sha256": sha256(measurement_report_path),
            "measurement_matrix_sha256": sha256(matrix_path),
            "design_ledger_sha256": sha256(args.design_ledger),
            "abundance_response_sha256": sha256(abundance_path),
            "detection_response_sha256": sha256(detection_path),
        },
        "claim_boundary": [
            "This is an observational plasma group response, not an unconfounded causal SARS-CoV-2 effect.",
            "The public ISA sheet does not provide a covariate-complete age/sex adjustment for this analysis.",
            "If acquisition-time order is unavailable or nearly separates groups, technical-batch confounding remains unresolved and the source cannot serve as positive biological truth.",
            "Unpartitioned MS2-linked entities are not unknown metabolites; known-only versus unresolved claims are forbidden until a frozen library search is applied.",
            "Shared COVID naming across studies is design relevance, not signed molecular-response truth.",
            "Completing one response object does not establish cross-study retrieval or unknown-information increment.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
