#!/usr/bin/env python3
"""Prepare the paired COVID response for the open Contrast Atlas collection.

Abundance and differential-detection effects remain separate.  Only the
continuous paired-abundance channel enters the current abundance-based open
collection; the detection channel is frozen beside it for a future
channel-aware comparison and is never silently collapsed into abundance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 3041
EXPECTED_USABLE_MS2 = 1616


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise RuntimeError(f"{label} is not a strict boolean column")
    return normalized.map(mapping).astype(bool)


def response_channel(
    frame: pd.DataFrame,
    *,
    qc_column: str,
    beta_column: str,
    se_column: str,
    prevalence: pd.Series,
    model: str,
) -> pd.DataFrame:
    eligible = strict_bool(frame[qc_column], qc_column) & strict_bool(
        frame["has_usable_raw_ms2"], "has_usable_raw_ms2"
    )
    result = pd.DataFrame(
        {
            "entity_id": frame.loc[eligible, "entity_id"].astype(str),
            "beta": pd.to_numeric(frame.loc[eligible, beta_column], errors="raise"),
            "se": pd.to_numeric(frame.loc[eligible, se_column], errors="raise"),
            "prevalence": prevalence.loc[eligible].astype(float),
            "analysis_eligible": True,
            "n_positive": 39,
            "n_negative": 39,
            "effect_model": model,
        }
    )
    if result.entity_id.duplicated().any():
        raise RuntimeError("duplicate eligible entity IDs")
    finite = np.isfinite(result[["beta", "se", "prevalence"]].to_numpy(float)).all()
    if not finite or (result.se <= 0).any():
        raise RuntimeError("eligible response contains non-finite beta/prevalence or non-positive SE")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linked-response",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_msv000092887_raw_ms2_link_v1_20261022/"
            "paired_response_with_raw_ms2.tsv.gz"
        ),
    )
    parser.add_argument(
        "--link-report",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_msv000092887_raw_ms2_link_v1_20261022/"
            "report.json"
        ),
    )
    parser.add_argument(
        "--representative-mgf",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_msv000092887_raw_ms2_link_v1_20261022/"
            "representative_ms2.mgf"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_msv000092887_open_source_v1_20261022"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    link_report = json.loads(args.link_report.read_text(encoding="utf-8"))
    if link_report.get("status") != "PHENOTYPE_BLIND_RAW_MS2_LINK_COMPLETE":
        raise RuntimeError("raw-MS2 linkage is not complete")
    frame = pd.read_csv(args.linked_response, sep="\t", dtype={"entity_id": str})
    if len(frame) != EXPECTED_ROWS or frame.entity_id.nunique() != EXPECTED_ROWS:
        raise RuntimeError("expected 3,041 unique response features")
    usable = strict_bool(frame.has_usable_raw_ms2, "has_usable_raw_ms2")
    if int(usable.sum()) != EXPECTED_USABLE_MS2:
        raise RuntimeError("usable-MS2 denominator drift")
    allowed_partition = {"library_supported", "unresolved_source_local"}
    observed_partition = set(frame.loc[usable, "source_local_ms2_partition"].astype(str))
    if observed_partition != allowed_partition:
        raise RuntimeError(f"unexpected usable identity partitions: {observed_partition}")

    abundance = response_channel(
        frame,
        qc_column="abundance_qc_pass",
        beta_column="abundance_beta_log2_covid_vs_control",
        se_column="abundance_se",
        prevalence=pd.to_numeric(frame.abundance_n_both_present_pairs, errors="raise") / 39.0,
        model="paired mean log2 COVID/control among pairs detected in both; SE across pair differences",
    )
    detection = response_channel(
        frame,
        qc_column="detection_qc_pass",
        beta_column="detection_beta_log_odds_covid_vs_control",
        se_column="detection_se",
        prevalence=pd.to_numeric(frame.detection_n_discordant_pairs, errors="raise") / 39.0,
        model="Haldane-corrected paired differential-detection log odds; separate from abundance",
    )
    identity = frame.loc[
        usable, ["entity_id", "source_local_ms2_partition"]
    ].rename(columns={"source_local_ms2_partition": "identity_status"})
    identity["identity_status"] = identity.identity_status.replace(
        {"unresolved_source_local": "unresolved_entity"}
    )

    abundance_dir = args.output_dir / "covid_vs_age_sex_matched_control_abundance"
    detection_dir = args.output_dir / "secondary_detection_channel"
    abundance_dir.mkdir(parents=True)
    detection_dir.mkdir(parents=True)
    abundance_path = abundance_dir / "response_object.tsv.gz"
    detection_path = detection_dir / "response_object.tsv.gz"
    identity_path = args.output_dir / "identity_partition.tsv.gz"
    abundance.to_csv(abundance_path, sep="\t", index=False, compression="gzip")
    detection.to_csv(detection_path, sep="\t", index=False, compression="gzip")
    identity.to_csv(identity_path, sep="\t", index=False, compression="gzip")

    abundance_ids = set(abundance.entity_id)
    detection_ids = set(detection.entity_id)
    report = {
        "schema": "contrast_atlas_msv000092887_open_source_v1",
        "status": "DEVELOPMENT_CONSUMED_COVID_RESPONSE_SOURCE_PREPARED",
        "evidence_role": "development_consumed",
        "leakage_status": (
            "raw MS2 membership was frozen phenotype-blind before response effects were joined; "
            "the public disease outcome is open and this is not external validation"
        ),
        "denominators": {
            "matched_pairs": 39,
            "participants": 78,
            "all_features": len(frame),
            "features_with_usable_raw_ms2": int(usable.sum()),
            "library_supported_usable_ms2": int(identity.identity_status.eq("library_supported").sum()),
            "unresolved_usable_ms2": int(identity.identity_status.eq("unresolved_entity").sum()),
            "abundance_response_entities": len(abundance),
            "abundance_library_supported": int(
                identity.loc[identity.entity_id.isin(abundance_ids), "identity_status"].eq("library_supported").sum()
            ),
            "abundance_unresolved": int(
                identity.loc[identity.entity_id.isin(abundance_ids), "identity_status"].eq("unresolved_entity").sum()
            ),
            "detection_response_entities": len(detection),
            "detection_library_supported": int(
                identity.loc[identity.entity_id.isin(detection_ids), "identity_status"].eq("library_supported").sum()
            ),
            "detection_unresolved": int(
                identity.loc[identity.entity_id.isin(detection_ids), "identity_status"].eq("unresolved_entity").sum()
            ),
            "response_channel_overlap": len(abundance_ids & detection_ids),
        },
        "collection_policy": {
            "primary_open_collection_channel": "paired_abundance",
            "secondary_channel": "paired_differential_detection_frozen_not_collapsed",
            "reason": (
                "the current public collection contains continuous abundance effects; mixing detection log odds "
                "into the same candidate score before cross-source channel calibration is forbidden"
            ),
        },
        "provenance": {
            "linked_response_sha256": sha256(args.linked_response),
            "link_report_sha256": sha256(args.link_report),
            "representative_mgf_sha256": sha256(args.representative_mgf),
            "abundance_response_sha256": sha256(abundance_path),
            "detection_response_sha256": sha256(detection_path),
            "identity_partition_sha256": sha256(identity_path),
        },
        "claim_boundary": [
            "Source-local unresolved means no deposited reference-spectrum support plus a usable raw-MS2 link; it is not a unique molecule or a harmonized cross-source identity.",
            "The abundance channel is an observational COVID-versus-matched-control response, not a treatment effect.",
            "Preparing this source does not establish a cross-study COVID relation, response replication, or unknown-information increment.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
