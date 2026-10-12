#!/usr/bin/env python
"""Project abstaining public-evidence certificates into sample signatures.

This step is deliberately phenotype blind.  It combines within-study abundance
and measurement reliability with certificates that were frozen independently.
Unassigned or zero-confidence signal is retained as an explicit dustbin and is
never renormalized away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_ABUNDANCE = {"sample_id", "query_id", "abundance"}
REQUIRED_CERTIFICATE = {"query_id", "assigned_public_unit", "certificate_score"}
FORBIDDEN = (
    "truth", "label", "phenotype", "disease", "drug", "case", "control",
    "tumor", "tumour", "normal", "group", "outcome", "survival",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def reject_outcomes(frame: pd.DataFrame, name: str) -> None:
    contaminated = [column for column in frame.columns
                    if any(token in column.lower() for token in FORBIDDEN)]
    if contaminated:
        raise ValueError(f"{name} contains forbidden outcome columns: {contaminated}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abundance", type=Path, required=True,
                        help="Long CSV/Parquet: sample_id, query_id, abundance, optional reliability.")
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    abundance = read_table(args.abundance)
    certificates = read_table(args.certificates)
    missing_abundance = REQUIRED_ABUNDANCE - set(abundance.columns)
    missing_certificate = REQUIRED_CERTIFICATE - set(certificates.columns)
    if missing_abundance or missing_certificate:
        raise ValueError({
            "missing_abundance_columns": sorted(missing_abundance),
            "missing_certificate_columns": sorted(missing_certificate),
        })
    reject_outcomes(abundance, "abundance")
    reject_outcomes(certificates, "certificates")
    if certificates["query_id"].duplicated().any():
        raise ValueError("certificates must contain exactly one row per query_id")
    if abundance.duplicated(["sample_id", "query_id"]).any():
        raise ValueError("abundance must contain at most one row per sample_id/query_id")

    abundance = abundance.copy()
    abundance["abundance"] = pd.to_numeric(abundance["abundance"], errors="raise")
    if "reliability" not in abundance:
        abundance["reliability"] = 1.0
    abundance["reliability"] = pd.to_numeric(abundance["reliability"], errors="raise")
    if (not np.isfinite(abundance[["abundance", "reliability"]].to_numpy()).all()
            or (abundance["abundance"] < 0).any()
            or ((abundance["reliability"] < 0) | (abundance["reliability"] > 1)).any()):
        raise ValueError("abundance must be finite/nonnegative and reliability finite in [0,1]")

    certificates = certificates.copy()
    certificates["certificate_score"] = pd.to_numeric(
        certificates["certificate_score"], errors="raise")
    if (not np.isfinite(certificates["certificate_score"]).all()
            or ((certificates["certificate_score"] < 0)
                | (certificates["certificate_score"] > 1)).any()):
        raise ValueError("certificate_score must be finite in [0,1]")

    joined = abundance.merge(
        certificates[list(REQUIRED_CERTIFICATE)], on="query_id", how="left", validate="many_to_one"
    )
    joined["assigned_public_unit"] = joined["assigned_public_unit"].fillna("__NO_MATCH__")
    joined["certificate_score"] = joined["certificate_score"].fillna(0.0)
    joined["base_signal"] = np.log1p(joined["abundance"]) * joined["reliability"]
    joined["is_certified"] = ((joined["assigned_public_unit"] != "__NO_MATCH__")
                              & (joined["certificate_score"] > 0))
    joined["certified_signal"] = np.where(
        joined["is_certified"], joined["base_signal"] * joined["certificate_score"], 0.0
    )

    signature = (joined.loc[joined["certified_signal"] > 0]
                 .groupby(["sample_id", "assigned_public_unit"], as_index=False)["certified_signal"]
                 .sum())
    matched_total = signature.groupby("sample_id")["certified_signal"].transform("sum")
    signature["within_certified_fraction"] = np.divide(
        signature["certified_signal"], matched_total,
        out=np.zeros(len(signature), dtype=float), where=matched_total.to_numpy() > 0,
    )

    sample_qc = joined.groupby("sample_id", as_index=False).agg(
        observed_queries=("query_id", "nunique"),
        total_base_signal=("base_signal", "sum"),
        certified_base_signal=("base_signal", lambda values: float(
            values[joined.loc[values.index, "is_certified"]].sum())),
        certificate_weighted_signal=("certified_signal", "sum"),
    )
    sample_qc["certified_coverage"] = np.divide(
        sample_qc["certified_base_signal"], sample_qc["total_base_signal"],
        out=np.zeros(len(sample_qc), dtype=float),
        where=sample_qc["total_base_signal"].to_numpy() > 0,
    )
    sample_qc["dustbin_fraction"] = 1.0 - sample_qc["certified_coverage"]

    args.output.mkdir(parents=True)
    signature.to_csv(args.output / "sample_unit_signature.csv.gz", index=False, compression="gzip")
    sample_qc.to_csv(args.output / "sample_coverage_qc.csv", index=False)
    report = {
        "status": "PUBLIC_EVIDENCE_SAMPLE_PROJECTION_COMPLETE",
        "formal": False,
        "samples": int(joined["sample_id"].nunique()),
        "study_queries": int(joined["query_id"].nunique()),
        "certified_queries_observed": int(joined.loc[joined["is_certified"], "query_id"].nunique()),
        "public_units_observed": int(signature["assigned_public_unit"].nunique()),
        "coverage": {
            "median_certified_signal_fraction": float(sample_qc["certified_coverage"].median()),
            "minimum_certified_signal_fraction": float(sample_qc["certified_coverage"].min()),
            "maximum_certified_signal_fraction": float(sample_qc["certified_coverage"].max()),
        },
        "contracts": {
            "phenotype_blind": True,
            "explicit_dustbin": True,
            "unmatched_signal_not_renormalized_away": True,
            "within_certified_composition_is_conditional_on_coverage": True,
        },
        "provenance": {
            "abundance_sha256": sha256(args.abundance),
            "certificates_sha256": sha256(args.certificates),
        },
        "claim_limit": (
            "Sample-level occupancy of certified public evidence units only; no molecular identity, "
            "pathway, function, phenotype, or mechanism claim."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
