#!/usr/bin/env python3
"""Select a metadata-only, repository-scale contrast compute wave.

Accession is not treated as source independence.  Panels are grouped by a
publication identifier when available and exact design fingerprints are used
to flag accession mirrors/updates.  The resulting source-family identifiers
are preliminary: cohort overlap still requires manual/publication review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def stable_id(*parts: object) -> str:
    token = "\x1f".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:16]


def publication_id(value: object) -> str:
    text = str(value or "").strip()
    match = re.match(r"(\d{6,9})###", text)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--massive-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-accessions-per-stratum", type=int, default=2)
    args = parser.parse_args()

    registry = pd.read_csv(args.registry, sep="\t")
    required = {
        "dataset_accession", "comparison_stratum", "contrast_field", "level_a", "level_b",
        "taxonomy", "body_part", "instrument", "polarity", "chromatography",
        "n_units_a", "n_units_b", "n_ms2_spectra_a", "n_ms2_spectra_b",
    }
    missing = required - set(registry)
    if missing:
        raise ValueError(f"registry lacks columns: {sorted(missing)}")
    stratum_n = registry.groupby("comparison_stratum")["dataset_accession"].nunique()
    eligible_strata = set(stratum_n[stratum_n >= args.min_accessions_per_stratum].index)
    wave = registry.loc[registry["comparison_stratum"].isin(eligible_strata)].copy()

    meta_columns = ["dataset", "title", "user", "publication_list", "description"]
    massive = pd.read_csv(
        args.massive_metadata, sep="\t", usecols=meta_columns,
        dtype=str, keep_default_na=False,
    ).drop_duplicates("dataset")
    wave = wave.merge(
        massive, left_on="dataset_accession", right_on="dataset",
        how="left", validate="many_to_one",
    ).drop(columns=["dataset"])
    wave["publication_id"] = wave["publication_list"].map(publication_id)

    fingerprint_columns = [
        "contrast_field", "level_a", "level_b", "taxonomy", "body_part",
        "instrument", "polarity", "chromatography", "n_units_a", "n_units_b",
        "n_ms2_spectra_a", "n_ms2_spectra_b",
    ]
    wave["design_fingerprint"] = [stable_id(*row) for row in wave[fingerprint_columns].itertuples(index=False, name=None)]
    fingerprint_accessions = wave.groupby("design_fingerprint")["dataset_accession"].transform("nunique")
    wave["exact_design_duplicate_suspect"] = fingerprint_accessions > 1

    # Build accession-level connected components. This is important because
    # one mirrored accession can contribute several different contrast rows;
    # source identity must never change from one contrast to another.
    accessions = sorted(wave["dataset_accession"].unique())
    parent = {value: value for value in accessions}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(values: list[str]) -> None:
        if len(values) < 2:
            return
        root = find(values[0])
        for value in values[1:]:
            other = find(value)
            if root != other:
                parent[other] = root

    for _, block in wave.loc[wave["publication_id"].ne("")].groupby("publication_id"):
        union(sorted(block["dataset_accession"].unique()))
    for _, block in wave.groupby("design_fingerprint"):
        union(sorted(block["dataset_accession"].unique()))
    components: dict[str, list[str]] = {}
    for accession in accessions:
        components.setdefault(find(accession), []).append(accession)
    source_by_accession = {
        accession: f"source:{stable_id(*sorted(members))}"
        for members in components.values() for accession in members
    }
    wave["preliminary_source_family"] = wave["dataset_accession"].map(source_by_accession)
    wave["source_independence_status"] = "preliminary_requires_cohort_review"
    wave["selection_role"] = "metadata_only_compute_wave_candidate"

    duplicate_rows = []
    for fingerprint, block in wave.groupby("design_fingerprint", sort=True):
        accessions = sorted(block["dataset_accession"].unique())
        if len(accessions) > 1:
            duplicate_rows.append({
                "design_fingerprint": fingerprint,
                "n_accessions": len(accessions),
                "accessions": ";".join(accessions),
                "titles": ";".join(sorted(set(block["title"].astype(str)))[:10]),
                "decision": "collapse_pending_manual_cohort_review",
            })

    stratum_summary = (
        wave.groupby("comparison_stratum", as_index=False)
        .agg(
            n_accessions=("dataset_accession", "nunique"),
            n_preliminary_sources=("preliminary_source_family", "nunique"),
            n_panels=("contrast_id", "size"),
            total_units_a=("n_units_a", "sum"),
            total_units_b=("n_units_b", "sum"),
            total_ms2_a=("n_ms2_spectra_a", "sum"),
            total_ms2_b=("n_ms2_spectra_b", "sum"),
        )
        .sort_values(["n_preliminary_sources", "n_accessions", "total_units_a"], ascending=False)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wave.to_csv(args.output_dir / "compute_wave.tsv", sep="\t", index=False)
    stratum_summary.to_csv(args.output_dir / "stratum_summary.tsv", sep="\t", index=False)
    pd.DataFrame(duplicate_rows).to_csv(
        args.output_dir / "duplicate_source_suspects.tsv", sep="\t", index=False
    )
    report = {
        "schema": "pan_redu_systematic_compute_wave_v1",
        "selection": {
            "min_accessions_per_species_body_stratum": args.min_accessions_per_stratum,
            "feature_abundance_opened": False,
            "cross_study_truth_opened": False,
        },
        "denominators": {
            "panels": int(len(wave)),
            "dataset_accessions": int(wave["dataset_accession"].nunique()),
            "preliminary_source_families": int(wave["preliminary_source_family"].nunique()),
            "comparison_strata": int(wave["comparison_stratum"].nunique()),
            "reported_unit_identifier_appearances": int((wave["n_units_a"] + wave["n_units_b"]).sum()),
            "reported_ms2_spectra": int((wave["n_ms2_spectra_a"] + wave["n_ms2_spectra_b"]).sum()),
            "exact_design_duplicate_suspect_panels": int(wave["exact_design_duplicate_suspect"].sum()),
        },
        "claim_boundary": [
            "Repeated metadata labels are candidate retrieval relationships, not biological truth.",
            "Preliminary source families prevent obvious accession duplication but do not prove cohort independence.",
            "ReDU unit identifiers may encode visits, sites, or aliquots and are not guaranteed independent biological units.",
            "Unit appearances also count a unit again when it participates in multiple contrasts.",
            "Reported MS2 counts qualify spectral availability; they are not metabolite abundance.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["denominators"], indent=2))


if __name__ == "__main__":
    main()
