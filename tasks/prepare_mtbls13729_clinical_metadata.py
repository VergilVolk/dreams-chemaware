"""Convert the original paper's Table S2 into sample-level analysis metadata.

The source workbook uses merged cells: pathological type is stated once per
10-patient block, and tumour molecular annotations apply to its matched normal.
This script makes that inheritance explicit and audits the published tissue
number against the local mzML-to-sample map before downstream subgroup tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/mtbls13729/pr5c01260_si_003.xlsx"),
    )
    parser.add_argument(
        "--sample-map", type=Path, default=Path("data/mtbls13729/sample_map.tsv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/mtbls13729/clinical_metadata_s2.tsv")
    )
    args = parser.parse_args()

    source = pd.read_excel(args.source, header=1)
    source = source.dropna(how="all").rename(
        columns={
            "Tissue number": "tissue_number",
            "Patient number": "patient_number",
            "Sample type": "sample_type",
            "Pathological type": "pathology",
            " BRAF mutation": "braf",
            "MMR status": "mmr",
        }
    )
    required = {"tissue_number", "patient_number", "sample_type", "pathology", "braf", "mmr"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Unexpected Table S2 columns; missing {sorted(missing)}")

    # The sheet deliberately leaves cells blank to represent merged values.
    source["pathology"] = source["pathology"].ffill()
    source["patient_number"] = source["patient_number"].ffill()
    source["braf"] = source["braf"].ffill()
    source["mmr"] = source["mmr"].ffill()
    source["tissue_number"] = source["tissue_number"].astype(int)
    source["patient_number"] = source["patient_number"].astype(int)

    pathology_to_suffix = {"Ltu": "Ltu", "Rtu": "Rtu", "Rmu": "Rmu"}
    if not set(source.pathology).issubset(pathology_to_suffix):
        raise ValueError(f"Unexpected pathology labels: {sorted(source.pathology.unique())}")
    suffix = source.pathology.map(pathology_to_suffix)
    normal_suffix = suffix.map({"Ltu": "LN", "Rtu": "RN", "Rmu": "RN"})
    source["sample_name"] = [
        f"P{patient:02d}-{tumour if sample_type == 'Tissue-tumor' else normal}"
        for patient, tumour, normal, sample_type in zip(
            source.patient_number, suffix, normal_suffix, source.sample_type
        )
    ]
    source["location"] = source.pathology.map({"Ltu": "Left", "Rtu": "Right", "Rmu": "Right"})
    source["histology"] = source.pathology.map({"Ltu": "Tubular", "Rtu": "Tubular", "Rmu": "Mucinous"})
    source["tissue"] = source.sample_type.map({"Tissue-tumor": "Tumor", "Tissue-normal": "Normal"})
    source = source[
        [
            "tissue_number",
            "patient_number",
            "sample_name",
            "tissue",
            "pathology",
            "location",
            "histology",
            "braf",
            "mmr",
        ]
    ]

    # ``sample_map.tsv`` is an intentionally headerless two-column lookup.
    sample_map = pd.read_csv(args.sample_map, sep="\t", header=None, names=["mzml_file", "sample_name_map"])
    numeric_name = source.tissue_number.astype(str) + ".mzML"
    audited = source.assign(mzml_file=numeric_name).merge(sample_map, on="mzml_file", how="left", suffixes=("", "_map"))
    mismatch = audited.sample_name != audited.sample_name_map
    if audited.sample_name_map.isna().any() or mismatch.any():
        bad = audited.loc[audited.sample_name_map.isna() | mismatch, ["tissue_number", "sample_name", "sample_name_map"]]
        raise ValueError(f"Table S2 does not agree with local sample map:\n{bad.to_string(index=False)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output, sep="\t", index=False)
    print(f"Saved {args.output} ({len(source)} samples; {source.patient_number.nunique()} matched patients)")
    print("Tumour pathology × MMR")
    print(source[source.tissue == "Tumor"].groupby(["pathology", "mmr"]).size().to_string())
    print("Tumour pathology × BRAF")
    print(source[source.tissue == "Tumor"].groupby(["pathology", "braf"]).size().to_string())


if __name__ == "__main__":
    main()
