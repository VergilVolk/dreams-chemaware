"""Build the declared positive-mode MS1 processing manifest for MTBLS1905.

The patient samples, pooled QC injections, and extraction blanks are kept as
distinct roles.  This prevents the common but invalid shortcut of treating QC
spectra as patient replicates, while retaining the QC/blank information needed
for the paper's stated feature-quality filters.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1905/FILES/DERIVED_FILES/HILIC_positive/"


def role(row: pd.Series) -> str | None:
    tissue = row.get("Factor Value[Tissue type]")
    if pd.notna(tissue):
        return "patient"
    kind = str(row.get("Characteristics[Sample type]", ""))
    name = str(row.get("Sample Name", ""))
    if "pooled quality control" in kind and re.fullmatch(r"QC\d+", name):
        return "pooled_qc"
    if "blank" in kind:
        return "blank"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("data/external/MTBLS1905/metadata/samples.tsv"))
    parser.add_argument("--directory-index", type=Path, default=Path("data/external/MTBLS1905/positive_directory.html"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/metadata/positive_ms1_processing_manifest.tsv"))
    args = parser.parse_args()

    samples = pd.read_csv(args.samples, sep="\t")
    samples["sample_role"] = samples.apply(role, axis=1)
    kept = samples[samples["sample_role"].notna()].copy()
    kept = kept[["Sample Name", "Factor Value[Tissue type]", "sample_role"]].rename(
        columns={"Sample Name": "sample_name", "Factor Value[Tissue type]": "tissue_type"}
    )
    available = set(re.findall(r'href="([^"/]+\.mzML)"', args.directory_index.read_text(encoding="utf-8", errors="replace")))
    kept["file_name"] = kept["sample_name"].astype(str) + ".mzML"
    kept["available"] = kept["file_name"].isin(available)
    missing = kept.loc[~kept["available"], "file_name"].tolist()
    if missing:
        raise ValueError(f"Processing files absent from HILIC-positive index: {missing}")
    kept["url"] = BASE_URL + kept["file_name"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.out, sep="\t", index=False)
    print(f"Prepared {len(kept)} files: {kept.sample_role.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
