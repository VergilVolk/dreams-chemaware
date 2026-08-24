"""Make a download manifest for HNSCC biological positive-mode mzML files.

Only patient C/M/N samples in the ISA sample sheet are retained; blanks and QC
are deliberately excluded here because they have separate explicit roles in
feature filtering.  The manifest is the auditable bridge between public data
and the future MS1 feature matrix.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("data/external/MTBLS1905/metadata/samples.tsv"))
    parser.add_argument("--directory-index", type=Path, default=Path("data/external/MTBLS1905/positive_directory.html"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/metadata/positive_patient_download_manifest.tsv"))
    args = parser.parse_args()
    samples = pd.read_csv(args.samples, sep="\t")
    patient = samples[samples["Factor Value[Tissue type]"].notna()].copy()
    patient = patient[["Sample Name", "Factor Value[Tissue type]"]].rename(columns={"Sample Name": "sample_name", "Factor Value[Tissue type]": "tissue_type"})
    available = set(re.findall(r'href="([^"/]+\.mzML)"', args.directory_index.read_text(encoding="utf-8", errors="replace")))
    patient["file_name"] = patient.sample_name.astype(str) + ".mzML"
    patient["available"] = patient.file_name.isin(available)
    if not patient.available.all():
        missing = patient.loc[~patient.available, "file_name"].tolist()
        raise ValueError(f"Patient files absent from public positive directory: {missing}")
    patient["url"] = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1905/FILES/DERIVED_FILES/HILIC_positive/" + patient.file_name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    patient.to_csv(args.out, sep="\t", index=False)
    counts = patient.tissue_type.value_counts().to_dict()
    print(f"Prepared {len(patient)} positive patient mzML downloads: {counts}")


if __name__ == "__main__":
    main()
