"""Export independent published HNSCC annotations as a fixed evaluation panel.

The source workbook is Supplementary Table S1 of Southam et al. (2026).  We
retain exactly one row with an explicit MSI annotation per named metabolite;
this avoids treating isotopes/adducts as independent biological discoveries.

Run with the bundled workspace Python because it includes openpyxl::

    <bundled-python> tasks/export_mtbls1905_published_targets.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/external/MTBLS1905/metadata/southam_2026_table_s1.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/metadata/published_independent_targets_s1.tsv"))
    args = parser.parse_args()
    source = pd.read_excel(args.source, header=2)
    targets = source[source["MSI level of the annotation"].notna()].copy()
    keep = [
        "Peak XCMS identifier", "Ion mode", "Metabolite annotation", "Metabolite group", "Ionform",
        "m/z in HNSCC data", "HNSCC retention time [RT] (s)", "MSI level of the annotation",
    ]
    targets = targets[keep].rename(columns={
        "m/z in HNSCC data": "precursor_mz", "HNSCC retention time [RT] (s)": "rt_sec",
        "MSI level of the annotation": "msi_level",
    })
    duplicated = targets["Metabolite annotation"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError("Published MSI-bearing target names must be unique; inspect source table")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out, sep="\t", index=False)
    print(f"Exported {len(targets)} independent published targets to {args.out}")


if __name__ == "__main__":
    main()
