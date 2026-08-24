"""Map the fixed published HNSCC target panel to public QC-DDA spectra.

This is not an annotation result.  It defines which published metabolites
actually have a directly observed MS2 spectrum in the same public QC-DDA
acquisition, using predeclared 10 ppm precursor and 15 second RT windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=Path("data/external/MTBLS1905/metadata/published_independent_targets_s1.tsv"))
    parser.add_argument("--inventory", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/qc_ms2_scan_inventory.tsv"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/published_target_qc_ms2_matches.tsv"))
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-sec", type=float, default=15.0)
    parser.add_argument("--ion-mode", default="POSITIVE", help="Acquisition polarity of the QC-DDA files")
    args = parser.parse_args()
    targets = pd.read_csv(args.targets, sep="\t")
    targets = targets[targets["Ion mode"].str.upper().eq(args.ion_mode.upper())].copy()
    inventory = pd.read_csv(args.inventory)
    rows: list[dict] = []
    for _, target in targets.iterrows():
        ppm = np.abs(inventory["precursor_mz"] - target["precursor_mz"]) / target["precursor_mz"] * 1e6
        rt_sec = np.abs(inventory["rt_min"] * 60.0 - target["rt_sec"])
        hits = inventory[(ppm <= args.ppm) & (rt_sec <= args.rt_sec)].copy()
        if hits.empty:
            rows.append({
                "metabolite": target["Metabolite annotation"], "xcms_feature": target["Peak XCMS identifier"],
                "published_msi": target["msi_level"], "target_mz": target["precursor_mz"], "target_rt_sec": target["rt_sec"],
                "n_qc_ms2_hits": 0,
            })
            continue
        for idx, hit in hits.iterrows():
            rows.append({
                "metabolite": target["Metabolite annotation"], "xcms_feature": target["Peak XCMS identifier"],
                "published_msi": target["msi_level"], "target_mz": target["precursor_mz"], "target_rt_sec": target["rt_sec"],
                "n_qc_ms2_hits": int(len(hits)), "source_file": hit.source_file, "spectrum_id": hit.spectrum_id,
                "query_precursor_mz": hit.precursor_mz, "query_rt_min": hit.rt_min,
                "mass_error_ppm": float(ppm.loc[idx]), "rt_error_sec": float(rt_sec.loc[idx]), "n_raw_peaks": int(hit.n_peaks),
            })
    matched = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.out, sep="\t", index=False)
    coverage = int((matched.groupby("metabolite")["n_qc_ms2_hits"].max() > 0).sum())
    report = {"published_independent_targets": int(len(targets)), "targets_with_direct_qc_ms2": coverage, "ppm": args.ppm, "rt_sec": args.rt_sec, "ion_mode": args.ion_mode.upper()}
    (args.out.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
