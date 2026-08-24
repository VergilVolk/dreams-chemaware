"""Compare prioritized MTBLS13729 candidates with the authors' reported MAF.

The comparison is deliberately conservative: precursor m/z and retention time
must both agree.  It is an overlap audit, not an identification procedure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data/mtbls13729/biology_candidates/candidate_ms2_coverage.csv"
DEFAULT_MAF = ROOT / "_mtbls13729_meta/m_MTBLS13729_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv"
DEFAULT_OUT = ROOT / "data/mtbls13729/biology_candidates/author_maf_overlap.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--maf", type=Path, default=DEFAULT_MAF)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ppm", type=float, default=10.0)
    p.add_argument("--rt-min", type=float, default=0.30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    candidates = candidates.loc[
        (candidates["panel"] == "pos_rp") & (candidates["n_ms2_spectra"] > 0)
    ].copy()

    maf = pd.read_csv(args.maf, sep="\t")
    maf = maf[[
        "database_identifier",
        "metabolite_identification",
        "mass_to_charge",
        "retention_time",
        "reliability",
    ]].copy()
    maf["mass_to_charge"] = pd.to_numeric(maf["mass_to_charge"], errors="coerce")
    maf["retention_time"] = pd.to_numeric(maf["retention_time"], errors="coerce")
    maf = maf.dropna(subset=["mass_to_charge", "retention_time"])

    rows: list[dict] = []
    for candidate in candidates.itertuples(index=False):
        ppm = (maf["mass_to_charge"] - candidate.mz).abs() / candidate.mz * 1e6
        rt_delta = (maf["retention_time"] - candidate.rt_sec / 60.0).abs()
        valid = (ppm <= args.ppm) & (rt_delta <= args.rt_min)
        if valid.any():
            score = np.hypot(ppm / args.ppm, rt_delta / args.rt_min)
            best_idx = score.where(valid, np.inf).idxmin()
            hit = maf.loc[best_idx]
            rows.append({
                "feature_id": int(candidate.feature_id),
                "candidate_mz": float(candidate.mz),
                "candidate_rt_min": float(candidate.rt_sec / 60.0),
                "n_ms2_spectra": int(candidate.n_ms2_spectra),
                "n_samples_with_ms2": int(candidate.n_samples_with_ms2),
                "author_overlap": True,
                "author_name": hit["metabolite_identification"],
                "author_database_id": hit["database_identifier"],
                "author_reliability": hit["reliability"],
                "author_mz": float(hit["mass_to_charge"]),
                "author_rt_min": float(hit["retention_time"]),
                "ppm_error": float((hit["mass_to_charge"] - candidate.mz) / candidate.mz * 1e6),
                "rt_error_min": float(hit["retention_time"] - candidate.rt_sec / 60.0),
            })
        else:
            rows.append({
                "feature_id": int(candidate.feature_id),
                "candidate_mz": float(candidate.mz),
                "candidate_rt_min": float(candidate.rt_sec / 60.0),
                "n_ms2_spectra": int(candidate.n_ms2_spectra),
                "n_samples_with_ms2": int(candidate.n_samples_with_ms2),
                "author_overlap": False,
                "author_name": "",
                "author_database_id": "",
                "author_reliability": "",
                "author_mz": np.nan,
                "author_rt_min": np.nan,
                "ppm_error": np.nan,
                "rt_error_min": np.nan,
            })

    out = pd.DataFrame(rows).sort_values(
        ["author_overlap", "n_samples_with_ms2"], ascending=[False, False]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    report = {
        "n_ms2_covered_candidates": int(len(out)),
        "n_overlapping_author_maf": int(out["author_overlap"].sum()),
        "n_not_in_author_maf": int((~out["author_overlap"]).sum()),
        "criteria": {"precursor_ppm": args.ppm, "rt_tolerance_min": args.rt_min},
        "warning": "Mass/RT overlap is not an independent structural identification.",
        "output": str(args.out),
    }
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["n_overlapping_author_maf"]:
        print(out.loc[out["author_overlap"], [
            "feature_id", "candidate_mz", "candidate_rt_min", "author_name",
            "author_database_id", "ppm_error", "rt_error_min"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
