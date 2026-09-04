#!/usr/bin/env python
"""Frozen sensitivity analysis for the OEP00006137 raw EIC re-extraction.

The primary extraction remains 5 ppm / +/-15 s.  Wider windows are reported
only as a retention-time-drift sensitivity analysis.  Missing peak areas are
handled with three pre-specified views: complete cases, one-half of the global
minimum positive area, and the global minimum positive area.  No result is
selected post hoc as the preferred biological estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RUNS = {
    "5ppm_15s_primary": "modified_guanosine_raw_reextraction_v1",
    "10ppm_15s_mass_sensitivity": "modified_guanosine_raw_reextraction_10ppm_15s",
    "5ppm_30s_rt_sensitivity": "modified_guanosine_raw_reextraction_5ppm_30s",
    "10ppm_30s_joint_sensitivity": "modified_guanosine_raw_reextraction_10ppm_30s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/external/OEP00006137_raw")
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path(
            "data/external/OEP00006137_support/modified_guanosine_level1_rows.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/modified_guanosine_raw_sensitivity_v1"
        ),
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def paired_stats(delta: np.ndarray) -> dict:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return {"n": 0}
    ttest = stats.ttest_1samp(delta, 0.0)
    try:
        wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    positive = int(np.sum(delta > 0))
    nonzero = int(np.sum(delta != 0))
    return {
        "n": int(delta.size),
        "mean_log2fc": float(np.mean(delta)),
        "median_log2fc": float(np.median(delta)),
        "positive_pairs": positive,
        "negative_pairs": int(np.sum(delta < 0)),
        "paired_t_p": float(ttest.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": (
            float(stats.binomtest(positive, nonzero, 0.5).pvalue) if nonzero else 1.0
        ),
    }


def paired_table(frame: pd.DataFrame, subtype: str) -> pd.DataFrame:
    subset = frame.loc[frame["subtype"].eq(subtype), ["patient", "tissue", "area"]]
    return subset.pivot(index="patient", columns="tissue", values="area").dropna()


def detection_audit(pivot: pd.DataFrame) -> dict:
    detected_n = pivot["N"].to_numpy() > 0
    detected_t = pivot["T"].to_numpy() > 0
    t_only = int(np.sum(detected_t & ~detected_n))
    n_only = int(np.sum(detected_n & ~detected_t))
    discordant = t_only + n_only
    return {
        "complete_pairs": int(len(pivot)),
        "both_detected": int(np.sum(detected_n & detected_t)),
        "tumor_only_detected": t_only,
        "normal_only_detected": n_only,
        "neither_detected": int(np.sum(~detected_n & ~detected_t)),
        "paired_detection_mcnemar_exact_p": (
            float(stats.binomtest(t_only, discordant, 0.5).pvalue)
            if discordant else 1.0
        ),
    }


def sensitivity_views(frame: pd.DataFrame, subtype: str) -> dict:
    pivot = paired_table(frame, subtype)
    positive = frame.loc[frame["area"] > 0, "area"].to_numpy(dtype=float)
    if positive.size == 0:
        raise RuntimeError("target has no positive raw areas")
    minimum = float(np.min(positive))
    complete = pivot.loc[(pivot["N"] > 0) & (pivot["T"] > 0)]
    views = {
        "complete_case": paired_stats(
            np.log2(complete["T"].to_numpy() / complete["N"].to_numpy())
        )
    }
    for name, replacement in (
        ("half_global_minimum", 0.5 * minimum),
        ("global_minimum", minimum),
    ):
        filled = pivot.mask(pivot <= 0, replacement)
        views[name] = paired_stats(
            np.log2(filled["T"].to_numpy() / filled["N"].to_numpy())
        )
        views[name]["replacement_area"] = replacement
    return {"detection": detection_audit(pivot), "abundance_views": views}


def published_effects(payload: dict, target_id: str, subtype: str) -> dict:
    rows = [
        row for row in payload["rows"]
        if str(row["Peak name"]).removesuffix("_a").removesuffix("_b") == target_id
    ]
    if not rows:
        raise RuntimeError(f"supplement target missing: {target_id}")
    row = rows[0]
    deltas = []
    for patient in range(1, 21):
        if subtype == "MSI-H":
            normal, tumor = row.get(f"MSI_N{patient}"), row.get(f"MSI.H_T{patient}")
        else:
            normal, tumor = row.get(f"MSS_N{patient}"), row.get(f"MSS_T{patient}")
        if normal is not None and tumor is not None and float(normal) > 0 and float(tumor) > 0:
            deltas.append(np.log2(float(tumor) / float(normal)))
    return paired_stats(np.asarray(deltas, dtype=float))


def main() -> None:
    args = parse_args()
    supplement = json.loads(args.supplement.read_text(encoding="utf-8"))
    window_rows = []
    run_payload = {}
    eic_hashes = {}
    recovered_rows = []
    primary_frame = None
    for run_name, directory in RUNS.items():
        base = args.raw_dir / directory
        summary_path = base / "summary.json"
        eic_path = base / "target_eic.csv.gz"
        if not summary_path.exists() or not eic_path.exists():
            raise FileNotFoundError(f"missing frozen extraction run: {base}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(eic_path)
        if run_name == "5ppm_15s_primary":
            primary_frame = frame.copy()
        run_payload[run_name] = summary
        eic_hashes[run_name] = sha256sum(eic_path)
        for target_id, target in summary["targets"].items():
            for subtype in ("MSI-H", "MSS"):
                effect = target[subtype]
                window_rows.append(
                    {
                        "run": run_name,
                        "ppm": summary["parameters"]["ppm"],
                        "rt_half_window_sec": summary["parameters"]["rt_half_window_sec"],
                        "target_id": target_id,
                        "subtype": subtype,
                        "detected_biological_samples": target["detected_biological_samples"],
                        **effect,
                        "published_spearman": summary["published_vs_reextracted"][target_id]["spearman_rho_log_area"],
                    }
                )

    if primary_frame is None:
        raise RuntimeError("primary extraction was not loaded")
    primary_m298 = primary_frame.loc[primary_frame["target_id"].eq("M298T55")]
    rt_m298 = pd.read_csv(
        args.raw_dir / RUNS["5ppm_30s_rt_sensitivity"] / "target_eic.csv.gz"
    )
    rt_m298 = rt_m298.loc[rt_m298["target_id"].eq("M298T55")]
    recovered = primary_m298[["sample", "area"]].rename(columns={"area": "primary_area"}).merge(
        rt_m298[["sample", "area", "apex_rt_sec", "apex_ppm"]].rename(
            columns={"area": "rt_sensitivity_area"}
        ),
        on="sample",
        how="inner",
    )
    recovered = recovered.loc[
        (recovered["primary_area"] <= 0) & (recovered["rt_sensitivity_area"] > 0)
    ].copy()
    recovered_rows = recovered.to_dict(orient="records")

    sensitivity = {}
    for run_name in RUNS:
        frame = pd.read_csv(
            args.raw_dir / RUNS[run_name] / "target_eic.csv.gz"
        )
        target = frame.loc[
            frame["target_id"].eq("M298T55") & frame["subtype"].notna()
        ].copy()
        sensitivity[run_name] = {
            subtype: sensitivity_views(target, subtype)
            for subtype in ("MSI-H", "MSS")
        }
    sensitivity["published_supplement"] = {
        subtype: published_effects(supplement, "M298T55", subtype)
        for subtype in ("MSI-H", "MSS")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_path = args.output_dir / "window_sensitivity.csv"
    recovery_path = args.output_dir / "m298_rt_recovery_samples.csv"
    pd.DataFrame(window_rows).to_csv(window_path, index=False)
    pd.DataFrame(recovered_rows).to_csv(recovery_path, index=False)
    report = {
        "status": "OEP00006137_modified_guanosine_sensitivity_complete",
        "formal": True,
        "primary_extraction": "5 ppm / +/-15 s",
        "m298t55_sensitivity": sensitivity,
        "m298t55_primary_zero_recovered_by_5ppm_30s": len(recovered_rows),
        "interpretation": (
            "Mass-window widening has little effect, whereas RT-window widening recovers "
            "most M298T55 non-detects and preserves a negative MSI-H direction. The wider "
            "window is sensitivity-only because it can merge neighboring isomeric peaks, "
            "as directly observed for the two m/z 296.1 targets."
        ),
        "provenance": {
            "supplement_sha256": sha256sum(args.supplement),
            "eic_sha256": eic_hashes,
            "window_csv_sha256": sha256sum(window_path),
            "recovery_csv_sha256": sha256sum(recovery_path),
        },
        "claim_limit": (
            "The sensitivity analysis bounds abundance effects under fixed extraction and "
            "left-censoring choices. It does not upgrade positional-isomer identity, and the "
            "wide-window result is not substituted for the primary analysis."
        ),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
