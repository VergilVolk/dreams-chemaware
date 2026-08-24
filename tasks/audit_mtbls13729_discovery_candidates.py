"""Audit discovery MS1 features before assigning biological identities.

This is a quality/sensitivity audit, not an annotation step.  It combines the
uniform EIC re-quantification output with acquisition order and reports whether
an apparent paired effect is vulnerable to low S/N, RT instability, or run-order
confounding.  The subtype blocks in MTBLS13729 are not randomized, so the audit
never declares a feature biologically validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sample_suffix(sample: str) -> str:
    return sample.split("-", 1)[1]


def paired_rows(frame: pd.DataFrame, tumor_suffix: str, normal_suffix: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for patient, group in frame.groupby("patient"):
        tumor = group[group["suffix"] == tumor_suffix]
        normal = group[group["suffix"] == normal_suffix]
        if len(tumor) != 1 or len(normal) != 1:
            continue
        t, n = tumor.iloc[0], normal.iloc[0]
        if not (bool(t.detected_eic) and bool(n.detected_eic)):
            continue
        rows.append(
            {
                "patient": patient,
                "log2_auc_delta": float(np.log2(t.eic_auc) - np.log2(n.eic_auc)),
                "apex_rt_delta_sec": float(t.eic_apex_rt - n.eic_apex_rt),
                "pair_center_order": float((t.injection_order + n.injection_order) / 2),
            }
        )
    return pd.DataFrame(rows)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    if valid.sum() < 5 or np.unique(x.to_numpy(float)[valid]).size < 2:
        return np.nan, np.nan
    result = spearmanr(x.to_numpy(float)[valid], y.to_numpy(float)[valid])
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="neg_rp")
    parser.add_argument(
        "--priority-table",
        type=Path,
        default=Path("data/mtbls13729/ms1_paired_analysis/neg_rp__discovery_priority_features.csv"),
    )
    parser.add_argument("--eic-dir", type=Path, default=Path("data/mtbls13729/ms1_eic_requant/per_sample"))
    parser.add_argument(
        "--acquisition-audit",
        type=Path,
        default=Path("data/mtbls13729/ms1_acquisition_audit/file_audit.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/discovery_candidate_audit"),
    )
    args = parser.parse_args()

    priority = pd.read_csv(args.priority_table)
    feature_ids = set(priority.feature_id.astype(int))
    acquisition = pd.read_csv(args.acquisition_audit)
    acquisition = acquisition[acquisition.panel == args.panel][["sample_name", "patient", "injection_order"]]

    frames = []
    for path in sorted(args.eic_dir.glob(f"{args.panel}__*__eic.csv.gz")):
        frame = pd.read_csv(path)
        frame = frame[frame.feature_id.astype(int).isin(feature_ids)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise SystemExit(f"No EIC rows found for {args.panel} and {len(feature_ids)} priority features")

    eic = pd.concat(frames, ignore_index=True)
    eic = eic.merge(acquisition, on="sample_name", how="left", validate="many_to_one")
    eic["suffix"] = eic.sample_name.map(sample_suffix)
    eic["log2_auc"] = np.log2(eic.eic_auc.where(eic.eic_auc > 0))

    summary_rows: list[dict[str, object]] = []
    pair_rows = []
    for feature_id, frame in eic.groupby("feature_id"):
        detected = frame[frame.detected_eic.astype(bool)].copy()
        order_rho, order_p = safe_spearman(detected.injection_order, detected.log2_auc)
        rmu = paired_rows(frame, "Rmu", "RN")
        rtu = paired_rows(frame, "Rtu", "RN")
        for label, pairs in (("Rmu_RN", rmu), ("Rtu_RN", rtu)):
            if not pairs.empty:
                pairs.insert(0, "feature_id", int(feature_id))
                pairs.insert(1, "comparison", label)
                pair_rows.append(pairs)

        rmu_order_rho, rmu_order_p = (
            safe_spearman(rmu.pair_center_order, rmu.log2_auc_delta) if not rmu.empty else (np.nan, np.nan)
        )
        rt_abs = np.abs(detected.eic_apex_delta_sec.to_numpy(float))
        # The simple EIC S/N estimator can return an inflated value when the
        # local baseline is exactly zero.  Report that failure mode explicitly
        # and only summarize S/N where a non-zero baseline was estimable.
        nonzero_baseline = detected[detected.eic_baseline > 0]
        zero_baseline_fraction = float((detected.eic_baseline <= 0).mean()) if len(detected) else np.nan
        median_snr = float(np.nanmedian(nonzero_baseline.eic_snr)) if len(nonzero_baseline) else np.nan
        rt_p90 = float(np.nanpercentile(rt_abs, 90)) if len(rt_abs) else np.nan
        summary_rows.append(
            {
                "feature_id": int(feature_id),
                "mz": float(priority.loc[priority.feature_id == feature_id, "mz"].iloc[0]),
                "rt_sec": float(priority.loc[priority.feature_id == feature_id, "rt_sec"].iloc[0]),
                "n_samples": int(len(frame)),
                "n_detected": int(len(detected)),
                "median_snr": median_snr,
                "zero_baseline_fraction": zero_baseline_fraction,
                "apex_abs_delta_p90_sec": rt_p90,
                "run_order_spearman": order_rho,
                "run_order_p": order_p,
                "rmu_n_pairs": int(len(rmu)),
                "rmu_mean_log2fc": float(rmu.log2_auc_delta.mean()) if not rmu.empty else np.nan,
                "rmu_mean_abs_pair_rt_delta_sec": float(rmu.apex_rt_delta_sec.abs().mean()) if not rmu.empty else np.nan,
                "rmu_delta_order_spearman": rmu_order_rho,
                "rmu_delta_order_p": rmu_order_p,
                "rt_instability_flag": bool(np.isfinite(rt_p90) and rt_p90 > 10.0),
                "low_snr_flag": bool(np.isfinite(median_snr) and median_snr < 3.0),
                "snr_unreliable_flag": bool(np.isfinite(zero_baseline_fraction) and zero_baseline_fraction > 0.5),
                "global_run_order_flag": bool(np.isfinite(order_rho) and abs(order_rho) >= 0.5 and order_p < 0.05),
                "subtype_block_confounded": True,
            }
        )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values("feature_id")
    pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    summary_path = out / f"{args.panel}__candidate_quality_summary.csv"
    pairs_path = out / f"{args.panel}__candidate_pair_details.csv"
    summary.to_csv(summary_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    report = {
        "status": "complete",
        "panel": args.panel,
        "n_candidates": int(len(summary)),
        "summary": str(summary_path),
        "pair_details": str(pairs_path),
        "interpretation_limit": (
            "Subtype blocks and injection order are partially confounded. Flags identify vulnerable candidates; "
            "absence of a flag does not establish biological causality."
        ),
    }
    (out / f"{args.panel}__report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
