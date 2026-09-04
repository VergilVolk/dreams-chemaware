#!/usr/bin/env python
"""Test whether the MTBLS13729 LCAC class signal is broad or driver-specific."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_signflip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-12)
        total += 1
    return float(exceed / total)


def pqn_factors(log_matrix: pd.DataFrame) -> pd.Series:
    reference = log_matrix.median(axis=1, skipna=True)
    return log_matrix.sub(reference, axis=0).median(axis=0, skipna=True)


def paired_delta(matrix: pd.DataFrame, tumour_suffix: str, normal_suffix: str) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for patient_number in range(1, 31):
        patient = f"P{patient_number:02d}"
        tumour, normal = f"{patient}-{tumour_suffix}", f"{patient}-{normal_suffix}"
        if tumour in matrix and normal in matrix:
            columns[patient] = matrix[tumour] - matrix[normal]
    return pd.DataFrame(columns)


def patient_class_scores(delta: pd.DataFrame) -> np.ndarray:
    return delta.median(axis=0, skipna=True).to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-dir",
        type=Path,
        default=Path("data/mtbls13729/acylcarnitine_panel_20260829"),
    )
    parser.add_argument(
        "--auc",
        type=Path,
        default=Path("data/mtbls13729/ms1_eic_requant/pos_rp__eic_auc_matrix.csv.gz"),
    )
    parser.add_argument(
        "--detected",
        type=Path,
        default=Path("data/mtbls13729/ms1_eic_requant/pos_rp__eic_detection_matrix.csv.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/acylcarnitine_breadth_20260829"),
    )
    args = parser.parse_args()

    feature_path = args.class_dir / "acylcarnitine_class_score_features.csv"
    class_report_path = args.class_dir / "class_score_report.json"
    for path in (feature_path, class_report_path, args.auc, args.detected):
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    hypotheses = pd.read_csv(feature_path)
    selected = hypotheses[hypotheses["used_in_class_score"].astype(bool)].copy()
    representatives = (
        selected.sort_values(["n_samples_with_ms2", "n_ms2_spectra"], ascending=False)
        .drop_duplicates("feature_id", keep="first")
    )
    feature_ids = representatives["feature_id"].astype(int).tolist()
    if len(feature_ids) < 5:
        raise RuntimeError(f"too few independent LCAC features: {len(feature_ids)}")

    auc = pd.read_csv(args.auc).set_index("feature_id")
    detected = pd.read_csv(args.detected).set_index("feature_id").astype(bool)
    missing = sorted(set(feature_ids) - set(auc.index.astype(int)))
    if missing:
        raise RuntimeError(f"selected LCAC features absent from EIC matrix: {missing}")
    full = auc.where(detected & (auc > 0))
    positive = full.stack().to_numpy(float)
    pseudocount = float(np.percentile(positive, 1) / 2)
    log_raw = np.log2(full + pseudocount)
    pqn = log_raw.loc[feature_ids].sub(pqn_factors(log_raw), axis=1)
    rmu = paired_delta(pqn, "Rmu", "RN")
    rtu = paired_delta(pqn, "Rtu", "RN")
    if rmu.shape[1] != 10 or rtu.shape[1] != 10:
        raise RuntimeError(f"expected 10 Rmu and 10 Rtu pairs, observed {rmu.shape[1]} and {rtu.shape[1]}")

    feature_summary = representatives.set_index("feature_id")[
        ["acyl_chain", "adduct", "n_samples_with_ms2", "n_ms2_spectra"]
    ].copy()
    feature_summary["rmu_mean_log2fc"] = rmu.mean(axis=1)
    feature_summary["rmu_median_log2fc"] = rmu.median(axis=1)
    feature_summary["rmu_positive_pairs"] = (rmu > 0).sum(axis=1)
    feature_summary["rtu_mean_log2fc"] = rtu.mean(axis=1)
    feature_summary["rtu_median_log2fc"] = rtu.median(axis=1)
    feature_summary["rtu_positive_pairs"] = (rtu > 0).sum(axis=1)
    feature_summary = feature_summary.reset_index()

    rmu_scores = patient_class_scores(rmu)
    main_mean = float(np.mean(rmu_scores))
    archived = json.loads(class_report_path.read_text(encoding="utf-8"))
    expected = float(archived["variants"]["pqn"]["Rmu_vs_RN"]["mean_class_log2fc"])
    if not np.isclose(main_mean, expected, atol=1e-10, rtol=0):
        raise RuntimeError(f"PQN class-score reproduction mismatch: {main_mean} vs {expected}")

    leave_feature_rows = []
    for feature_id in feature_ids:
        retained = rmu.drop(index=feature_id)
        scores = patient_class_scores(retained)
        leave_feature_rows.append({
            "omitted_feature_id": feature_id,
            "remaining_features": int(len(retained)),
            "mean_class_log2fc": float(np.mean(scores)),
            "median_class_log2fc": float(np.median(scores)),
            "exact_signflip_p": exact_signflip_p(scores),
        })
    leave_feature = pd.DataFrame(leave_feature_rows)

    leave_patient_rows = []
    for index, patient in enumerate(rmu.columns):
        retained = np.delete(rmu_scores, index)
        leave_patient_rows.append({
            "omitted_patient": patient,
            "remaining_pairs": int(len(retained)),
            "mean_class_log2fc": float(np.mean(retained)),
            "median_class_log2fc": float(np.median(retained)),
            "exact_signflip_p": exact_signflip_p(retained),
        })
    leave_patient = pd.DataFrame(leave_patient_rows)

    feature_summary.to_csv(output / "feature_effects.csv", index=False)
    leave_feature.to_csv(output / "leave_one_feature_out.csv", index=False)
    leave_patient.to_csv(output / "leave_one_patient_out.csv", index=False)
    payload = {
        "status": "mtbls13729_acylcarnitine_breadth_audit_complete",
        "formal": False,
        "features": int(len(feature_ids)),
        "chains": int(representatives["acyl_chain"].nunique()),
        "rmu_pairs": int(rmu.shape[1]),
        "rmu_mean_class_log2fc": main_mean,
        "rmu_exact_signflip_p": exact_signflip_p(rmu_scores),
        "features_with_positive_rmu_median": int((feature_summary["rmu_median_log2fc"] > 0).sum()),
        "fraction_features_with_positive_rmu_median": float(
            (feature_summary["rmu_median_log2fc"] > 0).mean()
        ),
        "leave_one_feature_out_minimum_mean_log2fc": float(leave_feature["mean_class_log2fc"].min()),
        "leave_one_feature_out_maximum_exact_p": float(leave_feature["exact_signflip_p"].max()),
        "leave_one_patient_out_minimum_mean_log2fc": float(leave_patient["mean_class_log2fc"].min()),
        "leave_one_patient_out_maximum_exact_p": float(leave_patient["exact_signflip_p"].max()),
        "c20_4_feature_3222_in_class": bool(3222 in feature_ids),
        "gates": {
            "class_score_exactly_reproduced": True,
            "at_least_70pct_features_positive": bool(
                (feature_summary["rmu_median_log2fc"] > 0).mean() >= 0.70
            ),
            "positive_after_every_single_feature_omission": bool(
                (leave_feature["mean_class_log2fc"] > 0).all()
            ),
            "positive_after_every_single_patient_omission": bool(
                (leave_patient["mean_class_log2fc"] > 0).all()
            ),
        },
        "provenance": {
            "class_report_sha256": sha256(class_report_path),
            "feature_table_sha256": sha256(feature_path),
            "auc_sha256": sha256(args.auc),
            "detection_sha256": sha256(args.detected),
        },
        "claim_limit": (
            "Breadth and leave-one-out stability show whether the discovery is distributed across the "
            "measured LCAC panel. They do not provide external replication, flux, or MSI Level 1 identity."
        ),
    }
    payload["gates"]["pass"] = bool(all(payload["gates"].values()))
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
