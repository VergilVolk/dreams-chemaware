"""Reconcile complete-detection and legacy pseudocount abundance protocols.

The legacy convergent-module table used log2(AUC + 1) for every patient pair.
That is harmless when both peaks are detected, but it converts an undetected
normal peak into a numerical zero and can create a very large fold change.
This audit freezes complete-detection estimates for the five source-table-
absent candidates and creates a non-destructive corrected integrated ledger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
OUT = BASE / "candidate_abundance_protocol_audit_v1"
CORRECTED = BASE / "integrated_biology_ledger_v3"
FEATURES = {150: "full_space_eic_v1", 1597: "biology_closure_eic_v1", 1717: "biology_closure_eic_v1", 3019: "biology_closure_eic_v1", 3222: "biology_closure_eic_v1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(values: np.ndarray, seed: int, repeats: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    for output in (OUT, CORRECTED):
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
        output.mkdir(parents=True, exist_ok=True)

    patient_rows: list[dict] = []
    summary_rows: list[dict] = []
    for feature_id, directory in FEATURES.items():
        auc_path = BASE / directory / "pos_rp__eic_auc_matrix.csv.gz"
        detection_path = BASE / directory / "pos_rp__eic_detection_matrix.csv.gz"
        auc = pd.read_csv(auc_path).set_index("feature_id")
        detection = pd.read_csv(detection_path).set_index("feature_id")
        if feature_id not in auc.index or feature_id not in detection.index:
            raise RuntimeError(f"feature {feature_id} missing from {directory}")
        values = auc.loc[feature_id]
        flags = detection.loc[feature_id].astype(bool)
        for number in range(21, 31):
            patient = f"P{number:02d}"
            tumour_col, normal_col = f"{patient}-Rmu", f"{patient}-RN"
            tumour = float(values[tumour_col])
            normal = float(values[normal_col])
            complete = bool(flags[tumour_col] and flags[normal_col] and tumour > 0 and normal > 0)
            legacy = float(np.log2(tumour + 1.0) - np.log2(normal + 1.0)) if np.isfinite(tumour) and np.isfinite(normal) else np.nan
            complete_delta = float(np.log2(tumour) - np.log2(normal)) if complete else np.nan
            patient_rows.append(
                {
                    "feature_id": feature_id,
                    "patient": patient,
                    "tumour_auc": tumour,
                    "normal_auc": normal,
                    "tumour_detected": bool(flags[tumour_col]),
                    "normal_detected": bool(flags[normal_col]),
                    "complete_detection_pair": complete,
                    "legacy_log2_auc_plus_1_delta": legacy,
                    "complete_detection_log2_delta": complete_delta,
                }
            )

    detail = pd.DataFrame(patient_rows)
    for feature_id, block in detail.groupby("feature_id", sort=True):
        legacy = block.legacy_log2_auc_plus_1_delta.dropna().to_numpy(float)
        complete = block.complete_detection_log2_delta.dropna().to_numpy(float)
        ci_low, ci_high = bootstrap(complete, 20260901 + int(feature_id))
        missing = block.loc[~block.complete_detection_pair, "patient"].tolist()
        summary_rows.append(
            {
                "feature_id": int(feature_id),
                "legacy_pairs": int(len(legacy)),
                "legacy_mean_log2fc": float(legacy.mean()),
                "complete_detection_pairs": int(len(complete)),
                "complete_detection_mean_log2fc": float(complete.mean()),
                "complete_detection_positive_pairs": int((complete > 0).sum()),
                "complete_detection_bootstrap_ci_low": ci_low,
                "complete_detection_bootstrap_ci_high": ci_high,
                "complete_detection_one_sided_sign_p": float(
                    binomtest(int((complete > 0).sum()), len(complete), 0.5, alternative="greater").pvalue
                ),
                "incomplete_patients": ";".join(missing),
                "legacy_minus_complete_mean": float(legacy.mean() - complete.mean()),
                "material_protocol_difference": bool(abs(legacy.mean() - complete.mean()) > 0.5),
            }
        )
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(OUT / "candidate_abundance_patient_protocols.csv", index=False)
    summary.to_csv(OUT / "candidate_abundance_protocol_summary.csv", index=False)

    material = summary.loc[summary.material_protocol_difference, "feature_id"].astype(int).tolist()
    if material != [1717]:
        raise RuntimeError(f"expected only feature 1717 to have a material protocol difference, got {material}")
    f1717 = detail.loc[detail.feature_id.eq(1717)]
    offending = f1717.loc[~f1717.complete_detection_pair, "patient"].tolist()
    if offending != ["P28"]:
        raise RuntimeError(f"feature 1717 discrepancy is not isolated to P28: {offending}")

    old_path = BASE / "integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
    corrected = pd.read_csv(old_path)
    corrected["abundance_protocol_v3"] = "unchanged from v2; protocol-compatible"
    s1717 = summary.loc[summary.feature_id.eq(1717)].iloc[0]
    mask = corrected.feature_id.eq(1717)
    if mask.sum() != 1:
        raise RuntimeError("feature 1717 is not unique in integrated ledger v2")
    corrected.loc[mask, "pairs"] = int(s1717.complete_detection_pairs)
    corrected.loc[mask, "mean_log2fc"] = float(s1717.complete_detection_mean_log2fc)
    corrected.loc[mask, "positive_pairs"] = int(s1717.complete_detection_positive_pairs)
    corrected.loc[mask, "abundance_bootstrap_ci_low"] = float(s1717.complete_detection_bootstrap_ci_low)
    corrected.loc[mask, "abundance_bootstrap_ci_high"] = float(s1717.complete_detection_bootstrap_ci_high)
    corrected.loc[mask, "one_sided_sign_p"] = float(s1717.complete_detection_one_sided_sign_p)
    corrected.loc[mask, "abundance_protocol_v3"] = "complete-detection pairs; P28 excluded because RN peak was undetected"
    corrected.to_csv(CORRECTED / "integrated_candidate_ledger_v3.csv", index=False)

    report = {
        "status": "mtbls13729_candidate_abundance_protocol_audit_complete",
        "formal": True,
        "features": int(len(summary)),
        "material_protocol_difference_features": material,
        "feature_1717_resolution": {
            "legacy": "10/10 positive, mean +4.817 log2 after treating an undetected P28 normal peak as numerical zero",
            "accepted": "9/9 complete-detection pairs, mean +3.009 log2",
            "cause": "legacy log2(AUC+1) included P28 despite an explicitly false RN detection flag, creating a +21.088 log2 pair",
            "decision": "use the complete-detection estimate; retain the biological direction but discard the inflated effect size",
        },
        "corrected_ledger": str((CORRECTED / "integrated_candidate_ledger_v3.csv").relative_to(ROOT)),
        "provenance": {
            "integrated_v2_sha256": sha256(old_path),
            "detail_sha256": sha256(OUT / "candidate_abundance_patient_protocols.csv"),
            "summary_sha256": sha256(OUT / "candidate_abundance_protocol_summary.csv"),
            "corrected_ledger_sha256": sha256(CORRECTED / "integrated_candidate_ledger_v3.csv"),
        },
        "claim_limit": "This corrects an abundance computation protocol. It does not improve structural identity, external replication, full-space FDR, flux or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (CORRECTED / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
