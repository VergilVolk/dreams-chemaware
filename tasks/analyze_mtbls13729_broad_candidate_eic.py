from __future__ import annotations

"""Analyze the frozen six-candidate raw-file targeted-EIC re-extraction.

The discovery matrix is used only to estimate sample-level PQN shifts.  All
candidate abundance values are newly integrated from mzML.  As candidates were
selected in the same biological cohort, this is a technical re-extraction
check, not independent biological validation.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, ttest_ind, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
EIC = ROOT / "data/mtbls13729/broad_candidate_eic_v1"
TARGETS = ROOT / "data/mtbls13729/broad_candidate_eic_targets_v1"
DISCOVERY = ROOT / "data/mtbls13729/ms1_consensus"
NOVELTY = ROOT / "data/mtbls13729/broad_candidate_novelty_audit_v1/candidate_original_paper_overlap.csv"
OUT = ROOT / "data/mtbls13729/broad_candidate_eic_analysis_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(series: pd.Series) -> pd.Series:
    values = series.to_numpy(float)
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    if not len(p):
        return pd.Series(out, index=series.index)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    out[np.flatnonzero(valid)[order]] = np.clip(adjusted, 0, 1)
    return pd.Series(out, index=series.index)


def discovery_pqn_factors(panel: str) -> pd.Series:
    matrix = pd.read_csv(
        DISCOVERY / f"{panel}__discovery_intensity_matrix.csv.gz"
    ).set_index("feature_id")
    log_positive = np.log2(matrix.where(matrix > 0))
    keep = log_positive.notna().mean(axis=1).ge(0.60)
    reference = log_positive.loc[keep].median(axis=1, skipna=True)
    return log_positive.loc[keep].sub(reference, axis=0).median(axis=0, skipna=True)


def paired_deltas(matrix: pd.DataFrame, tumor_suffix: str, normal_suffix: str) -> np.ndarray:
    patients = sorted(
        {column.split("-")[0] for column in matrix.index if column.endswith(f"-{tumor_suffix}")}
    )
    values = []
    for patient in patients:
        tumor, normal = f"{patient}-{tumor_suffix}", f"{patient}-{normal_suffix}"
        if tumor in matrix.index and normal in matrix.index:
            a, b = float(matrix.loc[tumor, "value"]), float(matrix.loc[normal, "value"])
            if np.isfinite(a) and np.isfinite(b):
                values.append(a - b)
    return np.asarray(values, dtype=float)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {f"{prefix}_n": int(len(values))}
    if not len(values):
        return result
    positives = int((values > 0).sum())
    nonzero = int((values != 0).sum())
    try:
        wp = float(wilcoxon(values, zero_method="wilcox").pvalue)
    except ValueError:
        wp = 1.0
    loo = np.asarray([np.mean(np.delete(values, i)) for i in range(len(values))]) if len(values) > 1 else values
    result.update(
        {
            f"{prefix}_mean_log2fc": float(np.mean(values)),
            f"{prefix}_median_log2fc": float(np.median(values)),
            f"{prefix}_positive_fraction": float(positives / len(values)),
            f"{prefix}_sign_p": float(binomtest(positives, nonzero, 0.5).pvalue) if nonzero else 1.0,
            f"{prefix}_ttest_p": float(ttest_1samp(values, 0).pvalue) if len(values) >= 2 and np.std(values) > 0 else 1.0,
            f"{prefix}_wilcoxon_p": wp,
            f"{prefix}_loo_sign_stability": float((np.sign(loo) == np.sign(np.mean(values))).mean()),
        }
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    novelty = pd.read_csv(NOVELTY)
    records: list[dict[str, object]] = []
    reports: dict[str, object] = {}
    for panel in ("neg_rp", "pos_rp"):
        auc = pd.read_csv(EIC / f"{panel}__eic_auc_matrix.csv.gz").set_index("feature_id")
        detected = pd.read_csv(EIC / f"{panel}__eic_detection_matrix.csv.gz").set_index("feature_id").astype(bool)
        auc = auc.where(detected & (auc > 0))
        factors = discovery_pqn_factors(panel).reindex(auc.columns)
        if factors.isna().any():
            raise RuntimeError(f"{panel}: missing full-feature PQN factors")

        per_sample = []
        for path in sorted((EIC / "per_sample").glob(f"{panel}__*__eic.csv.gz")):
            per_sample.append(pd.read_csv(path))
        diagnostics = pd.concat(per_sample, ignore_index=True)

        positive_values = auc.stack().to_numpy(float)
        pseudo = float(np.percentile(positive_values, 1) / 2) if len(positive_values) else 1.0
        log_raw = np.log2(auc + pseudo)
        log_pqn = log_raw.sub(factors, axis=1)
        for feature_id in auc.index.astype(int):
            record: dict[str, object] = {"panel": panel, "feature_id": feature_id, "pseudocount": pseudo}
            for name, matrix in (("eic_raw", log_raw), ("eic_full_pqn", log_pqn)):
                sample_frame = matrix.loc[feature_id].rename("value").to_frame()
                rmu = paired_deltas(sample_frame, "Rmu", "RN")
                rtu = paired_deltas(sample_frame, "Rtu", "RN")
                record.update(summarize(rmu, f"{name}_rmu"))
                record.update(summarize(rtu, f"{name}_rtu"))
                if len(rmu) >= 2 and len(rtu) >= 2:
                    record[f"{name}_interaction_log2fc"] = float(np.mean(rmu) - np.mean(rtu))
                    record[f"{name}_interaction_p"] = float(ttest_ind(rmu, rtu, equal_var=False).pvalue)

            diag = diagnostics.loc[diagnostics["feature_id"].eq(feature_id)]
            record.update(
                {
                    "eic_detection_fraction": float(diag["detected_eic"].mean()),
                    "eic_apex_delta_median_abs_sec": float(diag["eic_apex_delta_sec"].abs().median()),
                    "eic_apex_delta_p95_abs_sec": float(diag["eic_apex_delta_sec"].abs().quantile(0.95)),
                    "eic_median_snr": float(diag["eic_snr"].median()),
                    "eic_multiple_local_peak_fraction": float(diag["local_peak_count"].gt(1).mean()),
                }
            )
            records.append(record)
        reports[panel] = {
            "targets": int(len(auc)),
            "samples": int(len(auc.columns)),
            "pseudocount": pseudo,
        }

    result = pd.DataFrame(records).merge(novelty, on=["panel", "feature_id"], validate="one_to_one")
    for variant in ("eic_raw", "eic_full_pqn"):
        result[f"{variant}_rmu_ttest_q_six"] = bh(result[f"{variant}_rmu_ttest_p"])
    raw_effect = result["eic_raw_rmu_mean_log2fc"]
    pqn_effect = result["eic_full_pqn_rmu_mean_log2fc"]
    result["eic_direction_consistent"] = np.sign(raw_effect).eq(np.sign(pqn_effect))
    result["eic_min_abs_rmu_log2fc"] = pd.concat([raw_effect.abs(), pqn_effect.abs()], axis=1).min(axis=1)
    result["eic_max_rmu_sign_p"] = result[["eic_raw_rmu_sign_p", "eic_full_pqn_rmu_sign_p"]].max(axis=1)
    result["eic_max_rmu_ttest_p"] = result[["eic_raw_rmu_ttest_p", "eic_full_pqn_rmu_ttest_p"]].max(axis=1)
    result["technical_reextraction_pass"] = (
        result["eic_direction_consistent"]
        & result["eic_min_abs_rmu_log2fc"].ge(0.50)
        & result["eic_max_rmu_sign_p"].le(0.05)
        & result["eic_max_rmu_ttest_p"].le(0.05)
        & result["eic_detection_fraction"].ge(0.80)
        & result["eic_apex_delta_p95_abs_sec"].le(12.0)
    )
    result.to_csv(OUT / "candidate_targeted_eic_results.csv", index=False)

    report = {
        "status": "mtbls13729_broad_candidate_eic_analysis_complete",
        "formal": False,
        "panels": reports,
        "candidates": int(len(result)),
        "technical_reextraction_pass": int(result["technical_reextraction_pass"].sum()),
        "fdr10_screen_and_reextraction_pass": int((result["screen_fdr10"] & result["technical_reextraction_pass"]).sum()),
        "claim_limit": (
            "Targeted EIC re-extraction uses the same cohort as discovery and is not independent biological replication. "
            "It tests technical robustness to peak-table construction. Identities remain Level 2 and flux is not measured."
        ),
        "provenance": {
            "novelty_sha256": sha256(NOVELTY),
            "eic_report_sha256": sha256(EIC / "report.json"),
            "target_report_sha256": sha256(TARGETS / "report.json"),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = [
        "panel", "feature_id", "dreams_name", "screen_fdr10", "author_identity_rmu_significant",
        "eic_raw_rmu_mean_log2fc", "eic_full_pqn_rmu_mean_log2fc", "eic_max_rmu_sign_p",
        "eic_max_rmu_ttest_p", "eic_detection_fraction", "eic_apex_delta_p95_abs_sec",
        "technical_reextraction_pass",
    ]
    print(result[columns].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
