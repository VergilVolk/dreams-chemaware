from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/full_annotated_feature_audit_v1"
PANELS = ("neg_rp", "pos_rp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(float)
    out = np.full(arr.shape, np.nan)
    valid = np.isfinite(arr)
    p = arr[valid]
    if not len(p):
        return pd.Series(out, index=values.index)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    indices = np.flatnonzero(valid)[order]
    out[indices] = np.clip(adjusted, 0, 1)
    return pd.Series(out, index=values.index)


def pqn_factors(log_positive: pd.DataFrame, minimum_prevalence: float = 0.60) -> pd.Series:
    keep = log_positive.notna().mean(axis=1).ge(minimum_prevalence)
    reference = log_positive.loc[keep].median(axis=1, skipna=True)
    quotients = log_positive.loc[keep].sub(reference, axis=0)
    return quotients.median(axis=0, skipna=True)


def paired_values(matrix: pd.DataFrame, tumor_suffix: str, normal_suffix: str) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    patients = sorted({column.split("-")[0] for column in matrix.columns if column.endswith(f"-{tumor_suffix}")})
    pairs = [(f"{patient}-{tumor_suffix}", f"{patient}-{normal_suffix}") for patient in patients]
    for feature_id, row in matrix.iterrows():
        values = []
        for tumor, normal in pairs:
            if tumor not in row.index or normal not in row.index:
                continue
            a, b = float(row[tumor]), float(row[normal])
            if np.isfinite(a) and np.isfinite(b):
                values.append(a - b)
        result[int(feature_id)] = np.asarray(values, dtype=float)
    return result


def summarize(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    n = int(len(values))
    result: dict[str, float | int] = {f"{prefix}_n": n}
    if n == 0:
        return result
    positives = int(np.sum(values > 0))
    nonzero = int(np.sum(values != 0))
    result.update(
        {
            f"{prefix}_mean_log2fc": float(np.mean(values)),
            f"{prefix}_median_log2fc": float(np.median(values)),
            f"{prefix}_positive_fraction": float(positives / n),
            f"{prefix}_sign_p": float(binomtest(positives, nonzero, 0.5).pvalue) if nonzero else 1.0,
            f"{prefix}_ttest_p": float(ttest_1samp(values, 0).pvalue) if n >= 2 and np.std(values) > 0 else 1.0,
        }
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports: dict[str, object] = {}
    all_priority = []
    for panel in PANELS:
        matrix_path = ROOT / f"data/mtbls13729/ms1_consensus/{panel}__discovery_intensity_matrix.csv.gz"
        annotation_path = ROOT / f"data/mtbls13729/ms1_ms2_link/{panel}__feature_best_annotations.csv.gz"
        matrix = pd.read_csv(matrix_path).set_index("feature_id")
        annotation = pd.read_csv(annotation_path)

        positive = matrix.where(matrix > 0)
        log_raw = np.log2(positive)
        factors = pqn_factors(log_raw, 0.60)
        log_pqn = log_raw.sub(factors, axis=1)

        raw_rmu = paired_values(log_raw, "Rmu", "RN")
        pqn_rmu = paired_values(log_pqn, "Rmu", "RN")
        raw_rtu = paired_values(log_raw, "Rtu", "RN")
        pqn_rtu = paired_values(log_pqn, "Rtu", "RN")
        rows = []
        for feature_id in annotation["feature_id"].astype(int):
            row = {"feature_id": feature_id, "panel": panel}
            row.update(summarize(raw_rmu.get(feature_id, np.asarray([])), "raw_rmu"))
            row.update(summarize(pqn_rmu.get(feature_id, np.asarray([])), "pqn_rmu"))
            row.update(summarize(raw_rtu.get(feature_id, np.asarray([])), "raw_rtu"))
            row.update(summarize(pqn_rtu.get(feature_id, np.asarray([])), "pqn_rtu"))
            rows.append(row)
        result = annotation.merge(pd.DataFrame(rows), on="feature_id", how="left", validate="one_to_one")
        for variant in ("raw", "pqn"):
            result[f"{variant}_rmu_sign_q"] = bh(result[f"{variant}_rmu_sign_p"])
            result[f"{variant}_rmu_ttest_q"] = bh(result[f"{variant}_rmu_ttest_p"])

        raw_effect = result["raw_rmu_mean_log2fc"]
        pqn_effect = result["pqn_rmu_mean_log2fc"]
        result["rmu_direction_consistent"] = np.sign(raw_effect).eq(np.sign(pqn_effect))
        result["rmu_min_abs_log2fc"] = pd.concat([raw_effect.abs(), pqn_effect.abs()], axis=1).min(axis=1)
        result["rmu_min_pairs"] = result[["raw_rmu_n", "pqn_rmu_n"]].min(axis=1)
        result["multi_sample_ms2"] = result["n_support_samples"].fillna(0).ge(3)
        result["screen_priority"] = (
            result["rmu_min_pairs"].ge(8)
            & result["rmu_direction_consistent"]
            & result["rmu_min_abs_log2fc"].ge(0.75)
            & result[["raw_rmu_sign_p", "pqn_rmu_sign_p"]].max(axis=1).le(0.05)
            & result["multi_sample_ms2"]
            & result["structure_agreement_fraction"].fillna(0).ge(0.60)
        )
        result["screen_fdr10"] = (
            result["screen_priority"]
            & result[["raw_rmu_ttest_q", "pqn_rmu_ttest_q"]].max(axis=1).lt(0.10)
        )
        result = result.sort_values(
            ["screen_fdr10", "screen_priority", "rmu_min_abs_log2fc", "n_support_samples"],
            ascending=[False, False, False, False],
        )
        result.to_csv(OUT / f"{panel}__annotated_feature_audit.csv.gz", index=False)
        priority = result.loc[result["screen_priority"]].copy()
        priority.to_csv(OUT / f"{panel}__priority.csv", index=False)
        all_priority.append(priority)
        reports[panel] = {
            "annotated_features": int(len(result)),
            "features_with_8_rmu_pairs": int(result["rmu_min_pairs"].ge(8).sum()),
            "screen_priority": int(result["screen_priority"].sum()),
            "screen_fdr10": int(result["screen_fdr10"].sum()),
            "direction_consistent": int(result["rmu_direction_consistent"].sum()),
            "provenance": {
                "matrix_sha256": sha256(matrix_path),
                "annotation_sha256": sha256(annotation_path),
            },
        }

    combined = pd.concat(all_priority, ignore_index=True)
    combined.to_csv(OUT / "all_priority.csv", index=False)
    report = {
        "status": "mtbls13729_full_annotated_feature_audit_complete",
        "formal": False,
        "panels": reports,
        "total_priority": int(len(combined)),
        "contract": (
            "Phenotype-blind annotations intersected with discovery-matrix paired abundance. "
            "This is a broad candidate screen, not targeted-EIC confirmation or structure truth."
        ),
        "selection_rule": (
            "at least 8 complete Rmu pairs; raw and 60%-prevalence PQN same direction; "
            "minimum absolute log2FC >=0.75; both exact sign p<=0.05; >=3 MS2 samples; agreement>=0.60"
        ),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
