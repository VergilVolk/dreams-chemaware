#!/usr/bin/env python
"""Clinical-stratum sensitivity audit for frozen MTBLS13729 biology axes.

This is deliberately descriptive: the Rmu discovery subgroup contains ten
patients, all right-sided, with MMR and BRAF imbalance. Exact label
permutations are used where both groups have at least three observations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/clinical_axis_sensitivity_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_label_permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.r_[a, b]
    observed = abs(float(np.mean(a) - np.mean(b)))
    exceed = 0
    total = 0
    for choice in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(choice)] = True
        statistic = abs(float(np.mean(pooled[mask]) - np.mean(pooled[~mask])))
        exceed += int(statistic >= observed - 1e-12)
        total += 1
    return float(exceed / total)


def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.dropna().sort_values()
    if finite.empty:
        return result
    ranked = finite.to_numpy(float) * len(finite) / np.arange(1, len(finite) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result.loc[finite.index] = np.minimum(ranked, 1.0)
    return result


def compare(frame: pd.DataFrame, group: str, left: str, right: str, contrast: str) -> list[dict]:
    rows: list[dict] = []
    for axis, axis_frame in frame.groupby("axis", sort=False):
        a = axis_frame.loc[axis_frame[group] == left, "log2fc"].dropna().to_numpy(float)
        b = axis_frame.loc[axis_frame[group] == right, "log2fc"].dropna().to_numpy(float)
        p = exact_label_permutation_p(a, b) if len(a) >= 3 and len(b) >= 3 else np.nan
        rows.append(
            {
                "contrast": contrast,
                "axis": axis,
                "left_label": left,
                "right_label": right,
                "n_left": int(len(a)),
                "n_right": int(len(b)),
                "mean_left": float(np.mean(a)) if len(a) else np.nan,
                "mean_right": float(np.mean(b)) if len(b) else np.nan,
                "mean_difference_left_minus_right": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan,
                "median_left": float(np.median(a)) if len(a) else np.nan,
                "median_right": float(np.median(b)) if len(b) else np.nan,
                "exact_label_permutation_p": p,
            }
        )
    return rows


def main() -> None:
    module_path = ROOT / "data/mtbls13729/biology_closure_analysis_v1/module_patient_effects.csv"
    rmu_path = ROOT / "data/mtbls13729/biology_closure_analysis_v1/rmu_pair_deltas.csv"
    rtu_path = ROOT / "data/mtbls13729/biology_closure_analysis_v1/rtu_pair_deltas.csv"
    clinical_path = ROOT / "data/mtbls13729/clinical_metadata_s2.tsv"
    for path in (module_path, rmu_path, rtu_path, clinical_path):
        if not path.exists():
            raise FileNotFoundError(path)

    module = pd.read_csv(module_path)
    module = module[(module.normalization == "raw") & module.cohort.isin(["Rmu", "Rtu"])][
        ["cohort", "patient", "modified_guanosine_module_log2fc"]
    ].rename(columns={"modified_guanosine_module_log2fc": "log2fc"})
    module["axis"] = "modified_guanosine_module"

    feature_axes = {
        4966: "purine_like_4966",
        1717: "acetylated_polyamine_1717",
        3222: "long_chain_acylcarnitine_3222",
    }
    feature_frames = []
    for cohort, path in [("Rmu", rmu_path), ("Rtu", rtu_path)]:
        frame = pd.read_csv(path)
        frame = frame[frame.feature_id.isin(feature_axes)].copy()
        frame["cohort"] = cohort
        frame["axis"] = frame.feature_id.map(feature_axes)
        frame = frame.rename(columns={"delta_log2": "log2fc"})
        feature_frames.append(frame[["cohort", "patient", "axis", "log2fc"]])
    effects = pd.concat([module[["cohort", "patient", "axis", "log2fc"]], *feature_frames], ignore_index=True)

    clinical = pd.read_csv(clinical_path, sep="\t")
    clinical = clinical[clinical.tissue == "Tumor"].copy()
    clinical["patient"] = "P" + clinical.patient_number.astype(int).astype(str).str.zfill(2)
    effects = effects.merge(
        clinical[["patient", "pathology", "location", "braf", "mmr"]],
        on="patient",
        how="left",
        validate="many_to_one",
    )
    if effects[["pathology", "location", "braf", "mmr"]].isna().any().any():
        raise RuntimeError("clinical linkage incomplete")

    rmu = effects[effects.cohort == "Rmu"].copy()
    rows = compare(rmu, "mmr", "dMMR", "pMMR", "Rmu dMMR vs pMMR")
    rows += compare(rmu, "braf", "+", "-", "Rmu BRAF+ vs BRAF-")

    pmmr = effects[effects.mmr == "pMMR"].copy()
    rows += compare(pmmr, "cohort", "Rmu", "Rtu", "pMMR Rmu vs Rtu")
    comparisons = pd.DataFrame(rows)
    comparisons["bh_q_within_contrast"] = comparisons.groupby("contrast", sort=False)[
        "exact_label_permutation_p"
    ].transform(bh)

    OUT.mkdir(parents=True, exist_ok=True)
    effects.to_csv(OUT / "patient_axis_effects.csv", index=False)
    comparisons.to_csv(OUT / "clinical_axis_comparisons.csv", index=False)

    evaluable = comparisons.dropna(subset=["exact_label_permutation_p"])
    payload = {
        "status": "mtbls13729_clinical_axis_sensitivity_complete",
        "formal": False,
        "reason_formal_false": "post-discovery sensitivity analysis in ten Rmu patients; no independent validation and all Rmu are right-sided",
        "rmu_patients": int(rmu.patient.nunique()),
        "rmu_mmr_counts": rmu.drop_duplicates("patient").mmr.value_counts().to_dict(),
        "rmu_braf_counts": rmu.drop_duplicates("patient").braf.value_counts().to_dict(),
        "rmu_location_counts": rmu.drop_duplicates("patient").location.value_counts().to_dict(),
        "axes": effects.axis.drop_duplicates().tolist(),
        "evaluable_exact_comparisons": int(len(evaluable)),
        "minimum_nominal_p": float(evaluable.exact_label_permutation_p.min()) if len(evaluable) else None,
        "minimum_bh_q": float(evaluable.bh_q_within_contrast.min()) if len(evaluable) else None,
        "provenance": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (module_path, rmu_path, rtu_path, clinical_path)
        },
        "claim_limit": "This audit can identify gross MMR/BRAF dependence or pMMR histology trends. It cannot establish subtype specificity, adjust jointly for covariates, or rescue the lack of an independent Rmu metabolomics cohort.",
    }
    (OUT / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
