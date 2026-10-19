#!/usr/bin/env python
"""Build a study-level LCNEC contrast without effect-based feature selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pqn(matrix: np.ndarray) -> np.ndarray:
    reference = np.median(matrix, axis=0)
    quotients = np.divide(matrix, reference, out=np.full_like(matrix, np.nan), where=reference > 0)
    factors = np.nanmedian(np.where(quotients > 0, quotients, np.nan), axis=1)
    if np.any(~np.isfinite(factors) | (factors <= 0)):
        raise RuntimeError("PQN normalization has nonpositive or nonfinite sample factors")
    return matrix / factors[:, None]


def paired_effects(
    matrix: np.ndarray,
    study_rows: pd.DataFrame,
    row_by_sample: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    positive = np.where(matrix > 0, matrix, np.inf)
    pseudocount = np.min(positive, axis=0) / 2.0
    pseudocount = np.where(np.isfinite(pseudocount) & (pseudocount > 0), pseudocount, 1.0)
    pairs: dict[str, dict[str, int]] = {}
    for row in study_rows.itertuples(index=False):
        pairs.setdefault(str(row.SAMPLE_CODE), {})[str(row.GROUP_CODE)] = row_by_sample[str(row.SAMPLE_ID)]
    if len(pairs) != 34 or any(set(value) != {"TU", "NG"} for value in pairs.values()):
        raise RuntimeError("expected exactly 34 complete tumor/adjacent pairs")
    patient_ids = sorted(pairs)
    tumour = np.stack([matrix[pairs[patient]["TU"]] for patient in patient_ids])
    adjacent = np.stack([matrix[pairs[patient]["NG"]] for patient in patient_ids])
    effects = np.log2(tumour + pseudocount) - np.log2(adjacent + pseudocount)
    if not np.isfinite(effects).all():
        raise RuntimeError("nonfinite paired effects")
    return effects, patient_ids


def build(
    matrix_path: Path,
    overview_path: Path,
    target_ledger_path: Path,
    spectra_manifest_path: Path,
    min_study_detection: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cache = np.load(matrix_path, allow_pickle=False)
    required_arrays = {"area", "family_id", "sample_id", "sample_code", "group_code", "file_class", "mz", "rt_sec"}
    missing_arrays = required_arrays - set(cache.files)
    if missing_arrays:
        raise RuntimeError(f"matrix missing arrays: {sorted(missing_arrays)}")
    area = np.asarray(cache["area"], dtype=float)
    family_id = np.asarray(cache["family_id"], dtype=int)
    sample_id = np.asarray(cache["sample_id"]).astype(str)
    if area.shape != (85, 263) or len(np.unique(family_id)) != 263:
        raise RuntimeError(f"expected frozen 85 x 263 matrix, observed {area.shape}")
    if not np.isfinite(area).all() or np.any(area < 0):
        raise RuntimeError("abundance matrix must be finite and nonnegative")

    overview = pd.read_csv(overview_path, sep="\t")
    if len(overview) != 85 or set(overview["SAMPLE_ID"].astype(str)) != set(sample_id):
        raise RuntimeError("overview and matrix sample universes differ")
    row_by_sample = {value: index for index, value in enumerate(sample_id)}
    study = overview.loc[overview["NOTE"].eq("Study sample")].copy()
    qc = overview.loc[overview["NOTE"].eq("QC sample")].copy()
    if len(study) != 68 or len(qc) != 9:
        raise RuntimeError("expected 68 study and 9 pooled-QC injections")
    if study["AMOUNT"].isna().any() or (study["AMOUNT"].astype(float) <= 0).any():
        raise RuntimeError("study tissue amounts must be positive")

    study_indices = np.asarray([row_by_sample[str(value)] for value in study["SAMPLE_ID"]], dtype=int)
    qc_indices = np.asarray([row_by_sample[str(value)] for value in qc["SAMPLE_ID"]], dtype=int)
    study_area = area[study_indices]
    per_mg = study_area / study["AMOUNT"].astype(float).to_numpy()[:, None]
    qc_injection = qc["INJECTION_ID"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    study_injection = study["INJECTION_ID"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    drift = per_mg.copy()
    for column in range(area.shape[1]):
        values = area[qc_indices, column]
        median = float(np.median(values))
        if median <= 0:
            raise RuntimeError(f"QC median is nonpositive for family {family_id[column]}")
        factor = np.interp(study_injection, qc_injection, values / median)
        if np.any(~np.isfinite(factor) | (factor <= 0)):
            raise RuntimeError(f"invalid QC drift factor for family {family_id[column]}")
        drift[:, column] /= factor
    matrices = {
        "raw": study_area,
        "per_mg": per_mg,
        "per_mg_pqn": pqn(per_mg),
        "per_mg_drift_pqn": pqn(drift),
    }
    effects_by_world: dict[str, np.ndarray] = {}
    patients: list[str] | None = None
    study_reindexed = study.set_index(study["SAMPLE_ID"].astype(str)).loc[sample_id[study_indices]].reset_index(drop=True)
    local_row_by_sample = {str(value): index for index, value in enumerate(study_reindexed["SAMPLE_ID"])}
    for name, matrix in matrices.items():
        effects, current_patients = paired_effects(matrix, study_reindexed, local_row_by_sample)
        if patients is None:
            patients = current_patients
        elif current_patients != patients:
            raise RuntimeError("patient order changes across normalization worlds")
        effects_by_world[name] = effects
    assert patients is not None

    target_ledger = pd.read_csv(target_ledger_path).set_index("family_id")
    spectra = pd.read_csv(spectra_manifest_path).set_index("family_id")
    if set(map(int, family_id)) - set(map(int, target_ledger.index)):
        raise RuntimeError("target ledger does not cover matrix families")
    if set(map(int, family_id)) != set(map(int, spectra.index)):
        raise RuntimeError("representative MS2 manifest does not exactly cover matrix families")
    target_ledger = target_ledger.loc[family_id]
    spectra = spectra.loc[family_id]

    detection = np.mean(study_area > 0, axis=0)
    eligible = detection >= min_study_detection
    primary = effects_by_world["per_mg_drift_pqn"]
    beta = primary.mean(axis=0)
    se = primary.std(axis=0, ddof=1) / math.sqrt(primary.shape[0])
    if np.any(~np.isfinite(beta[eligible])) or np.any(~np.isfinite(se[eligible]) | (se[eligible] <= 0)):
        raise RuntimeError("eligible entities require finite beta and positive empirical SE")

    study_id = "LCNEC_MTB22_P073_HSST3n"
    contrast_id = "paired_tumor_vs_adjacent"
    entity_ids = np.asarray([f"hsst3n_family_{value}" for value in family_id])
    records = pd.DataFrame({
        "study_id": study_id,
        "source_family": "LCNEC_MTB22_P073",
        "contrast_id": contrast_id,
        "entity_id": entity_ids,
        "independent_unit_id": entity_ids,
        "identity_status": "unresolved_entity",
        "beta": beta,
        "se": se,
        "qc_pass": eligible,
        "detectable": eligible,
        "data_role": "development/consumed",
    })
    unit_rows = []
    for patient_index, patient in enumerate(patients):
        for entity_index, entity_id in enumerate(entity_ids):
            if eligible[entity_index]:
                unit_rows.append({
                    "study_id": study_id, "contrast_id": contrast_id,
                    "biological_unit_id": patient, "entity_id": entity_id,
                    "effect": float(primary[patient_index, entity_index]),
                })
    unit_effects = pd.DataFrame(unit_rows)

    sensitivity_rows = []
    world_beta = {name: values.mean(axis=0) for name, values in effects_by_world.items()}
    for index, entity_id in enumerate(entity_ids):
        row = {"entity_id": entity_id, "family_id": int(family_id[index])}
        values = []
        for name in ("raw", "per_mg", "per_mg_pqn", "per_mg_drift_pqn"):
            row[f"beta_{name}"] = float(world_beta[name][index])
            values.append(float(world_beta[name][index]))
        nonzero = [np.sign(value) for value in values if not np.isclose(value, 0.0)]
        row["direction_consistent"] = bool(len(set(nonzero)) <= 1)
        row["normalization_beta_range"] = float(max(values) - min(values))
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)

    qc_area = area[qc_indices]
    qc_mean = qc_area.mean(axis=0)
    qc_cv = np.divide(qc_area.std(axis=0, ddof=1), qc_mean, out=np.full_like(qc_mean, np.inf), where=qc_mean > 0)
    measurement = pd.DataFrame({
        "study_id": study_id,
        "contrast_id": contrast_id,
        "entity_id": entity_ids,
        "log_intensity": np.log1p(np.median(study_area, axis=0)),
        "detection_rate": detection,
        "qc_cv": qc_cv,
        "missingness": 1.0 - detection,
        "ms2_peak_count": spectra["n_peaks"].to_numpy(float),
        "measurement_bin": "HSST3n_negative",
    })
    report = {
        "status": "LCNEC_HSST3N_CONTRAST_ATLAS_RECORDS_COMPLETE",
        "formal": True,
        "study_id": study_id,
        "contrast_id": contrast_id,
        "data_role": "development/consumed",
        "target_selection_used_phenotype": False,
        "effect_significance_used_for_selection": False,
        "identity_assignment": "all unresolved pending effect-blind full-universe annotation",
        "registered_families": int(len(family_id)),
        "eligible_families": int(eligible.sum()),
        "ineligible_low_detection_families": int((~eligible).sum()),
        "minimum_study_detection": min_study_detection,
        "biological_denominator": {"paired_patients": len(patients), "unit": "patient"},
        "primary_normalization": "tissue-mass + pooled-QC linear drift + PQN; all steps effect-blind",
        "sensitivity_normalizations": list(matrices),
        "direction_consistent_all_four": int(sensitivity.loc[eligible, "direction_consistent"].sum()),
        "input_sha256": {
            "matrix": sha256(matrix_path),
            "overview": sha256(overview_path),
            "target_ledger": sha256(target_ledger_path),
            "spectra_manifest": sha256(spectra_manifest_path),
        },
        "claim_limit": "Study-level consumed contrast construction only; no cross-study retrieval, unknown increment, identity, mechanism, or external validation.",
    }
    return records, unit_effects, measurement, sensitivity, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("data/validation/lcnec_hsst3n_all_qc_eic_v1/all_qc_family_eic_matrix.npz"))
    parser.add_argument("--overview", type=Path, default=Path("data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt"))
    parser.add_argument("--target-ledger", type=Path, default=Path("data/validation/lcnec_global_remodeling_stage1/run_2352292/qc_headroom/precursor_family_ledger.csv"))
    parser.add_argument("--spectra-manifest", type=Path, default=Path("data/validation/lcnec_hsst3n_all_qc_annotation_v1/priority_dark_modules.csv"))
    parser.add_argument("--min-study-detection", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records, unit_effects, measurement, sensitivity, report = build(
        args.matrix, args.overview, args.target_ledger, args.spectra_manifest,
        args.min_study_detection,
    )
    args.output_dir.mkdir(parents=True)
    records.to_csv(args.output_dir / "contrast_records.csv.gz", index=False, compression="gzip")
    unit_effects.to_csv(args.output_dir / "contrast_unit_effects.csv.gz", index=False, compression="gzip")
    measurement.to_csv(args.output_dir / "measurement_covariates.csv.gz", index=False, compression="gzip")
    sensitivity.to_csv(args.output_dir / "normalization_sensitivity.csv.gz", index=False, compression="gzip")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
