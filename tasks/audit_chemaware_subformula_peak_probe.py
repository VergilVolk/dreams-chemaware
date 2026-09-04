"""No-training audit of formula-constrained peak supervision for DreaMS tokens.

Each [M+H]+ peak is assigned its nearest RDBE-valid elemental subformula of the
known precursor formula.  A frozen ridge probe predicts fragment and neutral-
loss compositions from official DreaMS peak tokens.  Equal-capacity controls
use only mass/intensity Fourier features or rotate labels among peaks in the
same spectrum.  Formula folds are disjoint and one outer fold is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score

from audit_chemaware_candidate_differential_rules import decode, load_magma_module
from noise_final_core import stable_fold


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ridge(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(np.clip(sample_weight, 1e-4, None))[:, None]
    xw, yw = x * root, y * root
    gram = xw.T @ xw
    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, xw.T @ yw).astype(np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(0, keepdims=True)
    scale = np.clip(train.std(0, keepdims=True), 1e-4, None)
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def raw_fourier(mz: np.ndarray, precursor: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    loss = precursor - mz
    periods = np.geomspace(0.005, 1000.0, 21, dtype=np.float64)
    columns = []
    for values in (mz, loss, precursor):
        phase = 2.0 * np.pi * values[:, None] / periods[None, :]
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack((*columns, np.sqrt(np.clip(intensity, 0, None)), np.ones(len(mz))))


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", left, right) / np.clip(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12, None,
    )


def metrics(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict:
    variable = (target.max(0) - target.min(0)) > 0
    present = target > 0
    ap = [
        average_precision_score(present[:, index], prediction[:, index], sample_weight=weight)
        for index in np.flatnonzero(variable)
        if 0 < np.sum(present[:, index]) < len(target)
    ]
    error = np.mean(np.abs(target[:, variable] - prediction[:, variable]), axis=1)
    similarity = cosine_rows(target[:, variable], prediction[:, variable])
    return {
        "peaks": int(len(target)),
        "variable_composition_dimensions": int(np.sum(variable)),
        "weighted_mean_absolute_error": float(np.average(error, weights=weight)),
        "weighted_mean_cosine": float(np.average(similarity, weights=weight)),
        "weighted_macro_element_presence_auprc": float(np.mean(ap)) if ap else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adduct", default="[M+H]+")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=9061)
    parser.add_argument("--permutation-seed", type=int, default=9062)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--ppm-floor-mz", type=float, default=200.0)
    parser.add_argument(
        "--ms-pred-source-root", type=Path,
        default=ROOT / "data/external/ms-pred-src",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.projection_dim != 128:
        raise ValueError("raw Fourier control is fixed at the matched 128-dimensional capacity")

    # Reuse the dependency-light loader so importing chem_utils does not pull
    # optional torch-scatter modules into this deterministic formula audit.
    load_magma_module(str(args.ms_pred_source_root))
    from ms_pred.common import chem_utils  # type: ignore

    @lru_cache(maxsize=None)
    def formula_table(formula: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vectors, masses = chem_utils.get_all_subsets(formula)
        parent = chem_utils.formula_to_dense(formula)
        return vectors.astype(np.float32), masses.astype(np.float64), parent.astype(np.float32)

    all_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    tokens_all = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
    mz_all = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity_all = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    valid_all = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    precursor_all = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    with h5py.File(args.hdf5, "r") as handle:
        positions = np.asarray([
            index for index, row in enumerate(all_rows)
            if decode(handle["adduct"][row]) == args.adduct
        ], dtype=np.int64)
        rows = all_rows[positions]
        formulas = np.asarray([decode(handle["FORMULA"][row]) for row in rows])

    tokens = np.asarray(tokens_all[positions])
    mz = np.asarray(mz_all[positions], dtype=np.float64)
    intensity = np.asarray(intensity_all[positions], dtype=np.float64)
    valid = np.asarray(valid_all[positions], dtype=bool)
    precursor = np.asarray(precursor_all[positions], dtype=np.float64)
    row_fold = np.asarray([
        stable_fold(formula, args.folds, args.fold_seed) for formula in formulas
    ], dtype=np.int8)

    n_elements = len(chem_utils.VALID_ELEMENTS)
    target = np.zeros((len(rows), mz.shape[1], 2 * n_elements), dtype=np.float32)
    target_mask = np.zeros(mz.shape, dtype=bool)
    mass_error_ppm = np.full(mz.shape, np.nan, dtype=np.float32)
    adduct_mass = float(chem_utils.ion2mass[args.adduct])
    electron_mass = -float(chem_utils.ELECTRON_MASS)
    for position, formula in enumerate(formulas):
        if row_fold[position] == args.outer_fold:
            continue
        subforms, masses, parent = formula_table(str(formula))
        observed = mz[position]
        error = np.minimum(
            np.abs(observed[:, None] - (masses[None, :] + adduct_mass)),
            np.abs(observed[:, None] - (masses[None, :] + electron_mass)),
        )
        nearest = np.argmin(error, axis=1)
        absolute = error[np.arange(len(observed)), nearest]
        ppm_error = absolute / np.maximum(np.abs(observed), args.ppm_floor_mz) * 1e6
        keep = valid[position] & (ppm_error < args.ppm)
        fragment = subforms[nearest]
        loss = np.clip(parent[None, :] - fragment, 0, None)
        denominator = np.clip(parent, 1, None)
        target[position, :, :n_elements] = fragment / denominator
        target[position, :, n_elements:] = loss / denominator
        target_mask[position] = keep
        mass_error_ppm[position] = ppm_error

    train_rows = (row_fold != args.outer_fold) & (row_fold != args.inner_fold)
    inner_rows = row_fold == args.inner_fold
    tr_pos, tr_peak = np.where(target_mask & train_rows[:, None])
    in_pos, in_peak = np.where(target_mask & inner_rows[:, None])
    if len(tr_pos) < 100 or len(in_pos) < 50:
        raise RuntimeError("insufficient formula-assigned peaks")

    rng = np.random.default_rng(args.projection_seed)
    projection = rng.normal(
        0.0, 1.0 / np.sqrt(tokens.shape[2]),
        size=(tokens.shape[2], args.projection_dim),
    ).astype(np.float32)
    dream_train = tokens[tr_pos, tr_peak].astype(np.float32) @ projection
    dream_inner = tokens[in_pos, in_peak].astype(np.float32) @ projection
    dream_train, dream_inner, dream_mean, dream_scale = standardize(dream_train, dream_inner)

    raw_train = raw_fourier(
        mz[tr_pos, tr_peak], precursor[tr_pos], intensity[tr_pos, tr_peak],
    ).astype(np.float32)
    raw_inner = raw_fourier(
        mz[in_pos, in_peak], precursor[in_pos], intensity[in_pos, in_peak],
    ).astype(np.float32)
    raw_train, raw_inner, raw_mean, raw_scale = standardize(raw_train, raw_inner)
    y_train, y_inner = target[tr_pos, tr_peak], target[in_pos, in_peak]
    train_weight = np.sqrt(np.clip(intensity[tr_pos, tr_peak], 1e-4, None)).astype(np.float32)
    inner_weight = np.sqrt(np.clip(intensity[in_pos, in_peak], 1e-4, None)).astype(np.float32)

    permuted = target.copy()
    permutation_rng = np.random.default_rng(args.permutation_seed)
    for position in np.flatnonzero(train_rows):
        peaks = np.flatnonzero(target_mask[position])
        if len(peaks) > 1:
            permuted[position, peaks] = permuted[position, permutation_rng.permutation(peaks)]
    y_permuted = permuted[tr_pos, tr_peak]

    weights = {
        "dreams_correct": ridge(dream_train, y_train, train_weight, args.ridge_alpha),
        "raw_mass_correct": ridge(raw_train, y_train, train_weight, args.ridge_alpha),
        "dreams_peak_permuted": ridge(dream_train, y_permuted, train_weight, args.ridge_alpha),
    }
    result = {
        "dreams_correct": metrics(y_inner, dream_inner @ weights["dreams_correct"], inner_weight),
        "raw_mass_correct": metrics(y_inner, raw_inner @ weights["raw_mass_correct"], inner_weight),
        "dreams_peak_permuted": metrics(
            y_inner, dream_inner @ weights["dreams_peak_permuted"], inner_weight,
        ),
    }
    dream_mae = result["dreams_correct"]["weighted_mean_absolute_error"]
    raw_mae = result["raw_mass_correct"]["weighted_mean_absolute_error"]
    permuted_mae = result["dreams_peak_permuted"]["weighted_mean_absolute_error"]
    report = {
        "status": "no_training_formula_constrained_peak_probe",
        "formal_training_authorized": False,
        "dreaMS_parameters_updated": False,
        "selection": {
            "adduct": args.adduct,
            "spectra": len(rows),
            "train_spectra": int(np.sum(train_rows)),
            "inner_spectra": int(np.sum(inner_rows)),
            "outer_spectra_not_labeled_or_evaluated": int(np.sum(row_fold == args.outer_fold)),
            "formula_fold_overlap": 0,
        },
        "assignment": {
            "kind": "nearest_RDBE_valid_precursor_subformula",
            "ppm": args.ppm,
            "ppm_denominator_floor_mz": args.ppm_floor_mz,
            "train_assigned_peaks": int(len(tr_pos)),
            "inner_assigned_peaks": int(len(in_pos)),
            "inner_assignment_coverage": float(
                np.sum(target_mask[inner_rows]) / np.clip(np.sum(valid[inner_rows]), 1, None)
            ),
            "inner_median_mass_error_ppm": float(np.nanmedian(
                mass_error_ppm[target_mask & inner_rows[:, None]]
            )),
            "elements": list(chem_utils.VALID_ELEMENTS),
            "target": "normalized fragment formula concatenated with neutral-loss formula",
        },
        "probe": {
            "capacity": 128,
            "ridge_alpha": args.ridge_alpha,
            **result,
            "dreams_minus_raw_mass_mae_improvement": float(raw_mae - dream_mae),
            "dreams_minus_peak_permuted_mae_improvement": float(permuted_mae - dream_mae),
        },
        "pass_to_formula_auxiliary_peft": bool(
            raw_mae - dream_mae > 0.005 and permuted_mae - dream_mae > 0.005
        ),
        "gate": (
            "Official DreaMS peak tokens must reduce normalized composition MAE by more than "
            "0.005 versus both a matched 128-d mass/intensity control and peak-permuted labels."
        ),
        "claim_limit": (
            "A pass supports a training-only formula auxiliary objective. It does not establish "
            "isomer discrimination, fragment structures, or retrieval improvement."
        ),
        "provenance": {
            "ms_pred_chem_utils_sha256": sha256_file(
                args.ms_pred_source_root / "src/ms_pred/common/chem_utils.py"
            ),
        },
    }
    args.output.mkdir(parents=True)
    np.save(args.output / "rows.npy", rows)
    np.save(args.output / "formulas.npy", formulas)
    np.save(args.output / "row_fold.npy", row_fold)
    np.save(args.output / "target_mask.npy", target_mask)
    # Persist the frozen training-only labels so downstream matched pilots use
    # exactly this audited assignment rather than silently recomputing it.
    np.save(args.output / "target_f16.npy", target.astype(np.float16))
    np.save(args.output / "mass_error_ppm_f32.npy", mass_error_ppm)
    np.savez_compressed(
        args.output / "frozen_probe.npz",
        projection=projection,
        dream_mean=dream_mean,
        dream_scale=dream_scale,
        raw_mean=raw_mean,
        raw_scale=raw_scale,
        **weights,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
