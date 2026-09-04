"""Audit localized peak-to-substructure supervision for ChemAware PEFT.

This is a development-only, outer-fold-sealed feasibility test inspired by
MIST's peak/formula and auxiliary substructure supervision.  It constructs
approximate single-cut fragment fingerprints, assigns them to exact-mass
compatible peak or neutral-loss tokens, and asks whether a frozen low-rank
probe decodes the correct assignment better than a within-spectrum
peak-permuted control.

The probe and labels are training-only artifacts.  Candidate structures are
never inputs to the deployable shared spectrum encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score

from audit_chemaware_structural_peak_evidence import PROTON, decode, single_cut_fragments
from noise_final_core import stable_fold


def fragment_targets(
    mz: np.ndarray,
    precursor_mz: float,
    smiles: str,
    bits: int,
    ppm: float,
    floor_da: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a multi-hot fragment fingerprint and evidence mask per peak."""

    fragments = single_cut_fragments(smiles)
    output = np.zeros((len(mz), bits), dtype=np.uint8)
    mask = np.zeros(len(mz), dtype=bool)
    if not fragments:
        return output, mask
    masses = np.asarray([item[0] for item in fragments], dtype=np.float64)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    fingerprints = np.zeros((len(fragments), bits), dtype=np.uint8)
    for index, (_, fragment_smiles) in enumerate(fragments):
        molecule = Chem.MolFromSmiles(fragment_smiles)
        if molecule is None:
            continue
        vector = np.zeros(bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
        fingerprints[index] = vector.astype(np.uint8, copy=False)
    direct = np.minimum(
        np.abs(mz[:, None] - masses[None, :]),
        np.abs(mz[:, None] - (masses + PROTON)[None, :]),
    )
    neutral_loss = precursor_mz - mz
    loss_distance = np.abs(neutral_loss[:, None] - masses[None, :])
    tolerance = np.maximum(floor_da, np.abs(mz) * ppm * 1e-6)[:, None]
    matched = (direct <= tolerance) | (loss_distance <= tolerance)
    for peak in np.flatnonzero(np.any(matched, axis=1)):
        target = np.any(fingerprints[matched[peak]] != 0, axis=0)
        if np.any(target):
            output[peak] = target
            mask[peak] = True
    return output, mask


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, x.T @ y).astype(np.float32)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    usable = (np.sum(y_true, axis=0) > 0) & (np.sum(y_true, axis=0) < len(y_true))
    ap = [
        average_precision_score(y_true[:, bit], prediction[:, bit])
        for bit in np.flatnonzero(usable)
    ]
    prevalence = np.mean(y_true[:, usable], axis=0) if np.any(usable) else np.empty(0)
    return {
        "labeled_peaks": int(len(y_true)),
        "usable_bits": int(np.sum(usable)),
        "macro_auprc": float(np.mean(ap)) if ap else None,
        "macro_prevalence": float(np.mean(prevalence)) if len(prevalence) else None,
        "median_target_prediction_cosine": float(np.median(cosine_rows(y_true, prediction))),
        "mean_target_prediction_cosine": float(np.mean(cosine_rows(y_true, prediction))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=9041)
    parser.add_argument("--permutation-seed", type=int, default=9042)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--floor-da", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.bits < 16 or args.projection_dim < 16 or args.ridge_alpha <= 0:
        raise ValueError("invalid probe capacity or ridge strength")
    RDLogger.DisableLog("rdApp.*")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    tokens = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
    mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    valid = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    precursor = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    if tokens.shape[:2] != mz.shape or mz.shape != valid.shape:
        raise RuntimeError("token cache arrays are not aligned")

    targets = np.zeros((len(rows), mz.shape[1], args.bits), dtype=np.uint8)
    target_mask = np.zeros(mz.shape, dtype=bool)
    row_fold = np.empty(len(rows), dtype=np.int8)
    with h5py.File(args.hdf5, "r") as handle:
        for position, row in enumerate(rows):
            formula = decode(handle["FORMULA"][row])
            row_fold[position] = stable_fold(formula, args.folds, args.fold_seed)
            if row_fold[position] == args.outer_fold:
                continue
            smiles = decode(handle["smiles"][row])
            peak_target, peak_mask = fragment_targets(
                np.asarray(mz[position], dtype=np.float64),
                float(precursor[position]), smiles, args.bits, args.ppm, args.floor_da,
            )
            peak_mask &= np.asarray(valid[position], dtype=bool)
            targets[position] = peak_target
            target_mask[position] = peak_mask

    train_row = (row_fold != args.outer_fold) & (row_fold != args.inner_fold)
    inner_row = row_fold == args.inner_fold
    train_position, train_peak = np.where(target_mask & train_row[:, None])
    inner_position, inner_peak = np.where(target_mask & inner_row[:, None])
    if len(train_position) < 100 or len(inner_position) < 50:
        raise RuntimeError("insufficient matched peak targets for probe audit")

    rng = np.random.default_rng(args.projection_seed)
    projection = rng.normal(
        0.0, 1.0 / np.sqrt(tokens.shape[2]),
        size=(tokens.shape[2], args.projection_dim),
    ).astype(np.float32)
    x_train = np.asarray(tokens[train_position, train_peak], dtype=np.float32) @ projection
    x_inner = np.asarray(tokens[inner_position, inner_peak], dtype=np.float32) @ projection
    mean = np.mean(x_train, axis=0, keepdims=True)
    scale = np.std(x_train, axis=0, keepdims=True)
    scale = np.clip(scale, 1e-4, None)
    x_train = (x_train - mean) / scale
    x_inner = (x_inner - mean) / scale
    y_train = targets[train_position, train_peak].astype(np.float32)
    y_inner = targets[inner_position, inner_peak].astype(np.float32)

    permuted = targets.copy()
    permutation_rng = np.random.default_rng(args.permutation_seed)
    for position in np.flatnonzero(train_row):
        peaks = np.flatnonzero(target_mask[position])
        if len(peaks) > 1:
            permuted[position, peaks] = permuted[position, permutation_rng.permutation(peaks)]
    y_train_control = permuted[train_position, train_peak].astype(np.float32)

    correct_weight = fit_ridge(x_train, y_train, args.ridge_alpha)
    control_weight = fit_ridge(x_train, y_train_control, args.ridge_alpha)
    correct_prediction = x_inner @ correct_weight
    control_prediction = x_inner @ control_weight
    correct = metrics(y_inner, correct_prediction)
    control = metrics(y_inner, control_prediction)
    report = {
        "status": "chemaware_peak_substructure_probe_diagnostic",
        "outer_fold_not_labeled_or_evaluated": int(args.outer_fold),
        "inner_fold": int(args.inner_fold),
        "spectra": {
            "train": int(np.sum(train_row)),
            "inner": int(np.sum(inner_row)),
            "outer_masked": int(np.sum(row_fold == args.outer_fold)),
        },
        "target": {
            "kind": "single_cut_fragment_morgan_multi_hot",
            "bits": args.bits,
            "ppm": args.ppm,
            "floor_da": args.floor_da,
            "train_labeled_peaks": int(len(train_position)),
            "inner_labeled_peaks": int(len(inner_position)),
        },
        "probe": {
            "kind": "fixed_random_projection_plus_frozen_ridge",
            "projection_dim": args.projection_dim,
            "ridge_alpha": args.ridge_alpha,
            "correct": correct,
            "within_spectrum_peak_permuted_training_control": control,
            "delta_macro_auprc": (
                float(correct["macro_auprc"] - control["macro_auprc"])
                if correct["macro_auprc"] is not None and control["macro_auprc"] is not None else None
            ),
        },
        "claim_limit": (
            "A positive matched-control gap supports localized training-only chemical supervision. "
            "It does not identify fragments, prove fragmentation mechanisms, or establish retrieval gains."
        ),
    }
    np.save(args.output / "rows.npy", rows)
    np.save(args.output / "row_fold.npy", row_fold)
    np.save(args.output / "targets_u8.npy", targets)
    np.save(args.output / "target_mask.npy", target_mask)
    np.savez_compressed(
        args.output / "frozen_probe.npz",
        projection=projection,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        correct_weight=correct_weight,
        control_weight=control_weight,
    )
    report["artifacts_sha256"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("rows.npy", "row_fold.npy", "targets_u8.npy", "target_mask.npy", "frozen_probe.npz")
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
