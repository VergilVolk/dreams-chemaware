"""Formula-OOF learnability audit for MAGMa peak-to-fragment soft labels.

The official DreaMS backbone is frozen.  A fixed random projection plus ridge
probe asks whether its peak tokens encode the candidate-specific subgraph
content of MAGMa fragments that explain each observed peak.  Two controls keep
probe capacity fixed: labels are rotated among peaks of the same spectrum, or
are regenerated from another structure with the same molecular formula.

This is a necessary-condition audit only.  It does not update DreaMS and does
not authorize retrieval training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score

from audit_chemaware_candidate_differential_rules import (
    PROTON,
    decode,
    load_magma_module,
    stable_u64,
)
from noise_final_core import stable_fold


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def magma_fragment_table(
    smiles: str,
    bits: int,
    source_root: str,
    tree_depth: int,
    max_broken_bonds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H-shifted masses, atom-subset ids, and transferable fingerprints."""
    fragmentation = load_magma_module(source_root)
    engine = fragmentation.FragmentEngine(
        smiles,
        max_tree_depth=tree_depth,
        max_broken_bonds=max_broken_bonds,
    )
    engine.generate_fragments()
    _, fragment_int, _, masses, scores = engine.get_frag_masses()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, bits), dtype=np.float32),
        )
    root = (1 << molecule.GetNumAtoms()) - 1
    keep = np.isfinite(masses) & (masses > 0) & (fragment_int != root)
    masses = np.asarray(masses[keep], dtype=np.float64)
    fragment_int = np.asarray(fragment_int[keep], dtype=np.int64)
    scores = np.asarray(scores[keep], dtype=np.float64)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    fingerprint_by_fragment: dict[int, np.ndarray] = {}
    for value in np.unique(fragment_int):
        atoms = [
            index for index in range(molecule.GetNumAtoms())
            if int(value) & (1 << index)
        ]
        fragment_smiles = Chem.MolFragmentToSmiles(
            molecule, atomsToUse=atoms, canonical=True, isomericSmiles=False,
        )
        fragment = Chem.MolFromSmiles(fragment_smiles)
        vector = np.zeros(bits, dtype=np.int8)
        if fragment is not None and fragment.GetNumAtoms() > 0:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(fragment), vector)
        fingerprint_by_fragment[int(value)] = vector.astype(np.float32, copy=False)
    fingerprints = np.vstack([fingerprint_by_fragment[int(value)] for value in fragment_int])
    # Store the heuristic bond-cut score as an extra column temporarily; lower
    # score means a less costly fragment under this MAGMa implementation.
    return masses, fragment_int, np.column_stack((fingerprints, scores.astype(np.float32)))


def peak_targets(
    mz: np.ndarray,
    precursor_mz: float,
    smiles: str,
    bits: int,
    ppm: float,
    floor_da: float,
    top_k: int,
    score_temperature: float,
    source_root: str,
    tree_depth: int,
    max_broken_bonds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masses, fragment_int, packed = magma_fragment_table(
        smiles, bits, source_root, tree_depth, max_broken_bonds,
    )
    output = np.zeros((len(mz), bits), dtype=np.float32)
    mask = np.zeros(len(mz), dtype=bool)
    confidence = np.zeros(len(mz), dtype=np.float32)
    ambiguity = np.zeros(len(mz), dtype=np.int16)
    if not len(masses):
        return output, mask, confidence, ambiguity
    fingerprints, scores = packed[:, :bits], packed[:, bits]
    loss = precursor_mz - mz
    direct_error = np.abs(mz[:, None] - (masses[None, :] + PROTON))
    loss_error = np.abs(loss[:, None] - masses[None, :])
    direct_tolerance = np.maximum(floor_da, np.abs(mz) * ppm * 1e-6)[:, None]
    loss_tolerance = np.maximum(floor_da, np.abs(loss) * ppm * 1e-6)[:, None]
    compatible = (direct_error <= direct_tolerance) | (loss_error <= loss_tolerance)
    scaled_error = np.minimum(
        direct_error / np.clip(direct_tolerance, 1e-12, None),
        loss_error / np.clip(loss_tolerance, 1e-12, None),
    )
    for peak in np.flatnonzero(np.any(compatible, axis=1)):
        matches = np.flatnonzero(compatible[peak])
        # Aggregate H-shift variants and direct/loss explanations by atom set
        # before measuring ambiguity; otherwise one fragment is counted many times.
        log_weight = -scores[matches] / score_temperature - 0.5 * scaled_error[peak, matches] ** 2
        aggregate: dict[int, float] = {}
        for index, logit in zip(matches, log_weight):
            value = int(fragment_int[index])
            aggregate[value] = aggregate.get(value, 0.0) + math.exp(float(np.clip(logit, -50, 50)))
        ranked = sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        weights = np.asarray([item[1] for item in ranked], dtype=np.float64)
        weights /= np.clip(weights.sum(), 1e-12, None)
        selected_fp = np.vstack([
            fingerprints[np.flatnonzero(fragment_int == value)[0]] for value, _ in ranked
        ])
        output[peak] = weights @ selected_fp
        ambiguity[peak] = len(ranked)
        if len(weights) == 1:
            confidence[peak] = 1.0
        else:
            entropy = -float(np.sum(weights * np.log(np.clip(weights, 1e-12, None))))
            confidence[peak] = max(0.0, 1.0 - entropy / math.log(len(weights)))
        mask[peak] = bool(np.any(output[peak] > 0))
    return output, mask, confidence, ambiguity


def fit_weighted_ridge(x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(np.clip(weight, 1e-4, None))[:, None]
    xw, yw = x * root, y * root
    gram = xw.T @ xw
    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, xw.T @ yw).astype(np.float32)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def metrics(y_soft: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict:
    y_true = y_soft > 0.05
    usable = (np.sum(y_true, axis=0) > 0) & (np.sum(y_true, axis=0) < len(y_true))
    ap = [
        average_precision_score(y_true[:, bit], prediction[:, bit], sample_weight=weight)
        for bit in np.flatnonzero(usable)
    ]
    similarity = cosine_rows(y_soft, prediction)
    return {
        "labeled_peaks": int(len(y_true)),
        "usable_bits": int(np.sum(usable)),
        "weighted_macro_auprc": float(np.mean(ap)) if ap else None,
        "macro_prevalence": float(np.mean(y_true[:, usable])) if np.any(usable) else None,
        "weighted_mean_target_prediction_cosine": float(np.average(similarity, weights=weight)),
        "median_target_prediction_cosine": float(np.median(similarity)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-spectra", type=int, default=0)
    parser.add_argument("--adduct", default="[M+H]+")
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
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-temperature", type=float, default=3.0)
    parser.add_argument(
        "--magma-source-root", type=Path,
        default=ROOT / "data/external/ms-pred-src",
    )
    parser.add_argument("--magma-tree-depth", type=int, default=3)
    parser.add_argument("--magma-max-broken-bonds", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bits < 16 or args.projection_dim < 16 or args.ridge_alpha <= 0:
        raise ValueError("invalid probe capacity or ridge strength")
    RDLogger.DisableLog("rdApp.*")

    all_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    tokens_all = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
    mz_all = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    valid_all = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    precursor_all = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    with h5py.File(args.hdf5, "r") as handle:
        eligible = np.asarray([
            decode(handle["adduct"][row]) == args.adduct for row in all_rows
        ], dtype=bool)
        positions = np.flatnonzero(eligible)
        positions = sorted(
            positions,
            key=lambda position: stable_u64(int(all_rows[position]), seed=args.projection_seed),
        )
        if args.max_spectra > 0:
            positions = positions[:args.max_spectra]
        positions = np.asarray(sorted(positions), dtype=np.int64)
        rows = all_rows[positions]
        formulas = np.asarray([decode(handle["FORMULA"][row]) for row in rows])
        smiles = np.asarray([decode(handle["smiles"][row]) for row in rows])

    tokens = np.asarray(tokens_all[positions])
    mz = np.asarray(mz_all[positions])
    valid = np.asarray(valid_all[positions])
    precursor = np.asarray(precursor_all[positions])
    row_fold = np.asarray([
        stable_fold(formula, args.folds, args.fold_seed) for formula in formulas
    ], dtype=np.int8)

    # Same-formula donors are selected inside the non-outer pool.  Singleton
    # groups remain fixed and are explicitly reported.
    donors = np.arange(len(rows), dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for position, formula in enumerate(formulas):
        if row_fold[position] != args.outer_fold:
            groups.setdefault(str(formula), []).append(position)
    for formula, members in groups.items():
        members.sort(key=lambda p: stable_u64(formula, int(rows[p]), seed=args.permutation_seed))
        for index, position in enumerate(members):
            donors[position] = members[(index + 1) % len(members)]

    targets = np.zeros((len(rows), mz.shape[1], args.bits), dtype=np.float32)
    structure_control = np.zeros_like(targets)
    target_mask = np.zeros(mz.shape, dtype=bool)
    target_confidence = np.zeros(mz.shape, dtype=np.float32)
    target_ambiguity = np.zeros(mz.shape, dtype=np.int16)
    for position in range(len(rows)):
        if row_fold[position] == args.outer_fold:
            continue
        target, mask, confidence, ambiguity = peak_targets(
            mz[position], float(precursor[position]), smiles[position], args.bits,
            args.ppm, args.floor_da, args.top_k, args.score_temperature,
            str(args.magma_source_root), args.magma_tree_depth, args.magma_max_broken_bonds,
        )
        target_mask[position] = mask & valid[position]
        targets[position] = target
        target_confidence[position] = confidence
        target_ambiguity[position] = ambiguity
        donor = int(donors[position])
        donor_target, _, _, _ = peak_targets(
            mz[position], float(precursor[position]), smiles[donor], args.bits,
            args.ppm, args.floor_da, args.top_k, args.score_temperature,
            str(args.magma_source_root), args.magma_tree_depth, args.magma_max_broken_bonds,
        )
        structure_control[position] = donor_target

    train_row = (row_fold != args.outer_fold) & (row_fold != args.inner_fold)
    inner_row = row_fold == args.inner_fold
    train_position, train_peak = np.where(target_mask & train_row[:, None])
    inner_position, inner_peak = np.where(target_mask & inner_row[:, None])
    if len(train_position) < 100 or len(inner_position) < 50:
        raise RuntimeError("insufficient MAGMa-matched peak targets for probe audit")

    rng = np.random.default_rng(args.projection_seed)
    projection = rng.normal(
        0.0, 1.0 / np.sqrt(tokens.shape[2]),
        size=(tokens.shape[2], args.projection_dim),
    ).astype(np.float32)
    x_train = tokens[train_position, train_peak].astype(np.float32) @ projection
    x_inner = tokens[inner_position, inner_peak].astype(np.float32) @ projection
    mean, scale = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True)
    scale = np.clip(scale, 1e-4, None)
    x_train, x_inner = (x_train - mean) / scale, (x_inner - mean) / scale
    y_train = targets[train_position, train_peak]
    y_inner = targets[inner_position, inner_peak]
    train_weight = np.clip(target_confidence[train_position, train_peak], 0.05, 1.0)
    inner_weight = np.clip(target_confidence[inner_position, inner_peak], 0.05, 1.0)

    peak_control = targets.copy()
    permutation_rng = np.random.default_rng(args.permutation_seed)
    for position in np.flatnonzero(train_row):
        peaks = np.flatnonzero(target_mask[position])
        if len(peaks) > 1:
            peak_control[position, peaks] = peak_control[
                position, permutation_rng.permutation(peaks)
            ]
    controls = {
        "correct": y_train,
        "within_spectrum_peak_permuted": peak_control[train_position, train_peak],
        "same_formula_structure_permuted": structure_control[train_position, train_peak],
    }
    result = {}
    weights = {}
    for name, labels in controls.items():
        weight = fit_weighted_ridge(x_train, labels, train_weight, args.ridge_alpha)
        prediction = x_inner @ weight
        weights[name] = weight
        result[name] = metrics(y_inner, prediction, inner_weight)

    correct_ap = result["correct"]["weighted_macro_auprc"]
    peak_ap = result["within_spectrum_peak_permuted"]["weighted_macro_auprc"]
    structure_ap = result["same_formula_structure_permuted"]["weighted_macro_auprc"]
    report = {
        "status": "no_training_magma_peak_fragment_probe",
        "formal_training_authorized": False,
        "dreaMS_parameters_updated": False,
        "selection": {
            "adduct": args.adduct,
            "spectra": len(rows),
            "train_spectra": int(np.sum(train_row)),
            "inner_spectra": int(np.sum(inner_row)),
            "outer_spectra_not_labeled_or_evaluated": int(np.sum(row_fold == args.outer_fold)),
            "formula_fold_overlap": 0,
        },
        "target": {
            "kind": "posterior_weighted_magma_fragment_morgan_bits",
            "bits": args.bits,
            "top_k_fragments_per_peak": args.top_k,
            "score_temperature": args.score_temperature,
            "ppm": args.ppm,
            "floor_da": args.floor_da,
            "train_labeled_peaks": int(len(train_position)),
            "inner_labeled_peaks": int(len(inner_position)),
            "median_top_k_ambiguity": float(np.median(
                target_ambiguity[target_mask]
            )),
            "median_entropy_confidence": float(np.median(
                target_confidence[target_mask]
            )),
            "same_formula_structure_permutation_fixed_fraction": float(np.mean([
                donors[position] == position
                for position in np.flatnonzero(row_fold != args.outer_fold)
            ])),
        },
        "magma": {
            "source_root": str(args.magma_source_root.resolve()),
            "fragmentation_sha256": sha256_file(
                args.magma_source_root / "src/ms_pred/magma/fragmentation.py"
            ),
            "tree_depth": args.magma_tree_depth,
            "max_broken_bonds": args.magma_max_broken_bonds,
        },
        "probe": {
            "kind": "fixed_random_projection_plus_confidence_weighted_ridge",
            "projection_dim": args.projection_dim,
            "ridge_alpha": args.ridge_alpha,
            **result,
            "correct_minus_peak_permuted_auprc": float(correct_ap - peak_ap),
            "correct_minus_structure_permuted_auprc": float(correct_ap - structure_ap),
        },
        "pass_to_peak_level_peft": bool(
            correct_ap > peak_ap + 0.01 and correct_ap > structure_ap + 0.01
        ),
        "gate": (
            "Correct MAGMa peak-fragment labels must exceed both equal-capacity controls by "
            "more than 0.01 weighted macro-AUPRC on a formula-disjoint inner fold."
        ),
        "claim_limit": (
            "A pass only establishes that frozen DreaMS peak tokens contain decodable local "
            "chemical information. It does not establish fragment correctness or retrieval gain."
        ),
    }
    args.output.mkdir(parents=True)
    np.save(args.output / "rows.npy", rows)
    np.save(args.output / "row_fold.npy", row_fold)
    np.save(args.output / "target_mask.npy", target_mask)
    np.save(args.output / "target_confidence_f32.npy", target_confidence)
    np.savez_compressed(
        args.output / "frozen_probe.npz",
        projection=projection,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        **{f"{name}_weight": value for name, value in weights.items()},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

