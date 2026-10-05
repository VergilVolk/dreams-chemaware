"""Compile the confirmed positive rule matrix into candidate-centred targets.

Only same-formula candidate molecules receive a chemical residual.  Other
strict-mass candidates are exact zero, which prevents parent-formula signal
from masquerading as isomer discrimination.  Correct, structure-swapped, and
peak-permuted arms share the same query membership and score capacity.

The fold-0+1 matrix is valid for fold-2 action-utility confirmation only.  It
must be cross-fitted before it can supervise shared-embedding training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from mine_chemaware_domain_conditioned_action_rules import domain_label  # noqa: E402

FORMULA_PATTERN = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--predicates",
        type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_parent_predicates_rdkit_v1.json",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_positive_rule_matrix_pilot_v4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_rule_matrix_candidate_ledger_v1",
    )
    parser.add_argument("--intensity-threshold", type=float, default=0.01)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument(
        "--scope",
        choices=("same_formula_candidates", "formula_residual_all_candidates"),
        default="same_formula_candidates",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def take_rows(dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def exact_lookup(source_key: np.ndarray, target_key: np.ndarray) -> np.ndarray:
    source = np.asarray(source_key, dtype=np.int64)
    target = np.asarray(target_key, dtype=np.int64)
    if len(np.unique(source)) != len(source):
        raise RuntimeError("lookup source is not unique")
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    position = np.searchsorted(sorted_source, target)
    valid = position < len(sorted_source)
    matched = np.zeros(len(target), dtype=bool)
    matched[valid] = sorted_source[position[valid]] == target[valid]
    if not np.all(matched):
        raise RuntimeError(f"token lookup misses {int(np.sum(~matched))} rows")
    return order[position]


def stable_formula_folds(formulas: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little"
            )
            % 5
            for value in np.asarray(formulas).astype(str)
        ],
        dtype=np.int16,
    )


def observation_matrix(
    mz: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    precursor: np.ndarray,
    channels: list[dict],
    threshold: float,
) -> np.ndarray:
    output = np.zeros((len(mz), len(channels)), dtype=np.float32)
    target = np.asarray([item["value_da"] for item in channels], dtype=np.float64)
    tolerance = np.asarray(
        [item["match_tolerance_da"] for item in channels], dtype=np.float64
    )
    neutral = np.asarray([item["match_type"] == "mass_diff" for item in channels])
    for query in range(len(mz)):
        keep = (
            valid[query]
            & np.isfinite(mz[query])
            & np.isfinite(intensity[query])
            & (mz[query] > 0)
            & (intensity[query] >= threshold)
        )
        peaks = np.asarray(mz[query, keep], dtype=np.float64)
        if not len(peaks):
            continue
        for channel in np.flatnonzero(~neutral):
            output[query, channel] = float(
                np.min(np.abs(peaks - target[channel])) <= tolerance[channel]
            )
        losses = float(precursor[query]) - peaks
        losses = losses[losses > 0]
        if len(losses):
            for channel in np.flatnonzero(neutral):
                output[query, channel] = float(
                    np.min(np.abs(losses - target[channel])) <= tolerance[channel]
                )
    return output


def predicate_matrix(smiles: np.ndarray, predicates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    compiled = []
    for item in predicates:
        queries = [Chem.MolFromSmarts(value) for value in item["smarts_any"]]
        if any(value is None for value in queries):
            raise RuntimeError(f"invalid SMARTS in {item['predicate_id']}")
        compiled.append(queries)
    output = np.zeros((len(smiles), len(predicates)), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        valid[row] = molecule is not None
        if molecule is None:
            continue
        for column, queries in enumerate(compiled):
            output[row, column] = float(
                any(molecule.HasSubstructMatch(query) for query in queries)
            )
    return output, valid


def center_same_formula(
    raw: np.ndarray,
    query_ptr: np.ndarray,
    query_formula: np.ndarray,
    molecule_formula: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros_like(raw, dtype=np.float32)
    eligible = np.zeros(len(query_formula), dtype=bool)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        positions = np.arange(int(left), int(right), dtype=np.int64)
        positions = positions[molecule_formula[positions] == query_formula[query]]
        if len(positions) < 2:
            continue
        values = raw[positions].astype(np.float64)
        values -= np.mean(values)
        residual[positions] = values.astype(np.float32)
        eligible[query] = bool(np.any(values != 0.0))
    return residual, eligible


def center_all_candidates(
    raw: np.ndarray,
    query_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros_like(raw, dtype=np.float32)
    active = np.zeros(len(query_ptr) - 1, dtype=bool)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        values = raw[int(left) : int(right)].astype(np.float64)
        values -= np.mean(values)
        residual[int(left) : int(right)] = values.astype(np.float32)
        active[query] = bool(np.any(values != 0.0))
    return residual, active


def formula_residual_predicates(
    unique_presence: np.ndarray,
    unique_formula: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove formula means and build a fixed within-formula structure control."""
    residual = np.zeros_like(unique_presence, dtype=np.float32)
    swapped = np.zeros_like(unique_presence, dtype=np.float32)
    eligible = np.zeros(len(unique_presence), dtype=bool)
    for formula in np.unique(unique_formula):
        positions = np.flatnonzero(unique_formula == formula)
        if len(positions) < 2:
            continue
        values = unique_presence[positions].astype(np.float64)
        mean = np.mean(values, axis=0, keepdims=True)
        residual[positions] = (values - mean).astype(np.float32)
        offset = 1 + int(
            int.from_bytes(
                hashlib.sha256(f"{seed}|{formula}".encode()).digest()[:8], "little"
            )
            % (len(positions) - 1)
        )
        swapped[positions] = (np.roll(values, offset, axis=0) - mean).astype(np.float32)
        eligible[positions] = True
    return residual, swapped, eligible


def structure_swapped_predicates(
    presence: np.ndarray,
    query_ptr: np.ndarray,
    query_formula: np.ndarray,
    molecule_formula: np.ndarray,
    query_row: np.ndarray,
) -> np.ndarray:
    swapped = presence.copy()
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        positions = np.arange(int(left), int(right), dtype=np.int64)
        positions = positions[molecule_formula[positions] == query_formula[query]]
        if len(positions) < 2:
            continue
        offset = 1 + int(
            int.from_bytes(
                hashlib.sha256(str(int(query_row[query])).encode()).digest()[:8], "little"
            )
            % (len(positions) - 1)
        )
        swapped[positions] = np.roll(presence[positions], offset, axis=0)
    return swapped


def compile_raw_scores(
    evidence: np.ndarray,
    presence: np.ndarray,
    domains: np.ndarray,
    shared: np.ndarray,
    domain_names: np.ndarray,
    domain_matrices: np.ndarray,
    query_ptr: np.ndarray,
    query_formula: np.ndarray,
    molecule_formula: np.ndarray,
    same_formula_only: bool,
) -> np.ndarray:
    matrix_by_domain = {
        str(name): domain_matrices[index] for index, name in enumerate(domain_names)
    }
    raw = np.zeros(len(presence), dtype=np.float32)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        positions = np.arange(int(left), int(right), dtype=np.int64)
        if same_formula_only:
            positions = positions[molecule_formula[positions] == query_formula[query]]
        if len(positions) < 2 or not np.any(evidence[query]):
            continue
        matrix = matrix_by_domain.get(str(domains[query]), shared)
        latent = matrix @ evidence[query]
        raw[positions] = presence[positions] @ latent
    return raw


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.intensity_threshold != 0.01 or args.fold_seed != 20260935:
        raise ValueError("ledger protocol no longer matches the confirmed rule-matrix pilot")
    matrix_report = json.loads(
        (args.matrix_dir / "report.json").read_text(encoding="utf-8")
    )
    if (
        matrix_report.get("status") != "CHEMAWARE_POSITIVE_RULE_MATRIX_CONFIRMATION_PASS"
        or matrix_report.get("pass_to_direct_candidate_ledger") is not True
    ):
        raise RuntimeError("rule matrix did not pass both chemical-specificity controls")
    with np.load(args.matrix_dir / "rule_matrix.npz", allow_pickle=False) as loaded:
        shared = np.asarray(loaded["shared_matrix"], dtype=np.float32)
        domain_names = np.asarray(loaded["domain_names"]).astype(str)
        domain_matrices = np.asarray(loaded["domain_matrices"], dtype=np.float32)
        peak_shared = np.asarray(loaded["peak_permuted_shared_matrix"], dtype=np.float32)
        peak_domain_matrices = np.asarray(
            loaded["peak_permuted_domain_matrices"], dtype=np.float32
        )
        selected_channel = np.asarray(
            loaded["selected_observation_channel"], dtype=np.int64
        )

    predicates = json.loads(args.predicates.read_text(encoding="utf-8"))["predicates"]
    all_channels = json.loads(args.observations.read_text(encoding="utf-8"))["channels"]
    channels = [all_channels[int(index)] for index in selected_channel]
    if shared.shape != (len(predicates), len(channels)):
        raise RuntimeError("matrix dimensions do not match the rule registries")

    with np.load(args.manifest, allow_pickle=False) as body:
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(body["molecule_ptr"], dtype=np.int64)
        pair_candidate_row = np.asarray(body["pair_candidate_row"], dtype=np.int64)
        query_row = np.asarray(body["query_row"], dtype=np.int64)
        query_formula = np.asarray(body["query_formula"]).astype(str)
        query_adduct = np.asarray(body["query_adduct"]).astype(str)
        molecule_ik14 = np.asarray(body["molecule_ik14"]).astype(str)
        molecule_formula = np.asarray(body["molecule_formula"]).astype(str)

    token_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    token_position = exact_lookup(token_rows, query_row)
    mz = np.asarray(np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")[token_position])
    intensity = np.asarray(
        np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")[token_position]
    )
    valid_peak = np.asarray(
        np.load(args.token_dir / "valid.npy", mmap_mode="r")[token_position]
    )
    precursor = np.asarray(
        np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")[token_position],
        dtype=np.float64,
    )

    representative_row = pair_candidate_row[molecule_ptr[:-1]]
    unique_identity, identity_first, identity_inverse = np.unique(
        molecule_ik14, return_index=True, return_inverse=True
    )
    with h5py.File(args.data, "r") as handle:
        instrument = decode(take_rows(handle["INSTRUMENT_TYPE"], query_row))
        collision_energy = np.asarray(
            take_rows(handle["COLLISION_ENERGY"], query_row), dtype=np.float64
        )
        unique_rows = representative_row[identity_first]
        unique_smiles = decode(take_rows(handle["smiles"], unique_rows))
        observed_identity = decode(take_rows(handle["INCHIKEY"], unique_rows))
    observed_identity = np.asarray([value.split("-")[0] for value in observed_identity])
    if not np.array_equal(observed_identity, unique_identity):
        raise RuntimeError("candidate structure rows do not match graph identities")
    domains = np.asarray(
        [
            domain_label(adduct, instrument_value, energy)
            for adduct, instrument_value, energy in zip(
                query_adduct, instrument, collision_energy
            )
        ]
    )
    evidence = observation_matrix(
        mz, intensity, valid_peak, precursor, channels, args.intensity_threshold
    )
    unique_presence, unique_valid = predicate_matrix(unique_smiles, predicates)
    presence = unique_presence[identity_inverse]
    valid_structure = unique_valid[identity_inverse]
    unique_formula = molecule_formula[identity_first]
    if any(
        len(np.unique(molecule_formula[identity_inverse == index])) != 1
        for index in range(len(unique_identity))
    ):
        raise RuntimeError("one candidate identity maps to multiple molecular formulas")

    if args.scope == "formula_residual_all_candidates":
        unique_residual, unique_swapped, unique_formula_eligible = formula_residual_predicates(
            unique_presence, unique_formula, args.fold_seed
        )
        treatment_presence = unique_residual[identity_inverse]
        swapped_presence = unique_swapped[identity_inverse]
        formula_residual_eligible = unique_formula_eligible[identity_inverse]
        same_formula_only = False
    else:
        treatment_presence = presence
        swapped_presence = structure_swapped_predicates(
            presence, query_ptr, query_formula, molecule_formula, query_row
        )
        formula_residual_eligible = np.ones(len(presence), dtype=bool)
        same_formula_only = True

    complete_structure = np.ones(len(query_row), dtype=bool)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        positions = np.arange(int(left), int(right), dtype=np.int64)
        if same_formula_only:
            positions = positions[molecule_formula[positions] == query_formula[query]]
        complete_structure[query] = bool(
            len(positions) >= 2
            and np.all(valid_structure[positions])
            and np.any(formula_residual_eligible[positions])
        )
    evidence[~complete_structure] = 0.0

    correct_raw = compile_raw_scores(
        evidence,
        treatment_presence,
        domains,
        shared,
        domain_names,
        domain_matrices,
        query_ptr,
        query_formula,
        molecule_formula,
        same_formula_only,
    )
    structure_raw = compile_raw_scores(
        evidence,
        swapped_presence,
        domains,
        shared,
        domain_names,
        domain_matrices,
        query_ptr,
        query_formula,
        molecule_formula,
        same_formula_only,
    )
    peak_raw = compile_raw_scores(
        evidence,
        treatment_presence,
        domains,
        peak_shared,
        domain_names,
        peak_domain_matrices,
        query_ptr,
        query_formula,
        molecule_formula,
        same_formula_only,
    )
    if same_formula_only:
        correct, correct_active = center_same_formula(
            correct_raw, query_ptr, query_formula, molecule_formula
        )
        structure, _ = center_same_formula(
            structure_raw, query_ptr, query_formula, molecule_formula
        )
        peak, _ = center_same_formula(
            peak_raw, query_ptr, query_formula, molecule_formula
        )
    else:
        correct, correct_active = center_all_candidates(correct_raw, query_ptr)
        structure, _ = center_all_candidates(structure_raw, query_ptr)
        peak, _ = center_all_candidates(peak_raw, query_ptr)

    # All causal arms must be distinguishable on exactly the same queries.
    active = correct_active.copy()
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        if not active[query]:
            continue
        block = slice(int(left), int(right))
        active[query] = bool(
            not np.array_equal(correct[block], structure[block])
            and not np.array_equal(correct[block], peak[block])
        )
    inactive_candidate = np.repeat(~active, np.diff(query_ptr))
    for values in (correct, structure, peak):
        values[inactive_candidate] = 0.0
    formula_fold = stable_formula_folds(query_formula, args.fold_seed)

    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "ledger.npz",
        query_ptr=query_ptr,
        query_row=query_row,
        query_formula=query_formula,
        query_formula_fold=formula_fold,
        query_domain=domains,
        molecule_ik14=molecule_ik14,
        molecule_formula=molecule_formula,
        active_query=active,
        centered_residual=correct,
        structure_swapped_centered_residual=structure,
        peak_permuted_centered_residual=peak,
        query_observation=evidence,
        candidate_predicate_presence=presence,
        complete_same_formula_candidate_structure=complete_structure,
    )
    active_values = np.abs(correct[~inactive_candidate])
    report = {
        "status": "CHEMAWARE_RULE_MATRIX_CANDIDATE_LEDGER_CONFIRMATION_ONLY",
        "formal_training_authorized": False,
        "reason_training_not_authorized": (
            "the fold-0+1 matrix is OOF only for fold 2; training folds require "
            "their own cross-fitted matrices"
        ),
        "teacher_reads_identity_labels_or_dreams_geometry": False,
        "scope": args.scope,
        "cross_formula_candidate_residual_exact_zero": bool(
            np.all(
                correct[
                    molecule_formula
                    != np.repeat(query_formula, np.diff(query_ptr))
                ]
                == 0
            )
        ) if same_formula_only else None,
        "candidate_formula_mean_removed_before_scoring": bool(not same_formula_only),
        "queries": len(query_row),
        "same_formula_candidate_queries": int(np.sum(complete_structure)),
        "matched_three_arm_active_queries": int(np.sum(active)),
        "active_fraction": float(np.mean(active)),
        "active_by_formula_fold": {
            str(fold): int(np.sum(active & (formula_fold == fold))) for fold in range(5)
        },
        "active_target_absolute_quantiles": {
            str(q): float(np.quantile(active_values, q))
            for q in (0.5, 0.9, 0.99, 1.0)
        },
        "inactive_all_arms_exact_zero": bool(
            np.all(correct[inactive_candidate] == 0)
            and np.all(structure[inactive_candidate] == 0)
            and np.all(peak[inactive_candidate] == 0)
        ),
        "fold2_authorized_for_action_utility_audit": True,
        "fold3_embedding_evaluation_untouched": True,
        "fold4_reserve_untouched": True,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
