"""Compile confirmed structure-fragment rules into frozen candidate residuals.

This compiler deliberately does not read identity labels, official retrieval
scores, margins, or input Jacobians when constructing the teacher.  The rule
bank supplies independently confirmed parent-structure/observed-fragment
relations.  Candidate structures are training-only teacher information; the
eventual shared embedding remains spectrum-only at deployment.

The output is a development ledger, not authorization to train.  In
particular, a one-rule bank cannot construct a valid peak-permuted control and
therefore cannot by itself pass the complete chemical-specificity gate.
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

from chemaware_direct_chemical_prior_core import (  # noqa: E402
    compile_structure_fragment_prior,
)
from mine_chemaware_domain_conditioned_action_rules import domain_label  # noqa: E402


RULE_PREDICATE = re.compile(r"RDKIT_FG:(\d+):")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz",
    )
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
        "--rules",
        type=Path,
        default=ROOT
        / "data/validation/chemaware_stratified_meta_action_rules_local_20260907_v3"
        / "stratified_meta_rules.json",
    )
    parser.add_argument(
        "--predicates",
        type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_parent_predicates_rdkit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_direct_chemical_prior_ledger_v1",
    )
    parser.add_argument("--intensity-threshold", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def take_rows(dataset, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows, including duplicates, in caller order."""
    rows = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def exact_lookup(source_key: np.ndarray, target_key: np.ndarray) -> np.ndarray:
    """Map unique integer target keys into source positions, failing closed."""
    source = np.asarray(source_key, dtype=np.int64)
    target = np.asarray(target_key, dtype=np.int64)
    if len(np.unique(source)) != len(source):
        raise RuntimeError("lookup source keys are not unique")
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    position = np.searchsorted(sorted_source, target)
    valid = position < len(sorted_source)
    matched = np.zeros(len(target), dtype=bool)
    matched[valid] = sorted_source[position[valid]] == target[valid]
    if not np.all(matched):
        raise RuntimeError(f"lookup misses {int(np.sum(~matched))} required rows")
    return order[position]


def stable_formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little"
            )
            % folds
            for value in np.asarray(formulas).astype(str)
        ],
        dtype=np.int16,
    )


def parse_rule_bank(rule_path: Path, predicate_path: Path) -> tuple[list[dict], np.ndarray]:
    body = json.loads(rule_path.read_text(encoding="utf-8"))
    rules = list(body.get("rules", []))
    registry = json.loads(predicate_path.read_text(encoding="utf-8"))["predicates"]
    indices: list[int] = []
    for rule in rules:
        match = RULE_PREDICATE.search(str(rule.get("rule_id", "")))
        if match is None:
            raise RuntimeError(f"rule has no parseable predicate index: {rule.get('rule_id')}")
        index = int(match.group(1))
        if index >= len(registry):
            raise RuntimeError(f"predicate index {index} is outside the registry")
        expected = tuple(registry[index]["smarts_any"])
        observed = tuple(rule["parent_predicate"]["smarts_any"])
        if observed != expected:
            raise RuntimeError(f"rule/registry SMARTS mismatch for {rule['rule_id']}")
        if rule["observation"]["kind"] not in {"diagnostic_fragment", "neutral_loss"}:
            raise RuntimeError(f"unsupported observation kind in {rule['rule_id']}")
        indices.append(index)
    if not rules:
        raise RuntimeError("confirmed rule bank is empty")
    return rules, np.asarray(indices, dtype=np.int64)


def rule_confidence(rule: dict) -> float:
    """Conservative lower-CI/point-effect reliability, clipped to [0, 1]."""
    evidence = rule["evidence"]
    effect = float(evidence["confirmation_effect"])
    lower = float(evidence["formula_cluster_bootstrap_95ci"][0])
    return float(np.clip(lower / effect, 0.0, 1.0)) if effect > 0 else 0.0


def spectrum_rule_evidence(
    mz: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    precursor: np.ndarray,
    domains: np.ndarray,
    rules: list[dict],
    intensity_threshold: float,
) -> np.ndarray:
    """Binary evidence matching the rule-mining presence definition."""
    evidence = np.zeros((len(mz), len(rules)), dtype=np.float32)
    for query in range(len(mz)):
        keep = (
            valid[query]
            & np.isfinite(mz[query])
            & np.isfinite(intensity[query])
            & (mz[query] > 0)
            & (intensity[query] >= intensity_threshold)
        )
        peaks = np.asarray(mz[query, keep], dtype=np.float64)
        for rule_index, rule in enumerate(rules):
            if str(domains[query]) not in rule["context"]["supported_acquisition_domains"]:
                continue
            observation = rule["observation"]
            values = (
                peaks
                if observation["kind"] == "diagnostic_fragment"
                else float(precursor[query]) - peaks
            )
            if observation["kind"] == "neutral_loss":
                values = values[values > 0]
            target = float(observation["exact_mass_da"])
            tolerance = float(observation["tolerance_da"])
            observed = len(values) > 0 and np.min(np.abs(values - target)) <= tolerance
            domain_effects = rule["evidence"].get("supported_domain_effects")
            if domain_effects is None:
                raise RuntimeError(
                    f"rule {rule['rule_id']} lacks calibrated supported-domain effects"
                )
            effect = float(domain_effects.get(str(domains[query]), 0.0))
            if effect < 0 or effect > 1:
                raise RuntimeError(f"invalid supported-domain effect in {rule['rule_id']}")
            evidence[query, rule_index] = float(observed) * effect
    return evidence


def candidate_rule_presence(
    smiles: np.ndarray,
    rules: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate training-only parent predicates for every candidate molecule."""
    compiled = []
    for rule in rules:
        queries = [Chem.MolFromSmarts(value) for value in rule["parent_predicate"]["smarts_any"]]
        if any(value is None for value in queries):
            raise RuntimeError(f"invalid SMARTS in {rule['rule_id']}")
        compiled.append(queries)
    presence = np.zeros((len(smiles), len(rules)), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for candidate, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        valid[candidate] = molecule is not None
        if molecule is None:
            continue
        for rule_index, queries in enumerate(compiled):
            presence[candidate, rule_index] = float(
                any(molecule.HasSubstructMatch(query) for query in queries)
            )
    return presence, valid


def deterministic_structure_swap(
    presence: np.ndarray,
    query_ptr: np.ndarray,
    query_row: np.ndarray,
    molecule_formula: np.ndarray,
) -> np.ndarray:
    """Permute structures only among same-formula candidates, label-blind."""
    swapped = presence.copy()
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        block_positions = np.arange(int(left), int(right), dtype=np.int64)
        for formula in np.unique(molecule_formula[block_positions]):
            positions = block_positions[molecule_formula[block_positions] == formula]
            if len(positions) < 2:
                continue
            offset = 1 + int(
                int.from_bytes(
                    hashlib.sha256(
                        f"{int(query_row[query])}|{formula}".encode()
                    ).digest()[:8],
                    "little",
                )
                % (len(positions) - 1)
            )
            swapped[positions] = np.roll(presence[positions], offset, axis=0)
    return swapped


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.intensity_threshold != 0.01:
        raise ValueError("intensity threshold must match the confirmed rule miner (0.01)")
    if args.folds != 5 or args.fold_seed != 20260935:
        raise ValueError("formula-fold protocol must match the confirmed rule miner")

    rules, predicate_indices = parse_rule_bank(args.rules, args.predicates)
    with np.load(args.graph, allow_pickle=True) as graph:
        query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
        pair_candidate_row = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
        query_row = np.asarray(graph["query_row"], dtype=np.int64)
        query_formula = np.asarray(graph["query_formula"]).astype(str)
        molecule_ik14 = np.asarray(graph["molecule_ik14"]).astype(str)
        molecule_formula = np.asarray(graph["molecule_formula"]).astype(str)
    if int(query_ptr[-1]) != len(molecule_ik14):
        raise RuntimeError("query graph does not point to molecule candidates")
    if int(molecule_ptr[-1]) != len(pair_candidate_row):
        raise RuntimeError("molecule graph does not point to reference spectra")
    with np.load(args.manifest, allow_pickle=False) as manifest:
        manifest_row = np.asarray(manifest["query_row"], dtype=np.int64)
        manifest_adduct = np.asarray(manifest["query_adduct"]).astype(str)
    manifest_position = exact_lookup(manifest_row, query_row)
    query_adduct = manifest_adduct[manifest_position]

    token_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    token_position = exact_lookup(token_rows, query_row)
    mz = np.asarray(np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")[token_position])
    intensity = np.asarray(
        np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")[token_position]
    )
    valid = np.asarray(np.load(args.token_dir / "valid.npy", mmap_mode="r")[token_position])
    precursor = np.asarray(
        np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")[token_position],
        dtype=np.float64,
    )

    representative_row = pair_candidate_row[molecule_ptr[:-1]]
    unique_identity, identity_first, identity_inverse = np.unique(
        molecule_ik14, return_index=True, return_inverse=True
    )
    unique_representative_row = representative_row[identity_first]
    with h5py.File(args.data, "r") as handle:
        instrument = decode(take_rows(handle["INSTRUMENT_TYPE"], query_row))
        collision_energy = np.asarray(
            take_rows(handle["COLLISION_ENERGY"], query_row), dtype=np.float64
        )
        unique_candidate_smiles = decode(
            take_rows(handle["smiles"], unique_representative_row)
        )
        candidate_identity = decode(
            take_rows(handle["INCHIKEY"], unique_representative_row)
        )
    candidate_identity = np.asarray([value.split("-")[0] for value in candidate_identity])
    if not np.array_equal(candidate_identity, unique_identity):
        raise RuntimeError("candidate reference structures do not align with graph identities")

    domains = np.asarray(
        [
            domain_label(adduct, instrument_value, energy)
            for adduct, instrument_value, energy in zip(
                query_adduct, instrument, collision_energy
            )
        ]
    )
    evidence = spectrum_rule_evidence(
        mz,
        intensity,
        valid,
        precursor,
        domains,
        rules,
        args.intensity_threshold,
    )
    unique_presence, unique_valid_structure = candidate_rule_presence(
        unique_candidate_smiles, rules
    )
    presence = unique_presence[identity_inverse]
    valid_structure = unique_valid_structure[identity_inverse]

    # A missing candidate structure makes the entire candidate-distribution
    # teacher unidentified.  Fail that query closed instead of treating a
    # missing predicate as false.
    complete_candidate_structure = np.ones(len(query_row), dtype=bool)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        complete_candidate_structure[query] = bool(
            np.all(valid_structure[int(left) : int(right)])
        )
    evidence[~complete_candidate_structure] = 0.0

    confidence = np.asarray([rule_confidence(rule) for rule in rules], dtype=np.float32)
    ungated_prior = compile_structure_fragment_prior(
        evidence,
        presence,
        query_ptr,
        rule_confidence=confidence,
    )
    swapped_presence = deterministic_structure_swap(
        presence, query_ptr, query_row, molecule_formula
    )
    ungated_swapped = compile_structure_fragment_prior(
        evidence,
        swapped_presence,
        query_ptr,
        rule_confidence=confidence,
    )
    structure_control_distinct = np.zeros(len(query_row), dtype=bool)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        structure_control_distinct[query] = not np.array_equal(
            ungated_prior.centered_residual[int(left) : int(right)],
            ungated_swapped.centered_residual[int(left) : int(right)],
        )

    # A structure-swapped control is part of the scientific treatment, not a
    # later reporting decoration.  Queries for which same-formula swapping
    # cannot alter the target are excluded from both correct and control arms.
    evidence[~structure_control_distinct] = 0.0
    prior = compile_structure_fragment_prior(
        evidence,
        presence,
        query_ptr,
        rule_confidence=confidence,
    )
    swapped = compile_structure_fragment_prior(
        evidence,
        swapped_presence,
        query_ptr,
        rule_confidence=confidence,
    )
    formula_fold = stable_formula_folds(query_formula, args.folds, args.fold_seed)

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
        predicate_indices=predicate_indices,
        rule_confidence=confidence,
        query_rule_evidence=evidence,
        candidate_rule_presence=presence,
        candidate_structure_valid=valid_structure,
        complete_candidate_structure=complete_candidate_structure,
        raw_score=prior.raw_score,
        centered_residual=prior.centered_residual,
        active_query=prior.active_query,
        structure_swapped_centered_residual=swapped.centered_residual,
        structure_control_distinct=structure_control_distinct,
    )
    active_by_fold = {
        str(fold): int(np.sum(prior.active_query & (formula_fold == fold)))
        for fold in range(args.folds)
    }
    report = {
        "status": "CHEMAWARE_DIRECT_CHEMICAL_PRIOR_LEDGER_DEVELOPMENT_ONLY",
        "training_authorized": False,
        "reason_training_not_authorized": (
            "the admitted bank has fewer than two independent observation channels, "
            "so a peak-permuted chemical-specificity control is not identifiable"
            if len(rules) < 2
            else "ledger construction alone never authorizes training"
        ),
        "teacher_construction_reads": [
            "query spectrum peaks",
            "precursor and acquisition domain",
            "candidate parent structures",
            "independently confirmed rule bank",
        ],
        "teacher_construction_does_not_read": [
            "identity labels",
            "official DreaMS scores or margins",
            "DreaMS Jacobians",
        ],
        "queries": len(query_row),
        "candidate_molecules": len(molecule_ik14),
        "unique_candidate_identities_parsed": len(unique_identity),
        "rules": len(rules),
        "rules_with_independent_observation_channels": len(
            {(rule["observation"]["kind"], rule["observation"]["exact_mass_da"]) for rule in rules}
        ),
        "queries_in_supported_domain_with_observation_before_matched_control_gate": int(
            np.sum(ungated_prior.active_query)
        ),
        "queries_in_supported_domain_with_observation_after_matched_control_gate": int(
            np.sum(np.any(evidence > 0, axis=1))
        ),
        "queries_with_complete_candidate_structures": int(np.sum(complete_candidate_structure)),
        "candidate_discriminating_active_queries": int(np.sum(prior.active_query)),
        "candidate_discriminating_active_fraction": float(np.mean(prior.active_query)),
        "active_queries_by_formula_fold": active_by_fold,
        "structure_swapped_control_distinct_queries": int(
            np.sum(prior.active_query & structure_control_distinct)
        ),
        "inactive_residual_exact_zero": bool(
            np.array_equal(
                prior.centered_residual[
                    np.repeat(~prior.active_query, np.diff(query_ptr))
                ],
                np.zeros(
                    int(np.sum(np.diff(query_ptr)[~prior.active_query])), dtype=np.float32
                ),
            )
        ),
        "candidate_block_sum_max_abs": float(
            max(
                abs(float(np.sum(prior.centered_residual[int(left) : int(right)])))
                for left, right in zip(query_ptr[:-1], query_ptr[1:])
            )
        ),
        "output": str((args.output / "ledger.npz").resolve()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
