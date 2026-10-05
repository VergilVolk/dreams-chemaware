"""Mine empirical structure-conditioned spectral rules without evaluation leakage.

RDKit functional groups are parent-predicate hypotheses and the 120 ChemAware
channels are spectral-observation hypotheses.  Formula folds 0-1 select at
most a few channels per predicate; fold 2 confirms them with within-formula
structure-negative controls, sign-flip p-values, BH-FDR, and formula-cluster
bootstrap intervals.  Folds 3 and 4 are never inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_direct_action_core import formula_bootstrap  # noqa: E402


FORMULA_PATTERN = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def take_hdf5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary unique HDF5 rows while preserving caller order."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    if len(np.unique(sorted_rows)) != len(sorted_rows):
        raise ValueError("HDF5 row selection contains duplicates")
    sorted_values = np.asarray(dataset[sorted_rows])
    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return sorted_values[restore]


def stable_formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "little") % folds
        for value in np.asarray(formulas).astype(str)
    ], dtype=np.int16)


def nearest_detect(values: np.ndarray, targets: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.float64))
    targets = np.asarray(targets, dtype=np.float64)
    tolerance = np.asarray(tolerance, dtype=np.float64)
    if not len(values):
        return np.zeros(len(targets), dtype=bool)
    right = np.searchsorted(values, targets, side="left")
    left = np.clip(right - 1, 0, len(values) - 1)
    right = np.clip(right, 0, len(values) - 1)
    distance = np.minimum(np.abs(values[left] - targets), np.abs(values[right] - targets))
    return distance <= tolerance


def matched_formula_differences(
    observation: np.ndarray,
    predicate: np.ndarray,
    formula: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    observation = np.asarray(observation, dtype=np.float64)
    predicate = np.asarray(predicate, dtype=bool)
    formula = np.asarray(formula).astype(str)
    positions = np.asarray(positions, dtype=np.int64)
    groups = []
    differences = []
    positive_molecules = 0
    negative_molecules = 0
    local_formula = formula[positions]
    order = np.argsort(local_formula, kind="stable")
    ordered_positions = positions[order]
    ordered_formula = local_formula[order]
    boundaries = np.r_[0, np.flatnonzero(ordered_formula[1:] != ordered_formula[:-1]) + 1, len(order)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = ordered_positions[left:right]
        value = ordered_formula[left]
        positive = group[predicate[group]]
        negative = group[~predicate[group]]
        if not len(positive) or not len(negative):
            continue
        groups.append(value)
        differences.append(
            observation[positive].mean(axis=0) - observation[negative].mean(axis=0)
        )
        positive_molecules += len(positive)
        negative_molecules += len(negative)
    width = observation.shape[1]
    matrix = np.vstack(differences) if differences else np.empty((0, width), dtype=np.float64)
    return np.asarray(groups), matrix, int(positive_molecules), int(negative_molecules)


def sign_flip_pvalue(values: np.ndarray, seed: int, draws: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 1.0
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        take = min(1000, remaining)
        sign = rng.integers(0, 2, size=(take, len(values)), dtype=np.int8) * 2 - 1
        exceed += int(np.sum((sign * values).mean(axis=1) >= observed))
        remaining -= take
    return float((exceed + 1) / (draws + 1))


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues, kind="stable")
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--predicates", type=Path, default=ROOT / "dreams/models/chem_aware/chem_parent_predicates_rdkit_v1.json")
    parser.add_argument("--observations", type=Path, default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_empirical_structure_rules_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--discovery-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--intensity-threshold", type=float, default=0.01)
    parser.add_argument("--minimum-discovery-effect", type=float, default=0.05)
    parser.add_argument("--minimum-confirmation-effect", type=float, default=0.03)
    parser.add_argument("--minimum-formulas", type=int, default=10)
    parser.add_argument("--minimum-positive-molecules", type=int, default=20)
    parser.add_argument("--max-channels-per-predicate", type=int, default=3)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--max-query-rows", type=int, default=0,
                        help="Development smoke only; zero uses every manifest query row.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.draws < 10_000 or args.minimum_formulas < 10 or args.minimum_positive_molecules < 20:
        raise ValueError("empirical-rule admission thresholds were weakened")
    predicates_body = json.loads(args.predicates.read_text(encoding="utf-8"))
    observations_body = json.loads(args.observations.read_text(encoding="utf-8"))
    predicates = predicates_body["predicates"]
    channels = observations_body["channels"]
    admissible_channel = np.asarray([
        bool(item.get("observed_species_formulae"))
        and bool(FORMULA_PATTERN.fullmatch(str(item["observed_species_formulae"][0])))
        for item in channels
    ])
    with np.load(args.manifest) as manifest:
        query_rows = np.asarray(manifest["query_row"], dtype=np.int64)
        query_identity = np.asarray(manifest["query_ik14"]).astype(str)
        query_formula = np.asarray(manifest["query_formula"]).astype(str)
        query_adduct = np.asarray(manifest["query_adduct"]).astype(str)
    if args.max_query_rows:
        query_rows = query_rows[:args.max_query_rows]
        query_identity = query_identity[:args.max_query_rows]
        query_formula = query_formula[:args.max_query_rows]
        query_adduct = query_adduct[:args.max_query_rows]
    if len(np.unique(query_rows)) != len(query_rows):
        raise RuntimeError("manifest query rows are not unique")
    allowed_adduct = np.isin(query_adduct, ("[M+H]+", "[M+Na]+"))
    query_rows = query_rows[allowed_adduct]
    query_identity = query_identity[allowed_adduct]
    query_formula = query_formula[allowed_adduct]
    query_adduct = query_adduct[allowed_adduct]
    identity, first, inverse = np.unique(query_identity, return_index=True, return_inverse=True)
    identity_formula = query_formula[first]
    if any(len(np.unique(query_formula[inverse == i])) != 1 for i in range(len(identity))):
        raise RuntimeError("one identity maps to multiple formulas")
    fold = stable_formula_folds(identity_formula, args.folds, args.fold_seed)
    discovery = np.flatnonzero(np.isin(fold, args.discovery_folds))
    confirmation = np.flatnonzero(fold == args.confirmation_fold)
    heldout = np.flatnonzero(~np.isin(fold, (*args.discovery_folds, args.confirmation_fold)))
    preflight = {
        "status": "CHEMAWARE_STRUCTURE_RULE_MINING_PREFLIGHT_PASS",
        "query_rows": int(len(query_rows)), "identities": int(len(identity)),
        "formulas": int(len(np.unique(identity_formula))),
        "predicate_hypotheses": len(predicates), "observation_hypotheses": len(channels),
        "chemically_formula_specified_observations": int(np.sum(admissible_channel)),
        "discovery_identities": int(len(discovery)),
        "confirmation_identities": int(len(confirmation)),
        "heldout_identities_never_inspected": int(len(heldout)),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2)); return

    observation_sum = np.zeros((len(identity), len(channels)), dtype=np.float32)
    spectrum_count = np.zeros(len(identity), dtype=np.int32)
    target = np.asarray([item["value_da"] for item in channels], dtype=np.float64)
    tolerance = np.asarray([item["match_tolerance_da"] for item in channels], dtype=np.float64)
    neutral = np.asarray([item["match_type"] == "mass_diff" for item in channels])
    with h5py.File(args.data, "r") as handle:
        smiles = decode(take_hdf5_rows(handle["smiles"], query_rows))
        precursor = np.asarray(take_hdf5_rows(handle["precursor_mz"], query_rows), dtype=np.float64)
        spectra = np.asarray(take_hdf5_rows(handle["spectrum"], query_rows), dtype=np.float32)
        for position, spectrum in enumerate(spectra):
            valid = (spectrum[0] > 0) & (spectrum[1] >= args.intensity_threshold)
            mz = spectrum[0, valid]
            detected = np.zeros(len(channels), dtype=bool)
            detected[~neutral] = nearest_detect(mz, target[~neutral], tolerance[~neutral])
            loss = precursor[position] - mz
            loss = loss[loss > 0]
            detected[neutral] = nearest_detect(loss, target[neutral], tolerance[neutral])
            observation_sum[inverse[position]] += detected
            spectrum_count[inverse[position]] += 1
    observation = observation_sum / np.maximum(spectrum_count[:, None], 1)
    representative_smiles = smiles[first]
    query_molecules = [Chem.MolFromSmiles(value) for value in representative_smiles]
    valid_molecule = np.asarray([value is not None for value in query_molecules])
    predicate_matrix = np.zeros((len(identity), len(predicates)), dtype=bool)
    for predicate_index, item in enumerate(predicates):
        query = Chem.MolFromSmarts(item["smarts_any"][0])
        predicate_matrix[:, predicate_index] = [
            bool(molecule is not None and molecule.HasSubstructMatch(query))
            for molecule in query_molecules
        ]

    discovery_candidates = []
    for predicate_index, item in enumerate(predicates):
        groups, difference, positive, negative = matched_formula_differences(
            observation, predicate_matrix[:, predicate_index], identity_formula, discovery,
        )
        if len(groups) < args.minimum_formulas or positive < args.minimum_positive_molecules:
            continue
        effect = difference.mean(axis=0)
        allowed = np.flatnonzero(admissible_channel & (effect >= args.minimum_discovery_effect))
        order = allowed[np.argsort(-effect[allowed], kind="stable")]
        for channel_index in order[:args.max_channels_per_predicate]:
            discovery_candidates.append({
                "predicate_index": predicate_index, "channel_index": int(channel_index),
                "discovery_effect": float(effect[channel_index]),
                "discovery_formulas": int(len(groups)),
                "discovery_positive_molecules": positive,
                "discovery_negative_molecules": negative,
            })

    confirmation_rows = []
    for candidate_index, candidate in enumerate(discovery_candidates):
        predicate_index = candidate["predicate_index"]
        channel_index = candidate["channel_index"]
        groups, difference, positive, negative = matched_formula_differences(
            observation, predicate_matrix[:, predicate_index], identity_formula, confirmation,
        )
        values = difference[:, channel_index] if len(groups) else np.empty(0)
        ci = formula_bootstrap(values, groups, args.fold_seed + candidate_index, args.draws) if len(groups) else {
            "formula_macro_mean": 0.0, "formula_cluster_bootstrap_95ci": [0.0, 0.0],
            "formula_clusters": 0, "draws": args.draws,
        }
        confirmation_rows.append(candidate | {
            "confirmation_effect": float(np.mean(values)) if len(values) else 0.0,
            "confirmation_formulas": int(len(groups)),
            "confirmation_positive_molecules": positive,
            "confirmation_negative_molecules": negative,
            "pvalue": sign_flip_pvalue(values, args.fold_seed + 1000 + candidate_index, args.draws),
            "bootstrap": ci,
        })
    pvalues = np.asarray([item["pvalue"] for item in confirmation_rows], dtype=np.float64)
    qvalues = bh_qvalues(pvalues) if len(pvalues) else pvalues
    admitted_rules = []
    for index, (item, qvalue) in enumerate(zip(confirmation_rows, qvalues)):
        item["qvalue_bh"] = float(qvalue)
        passed = (
            item["confirmation_formulas"] >= args.minimum_formulas
            and item["confirmation_positive_molecules"] >= args.minimum_positive_molecules
            and item["confirmation_effect"] >= args.minimum_confirmation_effect
            and item["bootstrap"]["formula_cluster_bootstrap_95ci"][0] > 0
            and qvalue <= args.fdr
        )
        item["admitted"] = bool(passed)
        if not passed:
            continue
        predicate = predicates[item["predicate_index"]]
        channel = channels[item["channel_index"]]
        species_formula = str(channel["observed_species_formulae"][0])
        citation = "; ".join(map(str, channel.get("citation_text", []))) or "source registry record"
        admitted_rules.append({
            "rule_id": f"EMP:{predicate['predicate_id']}:{channel['channel_id']}",
            "claim_type": "empirical_structure_conditioned",
            "parent_predicate": {"smarts_any": predicate["smarts_any"]},
            "context": {"ion_mode": "positive", "adducts": channel["adduct_scope"]},
            "observation": {
                "kind": "neutral_loss" if channel["match_type"] == "mass_diff" else "diagnostic_fragment",
                "formula": species_formula, "exact_mass_da": float(channel["value_da"]),
                "tolerance_da": float(channel["match_tolerance_da"]),
            },
            "suggested_action": {
                "operation": "support_boost_observed_match", "may_add_new_mz": False,
                "requires_matched_controls": ["predicate_swapped", "matched_random_observed_peak", "clean_duplicate"],
            },
            "evidence": {
                "source_id": f"{predicate['predicate_id']}+{channel['channel_id']}",
                "citation": citation,
                "independent_molecules": int(item["confirmation_positive_molecules"]),
                "independent_formulas": int(item["confirmation_formulas"]),
                "formula_disjoint_confirmation_pass": True,
                "confirmation_effect": float(item["confirmation_effect"]),
                "sign_flip_pvalue": float(item["pvalue"]),
                "bh_qvalue": float(item["qvalue_bh"]),
                "bootstrap_95ci": item["bootstrap"]["formula_cluster_bootstrap_95ci"],
            },
        })
    report = {
        "status": "CHEMAWARE_EMPIRICAL_RULES_ADMITTED" if admitted_rules else "CHEMAWARE_EMPIRICAL_RULES_NONE_ADMITTED",
        "formal_training_authorized": False,
        "preflight": preflight,
        "counts": {
            "valid_smiles_identities": int(np.sum(valid_molecule)),
            "discovery_candidates": len(discovery_candidates),
            "confirmed_rules": len(admitted_rules),
        },
        "thresholds": {
            "minimum_formulas": args.minimum_formulas,
            "minimum_positive_molecules": args.minimum_positive_molecules,
            "minimum_discovery_effect": args.minimum_discovery_effect,
            "minimum_confirmation_effect": args.minimum_confirmation_effect,
            "fdr_bh": args.fdr, "draws": args.draws,
        },
        "scope": {
            "within_formula_structure_negative_controls": True,
            "formula_disjoint_confirmation": True,
            "embedding_evaluation_folds_inspected": False,
            "outer_fold_inspected": False,
            "association_not_mechanistic_claim": True,
        },
        "candidates": confirmation_rows,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "hdf5_sha256": sha256_file(args.data),
            "predicate_registry_sha256": sha256_file(args.predicates),
            "observation_registry_sha256": sha256_file(args.observations),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "curated_rules.json").write_text(
        json.dumps({"schema": "chemaware_empirical_rules_v1", "rules": admitted_rules}, indent=2),
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "identity_evidence.npz",
        identity=identity, formula=identity_formula, formula_fold=fold,
        predicate_presence=predicate_matrix,
        observation_prevalence=observation.astype(np.float16),
    )
    print(json.dumps({key: report[key] for key in ("status", "counts", "scope", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
