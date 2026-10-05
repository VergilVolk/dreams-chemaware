"""Mine condition-specific empirical rules for ChemAware support actions.

This is the condition-aware successor to the v1 rule miner.  Spectra are never
pooled across adduct, instrument family, or collision-energy band.  Within
each domain, formula-matched identities with and without an RDKit parent
predicate define the contrast.  Formula folds 0-1 discover, fold 2 confirms,
and folds 3-4 remain untouched.  Only positive, observed-peak associations can
become support-action hypotheses; absence/negative associations are never
compiled into executable conflict actions.
"""

from __future__ import annotations

import argparse
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

from chemaware_direct_action_core import formula_bootstrap
from mine_chemaware_structure_conditioned_rules import (
    bh_qvalues,
    decode,
    matched_formula_differences,
    nearest_detect,
    sha256_file,
    sign_flip_pvalue,
    stable_formula_folds,
    take_hdf5_rows,
)

FORMULA_PATTERN = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
SUPPORTED_ADDUCTS = ("[M+H]+", "[M+Na]+")
SUPPORTED_INSTRUMENTS = ("Orbitrap", "QTOF")


def collision_energy_band(value: float) -> str:
    """Predeclared 10--20 unit bands; no outcome-dependent cut point is fitted."""
    if not np.isfinite(value):
        return "missing"
    if value <= 10.0:
        return "very_low_le_10"
    if value <= 20.0:
        return "low_10_20"
    if value <= 30.0:
        return "medium_low_20_30"
    if value <= 40.0:
        return "medium_high_30_40"
    if value <= 60.0:
        return "high_40_60"
    return "very_high_gt_60"


def domain_label(adduct: str, instrument: str, collision_energy: float) -> str:
    instrument_value = instrument if instrument in SUPPORTED_INSTRUMENTS else "unknown"
    return f"{adduct}|{instrument_value}|{collision_energy_band(collision_energy)}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
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
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_domain_conditioned_action_rules_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--discovery-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--intensity-threshold", type=float, default=0.01)
    parser.add_argument("--minimum-discovery-effect", type=float, default=0.08)
    parser.add_argument("--minimum-confirmation-effect", type=float, default=0.05)
    parser.add_argument("--minimum-formulas", type=int, default=12)
    parser.add_argument("--minimum-positive-identities", type=int, default=24)
    parser.add_argument("--minimum-domain-spectra", type=int, default=250)
    parser.add_argument("--max-channels-per-predicate-domain", type=int, default=2)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--spectrum-chunk-size", type=int, default=1024)
    parser.add_argument("--max-query-rows", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists() and not args.preflight_only:
        raise FileExistsError(args.output)
    if (
        args.draws < 10_000
        or args.minimum_formulas < 12
        or args.minimum_positive_identities < 24
        or args.minimum_domain_spectra < 250
    ):
        raise ValueError("domain-conditioned rule-admission thresholds were weakened")
    if args.spectrum_chunk_size < 1:
        raise ValueError("spectrum chunk size must be positive")
    roles = (*args.discovery_folds, args.confirmation_fold)
    if len(set(roles)) != len(roles) or min(roles) < 0 or max(roles) >= args.folds:
        raise ValueError("discovery and confirmation formula folds are invalid")

    predicate_body = json.loads(args.predicates.read_text(encoding="utf-8"))
    observation_body = json.loads(args.observations.read_text(encoding="utf-8"))
    predicates = predicate_body["predicates"]
    channels = observation_body["channels"]
    admissible_channel = np.asarray(
        [
            bool(item.get("observed_species_formulae"))
            and bool(
                FORMULA_PATTERN.fullmatch(str(item["observed_species_formulae"][0]))
            )
            for item in channels
        ]
    )

    with np.load(args.manifest) as manifest:
        query_rows = np.asarray(manifest["query_row"], dtype=np.int64)
        query_identity = np.asarray(manifest["query_ik14"]).astype(str)
        query_formula = np.asarray(manifest["query_formula"]).astype(str)
        query_adduct = np.asarray(manifest["query_adduct"]).astype(str)
    if args.max_query_rows:
        query_rows = query_rows[: args.max_query_rows]
        query_identity = query_identity[: args.max_query_rows]
        query_formula = query_formula[: args.max_query_rows]
        query_adduct = query_adduct[: args.max_query_rows]
    if len(np.unique(query_rows)) != len(query_rows):
        raise RuntimeError("manifest query rows must be unique")
    allowed = np.isin(query_adduct, SUPPORTED_ADDUCTS)
    query_rows = query_rows[allowed]
    query_identity = query_identity[allowed]
    query_formula = query_formula[allowed]
    query_adduct = query_adduct[allowed]

    identity, first, inverse = np.unique(
        query_identity,
        return_index=True,
        return_inverse=True,
    )
    identity_formula = query_formula[first]
    if any(
        len(np.unique(query_formula[inverse == i])) != 1 for i in range(len(identity))
    ):
        raise RuntimeError("one identity maps to multiple parent formulas")
    formula_fold = stable_formula_folds(identity_formula, args.folds, args.fold_seed)

    with h5py.File(args.data, "r") as handle:
        instrument = decode(take_hdf5_rows(handle["INSTRUMENT_TYPE"], query_rows))
        collision_energy = np.asarray(
            take_hdf5_rows(handle["COLLISION_ENERGY"], query_rows),
            dtype=np.float64,
        )
        precursor = np.asarray(
            take_hdf5_rows(handle["precursor_mz"], query_rows),
            dtype=np.float64,
        )
        representative_smiles = decode(
            take_hdf5_rows(handle["smiles"], query_rows[first])
        )

    domain = np.asarray(
        [
            domain_label(adduct, instrument_value, energy)
            for adduct, instrument_value, energy in zip(
                query_adduct,
                instrument,
                collision_energy,
            )
        ]
    )
    unit_key = [
        (int(identity_index), str(domain_value))
        for identity_index, domain_value in zip(inverse, domain)
    ]
    unique_key = sorted(set(unit_key), key=lambda value: (value[1], value[0]))
    unit_position = {value: index for index, value in enumerate(unique_key)}
    spectrum_unit = np.asarray(
        [unit_position[value] for value in unit_key], dtype=np.int64
    )
    unit_identity = np.asarray([value[0] for value in unique_key], dtype=np.int64)
    unit_domain = np.asarray([value[1] for value in unique_key])
    unit_formula = identity_formula[unit_identity]
    unit_fold = formula_fold[unit_identity]
    unit_spectra = np.bincount(spectrum_unit, minlength=len(unique_key)).astype(
        np.int32
    )
    domain_names, domain_spectrum_count = np.unique(domain, return_counts=True)
    sufficiently_large_domains = set(
        domain_names[domain_spectrum_count >= args.minimum_domain_spectra]
    )
    eligible_domains = {
        value
        for value in sufficiently_large_domains
        if "|unknown|" not in value and not value.endswith("|missing")
    }
    excluded_unknown_or_missing_domains = sorted(
        sufficiently_large_domains - eligible_domains
    )

    preflight = {
        "status": "CHEMAWARE_DOMAIN_RULE_MINING_PREFLIGHT_PASS",
        "query_spectra": len(query_rows),
        "identities": len(identity),
        "formulas": len(np.unique(identity_formula)),
        "identity_domain_units": len(unique_key),
        "domains": len(domain_names),
        "eligible_domains": sorted(eligible_domains),
        "excluded_unknown_or_missing_domains": excluded_unknown_or_missing_domains,
        "predicate_hypotheses": len(predicates),
        "observation_hypotheses": len(channels),
        "formula_specified_observations": int(np.sum(admissible_channel)),
        "heldout_formula_folds_never_inspected": sorted(
            set(range(args.folds)) - set(roles)
        ),
        "fold_protocol": {
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "discovery_folds": list(args.discovery_folds),
            "confirmation_fold": args.confirmation_fold,
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    target = np.asarray([item["value_da"] for item in channels], dtype=np.float64)
    tolerance = np.asarray(
        [item["match_tolerance_da"] for item in channels],
        dtype=np.float64,
    )
    neutral = np.asarray([item["match_type"] == "mass_diff" for item in channels])
    observation_sum = np.zeros((len(unique_key), len(channels)), dtype=np.float32)
    row_order = np.argsort(query_rows, kind="stable")
    with h5py.File(args.data, "r") as handle:
        for left in range(0, len(row_order), args.spectrum_chunk_size):
            order = row_order[left : left + args.spectrum_chunk_size]
            spectra = np.asarray(
                handle["spectrum"][query_rows[order]], dtype=np.float32
            )
            for local, query_position in enumerate(order):
                spectrum = spectra[local]
                valid_peak = (
                    np.isfinite(spectrum[0])
                    & np.isfinite(spectrum[1])
                    & (spectrum[0] > 0)
                    & (spectrum[1] >= args.intensity_threshold)
                )
                mz = spectrum[0, valid_peak]
                detected = np.zeros(len(channels), dtype=bool)
                detected[~neutral] = nearest_detect(
                    mz,
                    target[~neutral],
                    tolerance[~neutral],
                )
                loss = precursor[query_position] - mz
                loss = loss[loss > 0]
                detected[neutral] = nearest_detect(
                    loss,
                    target[neutral],
                    tolerance[neutral],
                )
                observation_sum[spectrum_unit[query_position]] += detected
    observation = observation_sum / np.maximum(unit_spectra[:, None], 1)

    molecules = [Chem.MolFromSmiles(value) for value in representative_smiles]
    valid_molecule = np.asarray([value is not None for value in molecules])
    predicate_matrix = np.zeros((len(identity), len(predicates)), dtype=bool)
    for predicate_index, item in enumerate(predicates):
        queries = [Chem.MolFromSmarts(value) for value in item["smarts_any"]]
        if any(value is None for value in queries):
            raise RuntimeError(f"invalid SMARTS in {item['predicate_id']}")
        predicate_matrix[:, predicate_index] = [
            bool(
                molecule is not None
                and any(molecule.HasSubstructMatch(query) for query in queries)
            )
            for molecule in molecules
        ]
    unit_predicate = predicate_matrix[unit_identity]
    valid_unit = valid_molecule[unit_identity]

    discovery_candidates = []
    for domain_value in sorted(eligible_domains):
        domain_positions = np.flatnonzero((unit_domain == domain_value) & valid_unit)
        discovery_positions = domain_positions[
            np.isin(unit_fold[domain_positions], args.discovery_folds)
        ]
        if not len(discovery_positions):
            continue
        for predicate_index, _predicate in enumerate(predicates):
            groups, difference, positive, negative = matched_formula_differences(
                observation,
                unit_predicate[:, predicate_index],
                unit_formula,
                discovery_positions,
            )
            if (
                len(groups) < args.minimum_formulas
                or positive < args.minimum_positive_identities
            ):
                continue
            effect = difference.mean(axis=0)
            allowed_channel = np.flatnonzero(
                admissible_channel & (effect >= args.minimum_discovery_effect)
            )
            order = allowed_channel[np.argsort(-effect[allowed_channel], kind="stable")]
            for channel_index in order[: args.max_channels_per_predicate_domain]:
                discovery_candidates.append(
                    {
                        "domain": domain_value,
                        "predicate_index": predicate_index,
                        "channel_index": int(channel_index),
                        "discovery_effect": float(effect[channel_index]),
                        "discovery_formulas": len(groups),
                        "discovery_positive_identities": int(positive),
                        "discovery_negative_identities": int(negative),
                    }
                )

    confirmation_rows = []
    for candidate_index, candidate in enumerate(discovery_candidates):
        domain_positions = np.flatnonzero(
            (unit_domain == candidate["domain"]) & valid_unit
        )
        confirmation_positions = domain_positions[
            unit_fold[domain_positions] == args.confirmation_fold
        ]
        groups, difference, positive, negative = matched_formula_differences(
            observation,
            unit_predicate[:, candidate["predicate_index"]],
            unit_formula,
            confirmation_positions,
        )
        channel_index = candidate["channel_index"]
        values = difference[:, channel_index] if len(groups) else np.empty(0)
        bootstrap = (
            formula_bootstrap(
                values,
                groups,
                args.fold_seed + candidate_index,
                args.draws,
            )
            if len(values)
            else {
                "formula_macro_mean": 0.0,
                "formula_cluster_bootstrap_95ci": [0.0, 0.0],
                "formula_clusters": 0,
                "draws": args.draws,
            }
        )
        confirmation_rows.append(
            candidate
            | {
                "confirmation_effect": float(np.mean(values)) if len(values) else 0.0,
                "confirmation_formulas": len(groups),
                "confirmation_positive_identities": int(positive),
                "confirmation_negative_identities": int(negative),
                "sign_flip_pvalue": sign_flip_pvalue(
                    values,
                    args.fold_seed + 1000 + candidate_index,
                    args.draws,
                ),
                "bootstrap": bootstrap,
            }
        )

    pvalues = np.asarray(
        [item["sign_flip_pvalue"] for item in confirmation_rows],
        dtype=np.float64,
    )
    qvalues = bh_qvalues(pvalues) if len(pvalues) else pvalues
    admitted = []
    for item, qvalue in zip(confirmation_rows, qvalues):
        item["bh_qvalue"] = float(qvalue)
        passed = (
            item["confirmation_formulas"] >= args.minimum_formulas
            and item["confirmation_positive_identities"]
            >= args.minimum_positive_identities
            and item["confirmation_effect"] >= args.minimum_confirmation_effect
            and item["bootstrap"]["formula_cluster_bootstrap_95ci"][0] > 0
            and qvalue <= args.fdr
        )
        item["admitted"] = bool(passed)
        if not passed:
            continue
        predicate = predicates[item["predicate_index"]]
        channel = channels[item["channel_index"]]
        adduct, instrument_value, energy_band = item["domain"].split("|")
        admitted.append(
            {
                "rule_id": (
                    f"EMP-DOM:{predicate['predicate_id']}:{channel['channel_id']}:"
                    f"{adduct}:{instrument_value}:{energy_band}"
                ),
                "claim_type": "empirical_structure_and_domain_conditioned_support",
                "parent_predicate": {"smarts_any": predicate["smarts_any"]},
                "context": {
                    "ion_mode": "positive",
                    "adduct": adduct,
                    "instrument_family": instrument_value,
                    "collision_energy_band": energy_band,
                },
                "observation": {
                    "kind": (
                        "neutral_loss"
                        if channel["match_type"] == "mass_diff"
                        else "diagnostic_fragment"
                    ),
                    "formula": str(channel["observed_species_formulae"][0]),
                    "exact_mass_da": float(channel["value_da"]),
                    "tolerance_da": float(channel["match_tolerance_da"]),
                },
                "suggested_action": {
                    "operation": "support_boost_observed_match",
                    "may_add_new_mz": False,
                    "negative_or_absence_action_authorized": False,
                    "requires_matched_controls": [
                        "same_domain_predicate_swapped",
                        "intensity_and_mass_matched_observed_peak",
                        "clean_duplicate",
                    ],
                },
                "evidence": {
                    "independent_formulas": int(item["confirmation_formulas"]),
                    "positive_identities": int(
                        item["confirmation_positive_identities"]
                    ),
                    "negative_identities": int(
                        item["confirmation_negative_identities"]
                    ),
                    "confirmation_effect": float(item["confirmation_effect"]),
                    "formula_cluster_bootstrap_95ci": item["bootstrap"][
                        "formula_cluster_bootstrap_95ci"
                    ],
                    "sign_flip_pvalue": float(item["sign_flip_pvalue"]),
                    "bh_qvalue": float(item["bh_qvalue"]),
                },
            }
        )

    report = {
        "status": (
            "CHEMAWARE_DOMAIN_CONDITIONED_RULES_ADMITTED"
            if admitted
            else "CHEMAWARE_DOMAIN_CONDITIONED_RULES_NONE_ADMITTED"
        ),
        "formal_training_authorized": False,
        "preflight": preflight,
        "counts": {
            "valid_smiles_identities": int(np.sum(valid_molecule)),
            "discovery_candidates": len(discovery_candidates),
            "confirmed_rules": len(admitted),
        },
        "thresholds": {
            "minimum_discovery_effect": args.minimum_discovery_effect,
            "minimum_confirmation_effect": args.minimum_confirmation_effect,
            "minimum_formulas": args.minimum_formulas,
            "minimum_positive_identities": args.minimum_positive_identities,
            "minimum_domain_spectra": args.minimum_domain_spectra,
            "fdr_bh": args.fdr,
            "draws": args.draws,
        },
        "fold_protocol": preflight["fold_protocol"],
        "scope": {
            "within_formula_structure_controls": True,
            "adduct_invariant_pooling": False,
            "instrument_invariant_pooling": False,
            "collision_energy_invariant_pooling": False,
            "unknown_instrument_rule_mining": False,
            "missing_collision_energy_rule_mining": False,
            "formula_disjoint_confirmation": True,
            "heldout_formula_folds_inspected": False,
            "negative_association_compiled_as_conflict": False,
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
    (args.output / "conditioned_rules.json").write_text(
        json.dumps(
            {
                "schema": "chemaware_domain_conditioned_action_rules_v1",
                "rules": admitted,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output / "identity_domain_evidence.npz",
        identity=identity,
        identity_formula=identity_formula,
        identity_formula_fold=formula_fold,
        unit_identity=unit_identity,
        unit_formula=unit_formula,
        unit_formula_fold=unit_fold,
        unit_domain=unit_domain,
        unit_spectra=unit_spectra,
        valid_structure_unit=valid_unit,
        predicate_presence=predicate_matrix,
        observation_prevalence=observation.astype(np.float16),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "formal_training_authorized": False,
                "counts": report["counts"],
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
