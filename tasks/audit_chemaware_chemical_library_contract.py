"""Audit the data contracts behind ChemAware before any further training.

This script keeps four evidence sources separate:

1. the MassSpecGym HDF5 used by the existing candidate graphs;
2. a deduplicated representative-spectrum retrieval library (currently ``unified_v3``);
3. the small mass-dense local diagnostic graph;
4. the core and MassBank-derived motif catalogues.

It intentionally does not score or train a model.  Its purpose is to detect
whether a resource is suitable as identity truth, a replicate-positive bank,
a retrieval gallery, or only a diagnostic motif catalogue.
"""
from __future__ import annotations

import argparse
import json
import mmap
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--unified-dir",
        type=Path,
        default=ROOT / "data/reference/unified_v3",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic",
    )
    parser.add_argument(
        "--mass-dense-audit",
        type=Path,
        default=ROOT / "data/validation/mass_dense_factor_cohort_audit.json",
    )
    parser.add_argument(
        "--core-rules",
        type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--massbank-rules",
        type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_massbank.json",
    )
    parser.add_argument(
        "--g0-report",
        type=Path,
        default=ROOT / "tasks/g8r_chemaware_g0_rule_cache.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_chemical_library_contract_audit_v3/report.json",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    return parser.parse_args()


def text_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str)


def integer_histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(map(int, values)).items())
    }


def counter_json(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def strict_candidate_eligibility(
    precursor: np.ndarray,
    adduct: np.ndarray,
    identity: np.ndarray,
    mask: np.ndarray,
    ppm: float,
) -> dict[str, int | float]:
    """Count query rows with both a same-identity reference and a hard negative."""
    selected = np.flatnonzero(mask)
    eligible = 0
    has_positive = 0
    has_negative = 0
    by_adduct: dict[str, dict[str, int]] = {}
    for ion in sorted(set(map(str, adduct[selected]))):
        rows = selected[adduct[selected] == ion]
        order = rows[np.argsort(precursor[rows], kind="stable")]
        masses = precursor[order]
        tolerance = masses * ppm * 1e-6
        total_count = (
            np.searchsorted(masses, masses + tolerance, side="right")
            - np.searchsorted(masses, masses - tolerance, side="left")
        )
        same_count = np.ones(len(order), dtype=np.int64)
        positions_by_identity: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(order):
            positions_by_identity[str(identity[row])].append(position)
        for positions in positions_by_identity.values():
            positions_array = np.asarray(positions, dtype=np.int64)
            group_masses = masses[positions_array]
            group_tolerance = group_masses * ppm * 1e-6
            same_count[positions_array] = (
                np.searchsorted(
                    group_masses, group_masses + group_tolerance, side="right"
                )
                - np.searchsorted(
                    group_masses, group_masses - group_tolerance, side="left"
                )
            )
        positive = same_count >= 2
        negative = total_count > same_count
        local_positive = int(np.sum(positive))
        local_negative = int(np.sum(negative))
        local_eligible = int(np.sum(positive & negative))
        has_positive += local_positive
        has_negative += local_negative
        eligible += local_eligible
        by_adduct[ion] = {
            "spectra": int(len(rows)),
            "has_same_identity_reference": local_positive,
            "has_different_identity_candidate": local_negative,
            "eligible_queries": local_eligible,
        }
    return {
        "spectra": int(len(selected)),
        "has_same_identity_reference": has_positive,
        "has_different_identity_candidate": has_negative,
        "eligible_queries": eligible,
        "eligible_fraction": float(eligible / len(selected)) if len(selected) else 0.0,
        "by_adduct": by_adduct,
    }


def summarize_hdf5(path: Path, ppm: float) -> tuple[dict, set[str]]:
    with h5py.File(path, "r") as handle:
        required = {
            "INCHIKEY", "SIMULATION_CHALLENGE", "fold", "adduct",
            "INSTRUMENT_TYPE", "COLLISION_ENERGY", "precursor_mz",
        }
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise RuntimeError(f"HDF5 is missing fields: {missing}")
        identity = text_array(handle["INCHIKEY"])
        simulation_challenge = text_array(handle["SIMULATION_CHALLENGE"])
        fold = text_array(handle["fold"])
        adduct = text_array(handle["adduct"])
        instrument = text_array(handle["INSTRUMENT_TYPE"])
        collision = np.asarray(handle["COLLISION_ENERGY"][:], dtype=np.float64)
        precursor = np.asarray(handle["precursor_mz"][:], dtype=np.float64)

    group_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in range(len(identity)):
        group_rows[(str(identity[row]), str(adduct[row]))].append(int(row))
    multiplicity = Counter(len(rows) for rows in group_rows.values())
    repeated = [rows for rows in group_rows.values() if len(rows) >= 2]
    instrument_diverse = 0
    ce_observable = 0
    ce_diverse = 0
    for rows in repeated:
        if len(set(map(str, instrument[rows]))) >= 2:
            instrument_diverse += 1
        finite = collision[rows][np.isfinite(collision[rows])]
        if len(finite) >= 2:
            ce_observable += 1
            ce_diverse += int(float(np.max(finite) - np.min(finite)) > 1e-9)

    split_eligibility = {
        name: strict_candidate_eligibility(
            precursor, adduct, identity, fold == name, ppm
        )
        for name in sorted(set(map(str, fold)))
    }
    return ({
        "path": str(path),
        "spectra": int(len(identity)),
        "identity_key_length_histogram": integer_histogram(map(len, identity)),
        "simulation_challenge_membership_counts": counter_json(Counter(simulation_challenge)),
        "simulation_challenge_semantics": (
            "Boolean eligibility mask for the spectrum-simulation benchmark; "
            "not a real-versus-synthetic spectrum provenance field"
        ),
        "fold_counts": counter_json(Counter(fold)),
        "spectra": int(len(identity)),
        "unique_ik14": int(len(set(map(str, identity)))),
        "identity_adduct_groups": int(len(group_rows)),
        "identity_adduct_multiplicity_histogram": integer_histogram(
            value for value, count in multiplicity.items() for _ in range(count)
        ),
        "groups_with_at_least_two_spectra": int(len(repeated)),
        "spectra_in_repeated_groups": int(sum(map(len, repeated))),
        "repeated_groups_with_multiple_instruments": instrument_diverse,
        "repeated_groups_with_two_observed_collision_energies": ce_observable,
        "repeated_groups_with_distinct_observed_collision_energies": ce_diverse,
        "instrument_counts": counter_json(Counter(instrument)),
        "collision_energy_finite": int(np.sum(np.isfinite(collision))),
        "strict_candidate_eligibility": split_eligibility,
        "identity_contract": "IK14 connectivity identity",
        "provenance_limit": (
            "This HDF5 does not contain a source-library field or an explicit "
            "experimental-versus-in-silico provenance field."
        ),
    }, set(map(str, identity)))


_UNIFIED_HEADER = re.compile(
    rb"(?m)^(INCHIKEY|ADDUCT|SOURCE|INSTRUMENT_TYPE|SOURCE_INSTRUMENT|"
    rb"COLLISION_ENERGY|FOLD|SIMULATION_CHALLENGE)=([^\r\n]*)"
)


def scan_unified_headers(path: Path) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Scan aligned identity headers and count optional condition headers."""
    identities: list[str] = []
    adducts: list[str] = []
    sources: list[str] = []
    condition_counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            for match in _UNIFIED_HEADER.finditer(mapped):
                key = match.group(1)
                value = match.group(2).decode("utf-8", "replace")
                if key == b"INCHIKEY":
                    identities.append(value)
                elif key == b"ADDUCT":
                    adducts.append(value)
                elif key == b"SOURCE":
                    sources.append(value)
                else:
                    condition_counts[key.decode("ascii")] += 1
    if not (len(identities) == len(adducts) == len(sources)):
        raise RuntimeError(
            f"unaligned unified headers in {path}: "
            f"IK={len(identities)} adduct={len(adducts)} source={len(sources)}"
        )
    return identities, adducts, sources, dict(condition_counts)


def summarize_unified(directory: Path) -> tuple[dict, set[str]]:
    report = json.loads((directory / "build_report.json").read_text(encoding="utf-8"))
    source = Counter()
    adduct = Counter()
    full_identity = set()
    block_identity = set()
    key_counts = Counter()
    spectra = 0
    condition_presence = Counter()
    for polarity, name in (("pos", "unified_pos.mgf"), ("neg", "unified_neg.mgf")):
        identities, adducts, sources, fields = scan_unified_headers(directory / name)
        condition_presence.update(fields)
        for ik, ion, origin in zip(identities, adducts, sources):
            spectra += 1
            full_identity.add(ik)
            block_identity.add(ik[:14])
            source[origin] += 1
            adduct[ion] += 1
            key_counts[(ik, polarity, ion)] += 1
    multiplicity = Counter(key_counts.values())
    preserved_condition_fields = {
        field: {
            "present": bool(condition_presence.get(field, 0)),
            "records": int(condition_presence.get(field, 0)),
        }
        for field in (
            "INSTRUMENT_TYPE", "SOURCE_INSTRUMENT", "COLLISION_ENERGY",
            "FOLD", "SIMULATION_CHALLENGE",
        )
    }
    return ({
        "path": str(directory),
        "build_report": report,
        "parsed_spectra": spectra,
        "source_counts_after_deduplication": counter_json(source),
        "adduct_counts": counter_json(adduct),
        "unique_full_inchikey": int(len(full_identity)),
        "unique_ik14": int(len(block_identity)),
        "full_inchikey_per_ik14_histogram": integer_histogram(
            Counter(ik[:14] for ik in full_identity).values()
        ),
        "identity_adduct_key_multiplicity_histogram": integer_histogram(
            value for value, count in multiplicity.items() for _ in range(count)
        ),
        "condition_metadata_fields_preserved": preserved_condition_fields,
        "deduplication_contract": (
            "at most max_spectra_per_compound per full-InChIKey/polarity/adduct; "
            "representative chosen by source quality then peak count"
        ),
        "identity_contract": "full 27-character InChIKey in MGF; IK14 recoverable",
    }, block_identity)


def summarize_diagnostic(directory: Path, unified_ik14: set[str]) -> dict:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    with np.load(directory / "graph.npz", allow_pickle=True) as graph:
        query_identity = set(map(str, graph["query_ik14"]))
        molecule_identity = set(map(str, graph["molecule_ik14"]))
        query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
        labels = np.asarray(graph["molecule_label"], dtype=np.int8)
        positive_reference_count = []
        for left, right in zip(query_ptr[:-1], query_ptr[1:]):
            block = labels[left:right]
            positive = int(left + np.flatnonzero(block == 1)[0])
            positive_reference_count.append(
                int(molecule_ptr[positive + 1] - molecule_ptr[positive])
            )
    return {
        "path": str(directory),
        "reported": report,
        "query_identity_count": len(query_identity),
        "candidate_identity_count": len(molecule_identity),
        "query_identity_coverage_in_representative_library": {
            "covered": len(query_identity & unified_ik14),
            "total": len(query_identity),
            "fraction": float(len(query_identity & unified_ik14) / len(query_identity)),
        },
        "candidate_identity_coverage_in_representative_library": {
            "covered": len(molecule_identity & unified_ik14),
            "total": len(molecule_identity),
            "fraction": float(len(molecule_identity & unified_ik14) / len(molecule_identity)),
        },
        "direct_positive_reference_count_histogram": integer_histogram(
            positive_reference_count
        ),
        "benchmark_contract": "mechanism-only mass-dense validation diagnostic",
    }


def summarize_rules(core_path: Path, massbank_path: Path, g0_path: Path) -> dict:
    core_body = json.loads(core_path.read_text(encoding="utf-8"))
    massbank_body = json.loads(massbank_path.read_text(encoding="utf-8"))
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    core = core_body["rules"]
    massbank = massbank_body["rules"]
    return {
        "core": {
            "rules": len(core),
            "categories": counter_json(Counter(rule.get("category") for rule in core)),
            "sources": counter_json(Counter(rule.get("source") for rule in core)),
        },
        "massbank_record_derived": {
            "rules": len(massbank),
            "categories": counter_json(Counter(rule.get("category") for rule in massbank)),
            "support_histogram": integer_histogram(rule.get("support", 0) for rule in massbank),
            "enabled_by_default": counter_json(
                Counter(bool(rule.get("enabled_by_default", False)) for rule in massbank)
            ),
            "recommended_action": counter_json(
                Counter(rule.get("recommended_action", "missing") for rule in massbank)
            ),
            "semantic_warning": (
                "record-level observations are not population-supported chemical rules; "
                "MassBank mass_diff values were generated from abs(precursor_mz-exact_mass), "
                "and CF entries from up to three lowest-m/z peaks"
            ),
        },
        "g0_observed_motif_cache": g0,
        "supervision_contract": (
            "observed spectrum motifs for coverage, conflict, QC, or uncertainty; "
            "not molecule-identity labels and not fragment-structure truth"
        ),
    }


def main() -> None:
    args = arguments()
    required = [
        args.hdf5,
        args.unified_dir / "build_report.json",
        args.unified_dir / "unified_pos.mgf",
        args.unified_dir / "unified_neg.mgf",
        args.diagnostic_dir / "report.json",
        args.diagnostic_dir / "graph.npz",
        args.mass_dense_audit,
        args.core_rules,
        args.massbank_rules,
        args.g0_report,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.ppm <= 0:
        raise ValueError("--ppm must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    hdf5, hdf5_ik14 = summarize_hdf5(args.hdf5, args.ppm)
    unified, unified_ik14 = summarize_unified(args.unified_dir)
    diagnostic = summarize_diagnostic(args.diagnostic_dir, unified_ik14)
    mass_dense = json.loads(args.mass_dense_audit.read_text(encoding="utf-8"))
    rules = summarize_rules(args.core_rules, args.massbank_rules, args.g0_report)

    output = {
        "status": "chemaware_chemical_library_contract_audit_complete",
        "training_was_run": False,
        "massspecgym_hdf5": hdf5,
        "representative_library": unified,
        "mass_dense_local_diagnostic": diagnostic,
        "mass_dense_source_selection": {
            key: mass_dense.get(key)
            for key in (
                "fold", "ppm", "min_ce_difference", "quality_control",
                "cross_condition_units_before_mass_filter", "mass_dense_units",
                "mass_dense_unique_molecules", "directed_negative_links",
                "decision_note",
            )
        },
        "rule_resources": rules,
        "cross_resource_identity": {
            "massspecgym_ik14": len(hdf5_ik14),
            "representative_library_ik14": len(unified_ik14),
            "overlap": len(hdf5_ik14 & unified_ik14),
            "massspecgym_coverage_in_representative_library": float(
                len(hdf5_ik14 & unified_ik14) / len(hdf5_ik14)
            ),
        },
        "decisions": {
            "representative_library_role": (
                "representative retrieval gallery only; not a replicate-positive training bank"
            ),
            "simulation_challenge_semantics_verified": (
                unified.get("build_report", {}).get("schema_semantics", {}).get("SIMULATION_CHALLENGE")
                == "MassSpecGym spectrum-simulation benchmark subset membership; "
                   "not experimental-versus-synthetic provenance and never used as a source filter"
            ),
            "massspecgym_hdf5_role": (
                "current labeled spectrum source with IK14 labels and repeated spectra; "
                "source provenance is absent in this HDF5 and SIMULATION_CHALLENGE is not provenance"
            ),
            "mass_dense_graph_role": (
                "mechanism diagnostic only; selected for cross-condition positives and mass-dense negatives"
            ),
            "massbank_rule_role": (
                "conflict/QC motif catalogue only; prohibited as identity or mechanistic-fragment supervision"
            ),
            "identity_role": (
                "use IK14 for the primary MS/MS identity task; retain full InChIKey for provenance and "
                "report stereochemical ambiguity separately"
            ),
            "next_training_allowed": False,
            "blocking_requirements": [
                "construct a non-deduplicated, metadata-preserving replicate-positive bank",
                "freeze identity/formula/scaffold leakage boundaries before sampling",
                "separate retrieval gallery records from training-only replicate spectra",
                "define the next objective against a spectral-only control on the same candidate graph",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
