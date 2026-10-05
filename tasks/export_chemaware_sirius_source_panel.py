"""Export the ChemAware training candidate graph as a SIRIUS 6 source panel.

This is a source-quality stage, not an embedding stage.  It exports only the
formula-role-0/1 training queries and their existing candidate structures.
SIRIUS is expected to compute fragmentation trees and probabilistic molecular
fingerprints from each query spectrum; the final ChemAware model will consume
only native identity triplets selected from those frozen results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path,
        default=(ROOT / "data/validation/chemaware_high_coverage_native"
                 / "run_2340524/evidence/train_triplet_evidence.npz"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=(ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1"
                 / "manifest.npz"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=4032)
    parser.add_argument("--expected-formulas", type=int, default=2518)
    parser.add_argument("--minimum-peaks", type=int, default=2)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def charge_from_adduct(adduct: str) -> str:
    if adduct.endswith("+") and not adduct.endswith("2+"):
        return "1+"
    if adduct.endswith("-") and not adduct.endswith("2-"):
        return "1-"
    raise ValueError(f"SIRIUS panel requires a singly charged adduct, got {adduct!r}")


def sirius_instrument_profile(instrument: str) -> str:
    """Map acquisition metadata to a SIRIUS formula-search profile."""
    normalized = str(instrument).strip().lower()
    if "orbitrap" in normalized:
        return "orbitrap"
    if "qtof" in normalized or "q-tof" in normalized or normalized == "tof":
        return "qtof"
    return "default"


def spectrum_peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(spectrum, dtype=np.float64)
    if values.shape[0] != 2:
        raise ValueError(f"expected a 2 x peaks spectrum, got {values.shape}")
    mass, intensity = values
    keep = (
        np.isfinite(mass) & np.isfinite(intensity)
        & (mass > 0.0) & (intensity > 0.0)
    )
    mass = mass[keep]
    intensity = intensity[keep]
    order = np.argsort(mass, kind="stable")
    return mass[order], intensity[order]


def controlled_spectrum(spectrum: np.ndarray, query: int, arm: str) -> np.ndarray:
    """Create deterministic equal-capacity spectral nulls without truth fields."""
    values = np.asarray(spectrum, dtype=np.float64).copy()
    valid = (
        np.isfinite(values[0]) & np.isfinite(values[1])
        & (values[0] > 0.0) & (values[1] > 0.0)
    )
    positions = np.flatnonzero(valid)
    if arm == "correct":
        return values
    if arm == "intensity_rank_permuted":
        shift = 1 + int(query) % max(1, len(positions) - 1)
        values[1, positions] = np.roll(values[1, positions], shift)
        return values
    if arm == "mass_shifted":
        sign = np.where(np.arange(len(positions)) % 2 == 0, 1.0, -1.0)
        offset = 0.37 + 0.02 * (int(query) % 3)
        values[0, positions] = np.maximum(0.01, values[0, positions] + sign * offset)
        return values
    raise ValueError(f"unknown SIRIUS control arm: {arm}")


def sirius_ms_text(
    feature_id: str, precursor_mz: float, adduct: str, formula: str | None,
    instrument: str, collision_energy: float, spectrum: np.ndarray,
) -> str:
    mass, intensity = spectrum_peaks(spectrum)
    instrument_label = str(instrument).strip()
    if instrument_label.lower() in {"", "nan", "none", "unknown"}:
        instrument_label = "Unknown (LCMS)"
    lines = [
        f">compound {feature_id}",
        f">feature_id {feature_id}",
        f">parentmass {precursor_mz:.8f}",
        f">ionization {adduct}",
        f">instrumentation {instrument_label}",
    ]
    if formula is not None:
        lines.insert(4, f">formula {formula}")
    if np.isfinite(collision_energy):
        # In SIRIUS .ms syntax, >collision both records the energy and starts
        # an MS2 block.  A comment such as #COLLISION_ENERGY is ignored.
        lines.append(f">collision {collision_energy:.8g}")
    else:
        lines.append(">ms2")
    lines.extend(
        f"{mz:.8f} {value:.10g}" for mz, value in zip(mass, intensity, strict=True)
    )
    return "\n".join(lines) + "\n"


def connectivity_smiles(smiles: str) -> str:
    """Canonicalize to the connectivity resolution used by IK14 identities."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def choose_structure_representative(smiles_values: set[str]) -> str:
    """Choose a stable, minimally charged IK14 representative for SIRIUS."""
    canonical = {connectivity_smiles(value) for value in smiles_values}
    if not canonical:
        raise ValueError("cannot choose a structure representative from an empty set")

    def score(value: str) -> tuple[int, int, str]:
        molecule = Chem.MolFromSmiles(value)
        assert molecule is not None
        absolute_charge = sum(abs(atom.GetFormalCharge()) for atom in molecule.GetAtoms())
        charged_atoms = sum(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms())
        return absolute_charge, charged_atoms, value

    return min(canonical, key=score)


def smiles_formula(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {smiles!r}")
    return rdMolDescriptors.CalcMolFormula(molecule)


def candidate_rows(
    manifest: Mapping[str, np.ndarray], query: int, local_candidate: int,
) -> np.ndarray:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    molecule = left + int(local_candidate)
    if not left <= molecule < right:
        raise IndexError(f"candidate {local_candidate} outside query {query}")
    pair_left, pair_right = map(
        int, manifest["molecule_ptr"][molecule:molecule + 2],
    )
    return np.asarray(
        manifest["pair_candidate_row"][pair_left:pair_right], dtype=np.int64,
    )


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    if "query" not in evidence:
        raise RuntimeError("training evidence lacks query registry")
    queries = np.asarray(evidence["query"], dtype=np.int64)
    if len(queries) != args.expected_queries or len(np.unique(queries)) != len(queries):
        raise RuntimeError(
            f"expected {args.expected_queries} unique role-0/1 queries, got {len(queries)}"
        )
    formulas = np.asarray(manifest["query_formula"])[queries].astype(str)
    if len(np.unique(formulas)) != args.expected_formulas:
        raise RuntimeError(
            f"expected {args.expected_formulas} role-0/1 formulas, "
            f"got {len(np.unique(formulas))}"
        )
    if np.any(queries < 0) or np.any(queries >= len(manifest["query_row"])):
        raise RuntimeError("query registry falls outside candidate manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_sirius_source_", dir=args.output.parent))
    spectra_dir = temporary / "query_ms"
    spectra_dir.mkdir()
    formula_spectra_dir = temporary / "query_formula_ms"
    formula_spectra_dir.mkdir()
    control_formula_dirs = {
        arm: temporary / f"query_formula_ms_{arm}"
        for arm in ("intensity_rank_permuted", "mass_shifted")
    }
    for directory in control_formula_dirs.values():
        directory.mkdir()
    query_records: list[dict[str, object]] = []
    formula_feature_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    structures: dict[str, dict[str, object]] = {}
    insufficient_peaks = 0
    off_formula_candidate_rows = 0
    queries_with_off_formula = 0
    true_candidate_formula_mismatches = 0

    try:
        with h5py.File(args.data, "r") as data:
            for query, manifest_formula in zip(queries, formulas, strict=True):
                query = int(query)
                row = int(manifest["query_row"][query])
                formula = text(data["FORMULA"][row])
                if formula != str(manifest_formula):
                    raise RuntimeError(
                        f"query {query} formula mismatch: {formula} versus {manifest_formula}"
                    )
                adduct = text(data["adduct"][row])
                precursor_mz = float(data["precursor_mz"][row])
                instrument = text(data["INSTRUMENT_TYPE"][row])
                instrument_profile = sirius_instrument_profile(instrument)
                collision = float(data["COLLISION_ENERGY"][row])
                spectrum = np.asarray(data["spectrum"][row], dtype=np.float64)
                mass, _intensity = spectrum_peaks(spectrum)
                insufficient_peaks += int(len(mass) < args.minimum_peaks)
                feature_id = f"q{query}_r{row}"
                left, right = map(int, manifest["query_ptr"][query:query + 2])
                labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
                if int(np.sum(labels)) != 1:
                    raise RuntimeError(f"query {query} does not have exactly one true candidate")
                candidate_formula_values = [
                    text(value)
                    for value in manifest["molecule_formula"][left:right]
                ]
                if any(not value for value in candidate_formula_values):
                    raise RuntimeError(f"query {query} has an empty candidate formula")
                true_formula = candidate_formula_values[int(np.flatnonzero(labels)[0])]
                true_candidate_formula_mismatches += int(true_formula != formula)
                unique_candidate_formulas = sorted(set(candidate_formula_values))
                off_formula_count = int(sum(
                    value != formula for value in candidate_formula_values
                ))
                off_formula_candidate_rows += off_formula_count
                queries_with_off_formula += int(off_formula_count > 0)

                # Deliberately omit the known true formula.  Each feature must be
                # submitted to SIRIUS with the complete per-query candidate-formula
                # set from query_registry.tsv; fixing `formula` here would remove
                # real off-formula competitors and leak the answer.
                query_profile_dir = spectra_dir / instrument_profile
                query_profile_dir.mkdir(exist_ok=True)
                ms_path = query_profile_dir / f"{feature_id}.ms"
                ms_path.write_text(
                    sirius_ms_text(
                        feature_id, precursor_mz, adduct, None,
                        instrument, collision, spectrum,
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                # Exhaustive fixed-formula replicas guarantee that every formula
                # in the formal candidate graph receives a fragmentation tree and
                # fingerprint.  All formulas are emitted, so fixing a formula in
                # each replica does not reveal which candidate is correct.
                for formula_index, candidate_formula in enumerate(
                    unique_candidate_formulas
                ):
                    formula_feature_id = f"{feature_id}_f{formula_index:02d}"
                    formula_profile_dir = formula_spectra_dir / instrument_profile
                    formula_profile_dir.mkdir(exist_ok=True)
                    formula_ms_path = formula_profile_dir / f"{formula_feature_id}.ms"
                    formula_ms_path.write_text(
                        sirius_ms_text(
                            formula_feature_id, precursor_mz, adduct,
                            candidate_formula, instrument, collision, spectrum,
                        ),
                        encoding="utf-8",
                        newline="\n",
                    )
                    formula_feature_records.append({
                        "formula_feature_id": formula_feature_id,
                        "intensity_rank_permuted_feature_id": (
                            f"{formula_feature_id}__intensity_rank_permuted"
                        ),
                        "mass_shifted_feature_id": f"{formula_feature_id}__mass_shifted",
                        "feature_id": feature_id,
                        "manifest_query": query,
                        "candidate_formula": candidate_formula,
                        "instrument_profile": instrument_profile,
                    })
                    for arm, directory in control_formula_dirs.items():
                        control_feature_id = f"{formula_feature_id}__{arm}"
                        control_profile_dir = directory / instrument_profile
                        control_profile_dir.mkdir(exist_ok=True)
                        (control_profile_dir / f"{control_feature_id}.ms").write_text(
                            sirius_ms_text(
                                control_feature_id, precursor_mz, adduct,
                                candidate_formula, instrument, collision,
                                controlled_spectrum(spectrum, query, arm),
                            ),
                            encoding="utf-8", newline="\n",
                        )
                query_records.append({
                    "feature_id": feature_id,
                    "manifest_query": query,
                    "spectrum_row": row,
                    "adduct": adduct,
                    "precursor_mz": f"{precursor_mz:.8f}",
                    "instrument": instrument,
                    "instrument_profile": instrument_profile,
                    "collision_energy": "" if not np.isfinite(collision) else f"{collision:.8g}",
                    "peaks": int(len(mass)),
                    "candidate_count": int(right - left),
                    "candidate_formula_count": int(len(unique_candidate_formulas)),
                    "candidate_formulas": ";".join(unique_candidate_formulas),
                })
                for candidate in range(right - left):
                    molecule = left + candidate
                    rows = candidate_rows(manifest, query, candidate)
                    if not len(rows):
                        raise RuntimeError(f"query {query} candidate {candidate} has no references")
                    ik14 = str(manifest["molecule_ik14"][molecule])
                    candidate_formula = str(manifest["molecule_formula"][molecule])
                    full_keys = {text(data["INCHIKEY"][int(index)]) for index in rows}
                    if not full_keys or any(key[:14] != ik14 for key in full_keys):
                        raise RuntimeError(
                            f"query {query} candidate {candidate} InChIKey mismatch"
                        )
                    smiles_variants = {
                        connectivity_smiles(text(data["smiles"][int(index)]))
                        for index in rows
                    }
                    structure = choose_structure_representative(smiles_variants)
                    if smiles_formula(structure) != candidate_formula:
                        raise RuntimeError(
                            f"query {query} candidate {candidate} structure/formula "
                            f"mismatch: {smiles_formula(structure)} versus {candidate_formula}"
                        )
                    prior = structures.setdefault(ik14, {
                        "representative": structure,
                        "variants": set(),
                        "full_keys": set(),
                        "formula": candidate_formula,
                    })
                    if prior["formula"] != candidate_formula:
                        raise RuntimeError(f"connectivity {ik14} maps to multiple formulas")
                    prior_variants = prior["variants"]
                    prior_keys = prior["full_keys"]
                    assert isinstance(prior_variants, set)
                    assert isinstance(prior_keys, set)
                    prior_variants.update(smiles_variants)
                    prior_keys.update(full_keys)
                    prior["representative"] = choose_structure_representative(
                        prior_variants
                    )
                    candidate_records.append({
                        "feature_id": feature_id,
                        "manifest_query": query,
                        "local_candidate": candidate,
                        "ik14": ik14,
                        "full_inchikey_variants": ";".join(sorted(full_keys)),
                        "formula": candidate_formula,
                        "smiles": structure,
                        "smiles_variants_json": json.dumps(
                            sorted(smiles_variants), separators=(",", ":")
                        ),
                        "reference_count": int(len(rows)),
                        "reference_rows": ";".join(map(str, rows.tolist())),
                    })

        for record in candidate_records:
            record["smiles"] = structures[str(record["ik14"])]["representative"]

        candidates_per_query_formula: dict[tuple[int, str], int] = {}
        for record in candidate_records:
            key = (int(record["manifest_query"]), str(record["formula"]))
            candidates_per_query_formula[key] = candidates_per_query_formula.get(key, 0) + 1
        maximum_candidates_per_query_formula = max(candidates_per_query_formula.values())
        global_structures_per_formula: dict[str, int] = {}
        for body in structures.values():
            formula = str(body["formula"])
            global_structures_per_formula[formula] = (
                global_structures_per_formula.get(formula, 0) + 1
            )
        maximum_global_structures_per_formula = max(
            global_structures_per_formula.values()
        )

        with (temporary / "query_registry.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(query_records[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(query_records)
        with (temporary / "candidate_ledger.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(candidate_records[0]), delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(candidate_records)
        with (temporary / "formula_feature_registry.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(formula_feature_records[0]), delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(formula_feature_records)
        # SIRIUS custom-db accepts a headerless SMILES, id, name TSV.
        with (temporary / "candidate_structures.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for ik14 in sorted(structures):
                structure = structures[ik14]["representative"]
                writer.writerow((structure, ik14, ik14))

        report = {
            "status": "CHEMAWARE_SIRIUS_SOURCE_PANEL_COMPLETE",
            "scope": "formula roles 0/1 only; roles 2/3/4 untouched",
            "queries": int(len(query_records)),
            "query_formula_features": int(len(formula_feature_records)),
            "controlled_query_formula_features": int(
                len(formula_feature_records) * len(control_formula_dirs)
            ),
            "formulas": int(len(np.unique(formulas))),
            "candidate_rows": int(len(candidate_records)),
            "unique_candidate_connectivities": int(len(structures)),
            "maximum_candidates_per_query_formula": int(
                maximum_candidates_per_query_formula
            ),
            "maximum_global_structures_per_formula": int(
                maximum_global_structures_per_formula
            ),
            "off_formula_candidate_rows": int(off_formula_candidate_rows),
            "off_formula_candidate_fraction": (
                float(off_formula_candidate_rows / len(candidate_records))
            ),
            "queries_with_off_formula_candidate": int(queries_with_off_formula),
            "queries_with_off_formula_candidate_fraction": (
                float(queries_with_off_formula / len(query_records))
            ),
            "true_candidate_formula_mismatches": int(
                true_candidate_formula_mismatches
            ),
            "queries_below_minimum_peaks": int(insufficient_peaks),
            "minimum_peaks": int(args.minimum_peaks),
            "adduct_counts": {
                value: int(sum(row["adduct"] == value for row in query_records))
                for value in sorted({str(row["adduct"]) for row in query_records})
            },
            "instrument_profile_query_counts": {
                value: int(sum(row["instrument_profile"] == value for row in query_records))
                for value in sorted({str(row["instrument_profile"]) for row in query_records})
            },
            "instrument_profile_formula_feature_counts": {
                value: int(sum(
                    row["instrument_profile"] == value for row in formula_feature_records
                ))
                for value in sorted({
                    str(row["instrument_profile"]) for row in formula_feature_records
                })
            },
            "inputs": {
                "evidence": str(args.evidence.resolve()),
                "evidence_sha256": sha256(args.evidence),
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256(args.manifest),
                "data": str(args.data.resolve()),
            },
            "outputs": {},
            "scientific_contract": {
                "formula_search": (
                    "known true formula is omitted from query_ms; query_formula_ms "
                    "contains one replica for every candidate formula, never only "
                    "the true formula"
                ),
                "candidate_space": "exact existing ChemAware candidate graph",
                "truth_blind_source_artifacts": (
                    "query_registry and candidate_ledger contain no candidate label"
                ),
                "two_layer_score": (
                    "raw fragmentation-tree evidence across fixed-formula replicas, "
                    "then CSI structure evidence within formula"
                ),
                "matched_controls": (
                    "each fixed-formula replica has intensity-rank-permuted and "
                    "mass-shifted spectra; candidate-role swapping is applied "
                    "after score import"
                ),
                "instrument_profiles": (
                    "correct and both spectral controls are sharded identically into "
                    "orbitrap, qtof, or no-explicit-profile runs"
                ),
                "deployment": "SIRIUS is training-time triplet supervision only",
                "next_gate": (
                    "candidate SIRIUS margin must add formula-OOF discrimination "
                    "conditional on the frozen Phase-A margin"
                ),
            },
            "gates": {
                "true_candidate_formula_matches_query": (
                    true_candidate_formula_mismatches == 0
                ),
                "known_true_formula_omitted_from_ms": True,
                "candidate_formula_sets_preserved": True,
                "all_candidate_formulas_materialized": (
                    len(formula_feature_records)
                    == sum(int(row["candidate_formula_count"]) for row in query_records)
                ),
                "truth_labels_omitted_from_source_ledgers": True,
                "matched_control_cardinality": all(
                    len(list(directory.rglob("*.ms"))) == len(formula_feature_records)
                    for directory in control_formula_dirs.values()
                ),
                "instrument_profile_cardinality": all(
                    len(list((directory / profile).glob("*.ms"))) == count
                    for directory in (formula_spectra_dir, *control_formula_dirs.values())
                    for profile, count in {
                        value: int(sum(
                            row["instrument_profile"] == value
                            for row in formula_feature_records
                        ))
                        for value in {
                            str(row["instrument_profile"])
                            for row in formula_feature_records
                        }
                    }.items()
                ),
                "all_global_formula_structures_fit_top_k_50_summary": (
                    maximum_global_structures_per_formula <= 50
                ),
            },
        }
        if not all(report["gates"].values()):
            raise RuntimeError(f"SIRIUS source-panel gates failed: {report['gates']}")
        for name in (
            "query_registry.tsv", "formula_feature_registry.tsv",
            "candidate_ledger.tsv", "candidate_structures.tsv",
        ):
            report["outputs"][name] = {
                "sha256": sha256(temporary / name),
                "bytes": int((temporary / name).stat().st_size),
            }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8", newline="\n",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
