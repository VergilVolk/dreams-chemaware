#!/usr/bin/env python
"""Prepare a score-blind Enveda-180 metadata manifest and contamination ledger.

This script never loads model weights or computes spectral similarity. It
validates spectrum/structure metadata, records collision conditions, removes
same-identity exact duplicates, and marks overlap with every supplied consumed
development source. Cross-identity identical peak hashes are emitted as an
explicit exclusion list for the later benchmark builder.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
PROTON = 1.007276466621


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open("r", encoding="utf-8", errors="replace")


def iter_mgf(path: Path):
    fields = None
    peaks = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                fields, peaks = {}, []
            elif line == "END IONS":
                if fields is not None:
                    yield fields, peaks
                fields = None
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()
            elif fields is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, intensity = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0:
                        peaks.append((mz, intensity))


def iter_mgf_headers(path: Path):
    fields = None
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                fields = {}
            elif line == "END IONS":
                if fields is not None:
                    yield fields
                fields = None
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()


def clean(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "none", "null", "nan", "n/a", "na"} else text


def decode(value) -> str:
    return value.decode("utf-8", "ignore").strip() if isinstance(value, bytes) else str(value).strip()


@lru_cache(maxsize=300_000)
def structure(smiles: str, inchikey: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    computed = Chem.MolToInchiKey(mol)
    if len(inchikey) >= 14 and computed[:14] != inchikey[:14]:
        return None
    return (
        computed,
        Chem.MolToSmiles(mol, isomericSmiles=True),
        rdMolDescriptors.CalcMolFormula(mol),
        float(Descriptors.ExactMolWt(mol)),
        Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False)
        if mol.GetNumAtoms() else "",
    )


def collision_energy(value: str) -> float | None:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", clean(value))
    if len(numbers) != 1:
        return None
    result = abs(float(numbers[0]))
    return result if math.isfinite(result) else None


def spectrum_hash(precursor_mz: float, peaks) -> str:
    maximum = max(intensity for _, intensity in peaks)
    digest = hashlib.sha256(f"{precursor_mz:.5f}|".encode())
    for mz, intensity in sorted(peaks):
        digest.update(f"{mz:.4f}:{intensity / maximum:.6f};".encode())
    return digest.hexdigest()


def secondary_spectrum_hash(peaks) -> str:
    """MSnLib-compatible top-128, scale-invariant BLAKE2b signature."""
    array = np.asarray(peaks, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not len(array):
        return ""
    if len(array) > 128:
        chosen = np.argsort(-array[:, 1], kind="stable")[:128]
        array = array[chosen]
    array = array[np.argsort(array[:, 0], kind="stable")]
    array[:, 1] /= array[:, 1].max()
    payload = np.stack(
        (np.round(array[:, 0], 2), np.round(array[:, 1], 4)), axis=1
    )
    return hashlib.blake2b(payload.tobytes(), digest_size=16).hexdigest()


def array_peaks(value) -> list[tuple[float, float]]:
    array = np.asarray(value)
    if array.ndim != 2:
        return []
    if array.shape[0] == 2:
        mzs, intensities = array[0], array[1]
    elif array.shape[1] == 2:
        mzs, intensities = array[:, 0], array[:, 1]
    else:
        return []
    return [
        (float(mz), float(intensity))
        for mz, intensity in zip(mzs, intensities)
        if math.isfinite(float(mz)) and math.isfinite(float(intensity))
        and float(mz) > 0 and float(intensity) > 0
    ]


def load_hdf5(
    path: Path,
    identities: set[str],
    formulas: set[str],
    spectrum_hashes: dict[str, set[str]],
) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    before_i, before_f = len(identities), len(formulas)
    before_s = {key: len(value) for key, value in spectrum_hashes.items()}
    with h5py.File(path, "r") as handle:
        if "INCHIKEY" in handle:
            identities.update(decode(v)[:14] for v in handle["INCHIKEY"][:] if len(decode(v)) >= 14)
        if "FORMULA" in handle:
            formulas.update(decode(v) for v in handle["FORMULA"][:] if decode(v))
        spectrum_key = next((key for key in ("spectrum", "SPECTRUM") if key in handle), None)
        precursor_key = next((key for key in ("precursor_mz", "PRECURSOR_MZ", "PRECURSOR M/Z") if key in handle), None)
        if spectrum_key and precursor_key:
            spectra = handle[spectrum_key]
            precursors = handle[precursor_key]
            if len(spectra) != len(precursors):
                raise RuntimeError(f"HDF5 spectrum/precursor length mismatch: {path}")
            for index in range(len(spectra)):
                peaks = array_peaks(spectra[index])
                precursor = float(np.asarray(precursors[index]).reshape(-1)[0])
                if peaks and math.isfinite(precursor) and precursor > 0:
                    spectrum_hashes["primary_sha256"].add(spectrum_hash(precursor, peaks))
                    spectrum_hashes["secondary_blake2b"].add(secondary_spectrum_hash(peaks))
    return {
        "path": str(path), "status": "loaded", "sha256": sha256_file(path),
        "new_ik14": len(identities)-before_i, "new_formula": len(formulas)-before_f,
        "new_spectrum_hash": {
            key: len(value) - before_s[key] for key, value in spectrum_hashes.items()
        },
    }


def load_csv(
    path: Path,
    identities: set[str],
    formulas: set[str],
    spectrum_hashes: dict[str, set[str]],
) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    before_i, before_f = len(identities), len(formulas)
    before_s = {key: len(value) for key, value in spectrum_hashes.items()}
    frame = pd.read_csv(path, low_memory=False)
    ik_col = next((c for c in ("ik14", "inchikey", "full_inchikey", "INCHIKEY") if c in frame), None)
    formula_col = next((c for c in ("formula", "structure_formula", "FORMULA") if c in frame), None)
    if ik_col:
        identities.update(v[:14] for v in frame[ik_col].fillna("").astype(str) if len(v) >= 14)
    if formula_col:
        formulas.update(v for v in frame[formula_col].fillna("").astype(str) if v)
    hash_col = next((c for c in ("spectrum_hash", "SPECTRUM_HASH") if c in frame), None)
    if hash_col:
        for value in frame[hash_col].fillna("").astype(str):
            if re.fullmatch(r"[0-9a-fA-F]{64}", value):
                spectrum_hashes["primary_sha256"].add(value.lower())
            elif re.fullmatch(r"[0-9a-fA-F]{32}", value):
                spectrum_hashes["secondary_blake2b"].add(value.lower())
    return {
        "path": str(path), "status": "loaded", "sha256": sha256_file(path),
        "new_ik14": len(identities)-before_i, "new_formula": len(formulas)-before_f,
        "new_spectrum_hash": {
            key: len(value) - before_s[key] for key, value in spectrum_hashes.items()
        },
    }


def load_mgf(
    path: Path,
    identities: set[str],
    formulas: set[str],
    spectrum_hashes: dict[str, set[str]],
) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    before_i, before_f = len(identities), len(formulas)
    before_s = {key: len(value) for key, value in spectrum_hashes.items()}
    for fields, peaks in iter_mgf(path):
        key = clean(fields.get("INCHIKEY"))
        if len(key) >= 14:
            identities.add(key[:14])
        value = clean(fields.get("FORMULA"))
        if value:
            formulas.add(value)
        try:
            precursor = float(clean(fields.get("PEPMASS")).split()[0])
        except (ValueError, IndexError):
            precursor = float("nan")
        if peaks and math.isfinite(precursor) and precursor > 0:
            spectrum_hashes["primary_sha256"].add(spectrum_hash(precursor, peaks))
            spectrum_hashes["secondary_blake2b"].add(secondary_spectrum_hash(peaks))
    return {
        "path": str(path), "status": "loaded", "sha256": sha256_file(path),
        "new_ik14": len(identities)-before_i, "new_formula": len(formulas)-before_f,
        "new_spectrum_hash": {
            key: len(value) - before_s[key] for key, value in spectrum_hashes.items()
        },
    }


def load_exclusion_registry(path: Path) -> list[dict]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != "unified_consumed_source_registry_v1":
        raise RuntimeError(f"unexpected exclusion registry schema: {path}")
    sources = body.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("exclusion registry must contain at least one source")
    names = [str(source.get("name", "")) for source in sources]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise RuntimeError("exclusion registry source names must be unique and non-empty")
    return sources


def write_json(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--exclude-hdf5", type=Path, action="append", default=[])
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--exclude-mgf", type=Path, action="append", default=[])
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    identities: set[str] = set()
    formulas: set[str] = set()
    consumed_spectrum_hashes: dict[str, set[str]] = {
        "primary_sha256": set(),
        "secondary_blake2b": set(),
    }
    exclusion_sources = []
    registered = load_exclusion_registry(args.exclusion_registry)
    registered.extend(
        {"name": f"cli_hdf5_{index}", "kind": "hdf5", "path": str(path), "required": True}
        for index, path in enumerate(args.exclude_hdf5)
    )
    registered.extend(
        {"name": f"cli_csv_{index}", "kind": "csv", "path": str(path), "required": True}
        for index, path in enumerate(args.exclude_csv)
    )
    registered.extend(
        {"name": f"cli_mgf_{index}", "kind": "mgf", "path": str(path), "required": True}
        for index, path in enumerate(args.exclude_mgf)
    )
    loaders = {"hdf5": load_hdf5, "csv": load_csv, "mgf": load_mgf}
    for source in registered:
        name = str(source["name"])
        kind = str(source.get("kind", ""))
        source_path = Path(str(source.get("path", "")))
        if kind not in loaders:
            raise RuntimeError(f"unsupported exclusion source kind for {name}: {kind}")
        if bool(source.get("required", True)) and not source_path.is_file():
            raise RuntimeError(f"required exclusion source missing: {name} -> {source_path}")
        result = loaders[kind](source_path, identities, formulas, consumed_spectrum_hashes)
        result.update({"name": name, "kind": kind, "required": bool(source.get("required", True))})
        exclusion_sources.append(result)
    missing_required = [
        row["name"] for row in exclusion_sources
        if row["required"] and row["status"] != "loaded"
    ]
    if missing_required:
        raise RuntimeError(f"required exclusion sources not loaded: {missing_required}")

    columns = [
        "row", "source_row", "title", "inchikey", "ik14", "smiles", "formula",
        "murcko_scaffold", "exact_mass", "precursor_mz", "adduct", "ion_mode",
        "collision_energy", "n_peaks", "spectrum_hash", "spectrum_hash_secondary", "consumed_identity_overlap",
        "consumed_formula_overlap", "consumed_spectrum_overlap", "primary_adduct", "primary_adduct_validated",
        "precursor_structure_ppm",
    ]
    counters = Counter()
    adducts, energies, modes, header_keys = Counter(), Counter(), Counter(), Counter()
    seen_same_identity: set[tuple[str, str]] = set()
    first_hash_identity: dict[str, str] = {}
    conflicting_hashes: set[str] = set()
    manifest_path = args.out / "eligible_records.csv.gz"
    with gzip.open(manifest_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        kept = 0
        for source_row, (fields, peaks) in enumerate(iter_mgf(args.mgf)):
            counters["raw_records"] += 1
            header_keys.update(fields)
            if source_row and source_row % args.progress_every == 0:
                print(f"enveda rows={source_row:,} kept={kept:,}", flush=True)
            if len(peaks) < args.min_peaks:
                counters["too_few_peaks"] += 1
                continue
            smiles, raw_key = clean(fields.get("SMILES")), clean(fields.get("INCHIKEY"))
            parsed = structure(smiles, raw_key)
            if parsed is None:
                counters["invalid_or_conflicting_structure"] += 1
                continue
            inchikey, canonical, formula, exact_mass, scaffold = parsed
            try:
                precursor_mz = float(clean(fields.get("PEPMASS")).split()[0])
            except (ValueError, IndexError):
                counters["invalid_precursor"] += 1
                continue
            energy = collision_energy(fields.get("COLLISION_ENERGIES", fields.get("COLLISION_ENERGY", "")))
            if energy is None:
                counters["missing_collision_energy"] += 1
                continue
            adduct = clean(fields.get("ADDUCT"))
            mode = clean(fields.get("IONMODE"))
            digest = spectrum_hash(precursor_mz, peaks)
            secondary_digest = secondary_spectrum_hash(peaks)
            ik14 = inchikey[:14]
            if (ik14, digest) in seen_same_identity:
                counters["same_identity_duplicate"] += 1
                continue
            seen_same_identity.add((ik14, digest))
            previous = first_hash_identity.setdefault(digest, ik14)
            if previous != ik14:
                conflicting_hashes.add(digest)
            primary = adduct in {"[M+H]+", "[M-H]-"}
            expected_mz = exact_mass + PROTON if adduct == "[M+H]+" else (
                exact_mass - PROTON if adduct == "[M-H]-" else float("nan")
            )
            precursor_ppm = (
                (precursor_mz - expected_mz) / expected_mz * 1e6
                if primary and expected_mz > 0 else float("nan")
            )
            primary_validated = bool(primary and abs(precursor_ppm) <= 30.0)
            row = {
                "row": kept,
                "source_row": source_row,
                "title": clean(fields.get("TITLE")),
                "inchikey": inchikey,
                "ik14": ik14,
                "smiles": canonical,
                "formula": formula,
                "murcko_scaffold": scaffold,
                "exact_mass": exact_mass,
                "precursor_mz": precursor_mz,
                "adduct": adduct,
                "ion_mode": mode,
                "collision_energy": energy,
                "n_peaks": len(peaks),
                "spectrum_hash": digest,
                "spectrum_hash_secondary": secondary_digest,
                "consumed_identity_overlap": ik14 in identities,
                "consumed_formula_overlap": formula in formulas,
                "consumed_spectrum_overlap": (
                    digest in consumed_spectrum_hashes["primary_sha256"]
                    or secondary_digest in consumed_spectrum_hashes["secondary_blake2b"]
                ),
                "primary_adduct": primary,
                "primary_adduct_validated": primary_validated,
                "precursor_structure_ppm": precursor_ppm,
            }
            writer.writerow(row)
            kept += 1
            counters["eligible_metadata_rows"] += 1
            counters["consumed_identity_overlap"] += int(row["consumed_identity_overlap"])
            counters["consumed_formula_overlap"] += int(row["consumed_formula_overlap"])
            counters["consumed_spectrum_overlap"] += int(row["consumed_spectrum_overlap"])
            counters["primary_adduct_rows"] += int(primary)
            counters["primary_adduct_validated_rows"] += int(primary_validated)
            adducts[adduct] += 1
            energies[str(energy)] += 1
            modes[mode] += 1

    conflict_path = args.out / "conflicting_spectrum_hashes.txt.gz"
    with gzip.open(conflict_path, "wt", encoding="utf-8") as handle:
        for value in sorted(conflicting_hashes):
            handle.write(value + "\n")
    report = {
        "status": "ENVEDA180_SCOREBLIND_MANIFEST_COMPLETE",
        "schema": "enveda180_scoreblind_manifest_v1",
        "source_mgf": str(args.mgf),
        "source_mgf_sha256": sha256_file(args.mgf),
        "performance_scores_opened": False,
        "model_loaded": False,
        "exclusion_policy_complete": True,
        "exclusion_registry": str(args.exclusion_registry),
        "exclusion_registry_sha256": sha256_file(args.exclusion_registry),
        "consumed_identity_count": len(identities),
        "consumed_formula_count": len(formulas),
        "consumed_spectrum_hash_count": {
            key: len(value) for key, value in consumed_spectrum_hashes.items()
        },
        "spectrum_hash_contracts": {
            "primary_sha256": "precursor 5dp + all peaks mz 4dp + max-normalized intensity 6dp",
            "secondary_blake2b": "MSnLib-compatible top128, mz 2dp, max-normalized intensity 4dp",
        },
        "exclusion_sources": exclusion_sources,
        "counts": dict(counters),
        "cross_identity_conflicting_hashes": len(conflicting_hashes),
        "adducts": dict(adducts.most_common()),
        "collision_energies": dict(energies.most_common()),
        "ion_modes": dict(modes.most_common()),
        "header_keys": sorted(header_keys),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "conflicting_hashes": str(conflict_path),
        "conflicting_hashes_sha256": sha256_file(conflict_path),
    }
    write_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
