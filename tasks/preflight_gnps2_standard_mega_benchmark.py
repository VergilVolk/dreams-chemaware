#!/usr/bin/env python
"""Truth-blind preflight for a strict GNPS2 Standard identity benchmark.

This module performs metadata and candidate-axis construction only.  It never
loads a model, an embedding, a prediction, or an evaluation score.  Its main
purpose is to answer, before expensive encoding, whether a set of GNPS2
standard-library MGF files can support a labelled, overlap-excluded identity
benchmark and how large that benchmark would actually be.

The report deliberately distinguishes five units:

* ``N_Q`` labelled query spectra;
* ``N_I`` unique query IK14 identity clusters;
* ``N_F`` unique query formula clusters;
* ``N_E`` query-candidate-molecule edges; and
* ``N_P`` query-reference-spectrum comparisons.

Only ``N_P`` can plausibly exceed ten million in the current public data.  It
must never be described as ten million labelled queries.

GNPS2 per-library JSON metadata can supplement structure fields absent from an
MGF.  The join is fail-closed: SpectrumID/USI keys must be unique on both sides,
and conflicting MGF/JSON chemistry invalidates the record.  A source without a
verifiable IK14 remains in the inventory but cannot enter an identity panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from prepare_enveda180_scoreblind_manifest import (
    load_csv,
    load_exclusion_registry,
    load_hdf5,
    load_mgf,
    secondary_spectrum_hash,
    sha256_file,
    spectrum_hash,
)


RDLogger.DisableLog("rdApp.*")

SOURCE_SCHEMA = "gnps2_standard_mega_source_manifest_v1"
REPORT_SCHEMA = "gnps2_standard_mega_preflight_v1"
DEFAULT_PPM = 10.0
DEFAULT_MIN_PEAKS = 10
TEN_MILLION = 10_000_000
DEFAULT_PAIR_HARD_PPM = 10.0
DEFAULT_PAIR_HARD_ABS_DA = 0.05
DEFAULT_PAIR_OUTER_ABS_DA = 2.0
DEFAULT_PAIR_POSITIVE_CAP = 8
DEFAULT_PAIR_NEGATIVE_CAP = 512

PROTON = 1.007276466621
SODIUM = 22.989218
POTASSIUM = 38.963158
AMMONIUM = 18.033823
WATER = 18.010565
FORMATE_MINUS_H = 44.998201
CHLORIDE = 34.969402

IK_FULL = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
IK14 = re.compile(r"^[A-Z]{14}$")
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
BRACKETED_ADDUCT = re.compile(r"\[(M[^\]]*)\](\d*)([+-])", re.IGNORECASE)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    mgf: Path
    slim_metadata: Path | None
    required: bool


@dataclass(frozen=True)
class Chemistry:
    inchikey: str
    ik14: str
    formula: str
    exact_mass: float
    canonical_smiles: str
    basis: str


def classify_identity_panel_status(counters: Counter) -> tuple[str, float]:
    """Classify source-level identity usability without hiding sparse labels.

    A handful of labelled records in an otherwise annotation-only library is
    useful inventory information, but it does not make the *source* a viable
    identity benchmark. Keep those rows eligible for a cross-source panel
    while describing the source itself conservatively.
    """
    raw = int(counters["raw_records"])
    eligible = int(counters["eligible_before_overlap"])
    fraction = float(eligible / raw) if raw else 0.0
    if eligible == 0 and raw > 0 and int(counters["no_verifiable_identity"]) > 0:
        return "annotation_only_no_verifiable_ik14", fraction
    if raw > 0 and eligible > 0 and fraction < 0.01:
        return "annotation_only_sparse_verifiable_ik14", fraction
    if eligible > 0:
        return "eligible_identity_records_present", fraction
    return "no_eligible_identity_records", fraction


def _clean(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "n/a", "na", "none", "null", "nan"} else text


def _lookup(fields: dict[str, object], *names: str) -> str:
    folded = {str(key).casefold(): value for key, value in fields.items()}
    for name in names:
        value = _clean(folded.get(name.casefold()))
        if value:
            return value
    return ""


def strict_iter_mgf(path: Path) -> Iterator[tuple[int, dict[str, str], list[tuple[float, float]]]]:
    """Yield complete MGF records and reject truncation/nesting."""
    fields: dict[str, str] | None = None
    peaks: list[tuple[float, float]] = []
    source_row = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line == "BEGIN IONS":
                if fields is not None:
                    raise RuntimeError(f"nested BEGIN IONS at {path}:{line_number}")
                fields, peaks = {}, []
            elif line == "END IONS":
                if fields is None:
                    raise RuntimeError(f"END IONS without BEGIN at {path}:{line_number}")
                yield source_row, fields, peaks
                source_row += 1
                fields = None
                peaks = []
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()
            elif fields is not None and line:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0:
                    peaks.append((mz, intensity))
    if fields is not None:
        raise RuntimeError(f"truncated MGF record at end of {path}")


def _json_records(path: Path) -> list[dict[str, object]]:
    """Load a GNPS2 per-library JSON or an intentionally slim JSON/JSONL map."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        result = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise RuntimeError(f"slim JSONL row {line_number} is not an object")
                result.append(value)
        return result
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("spectra"), list):
        records = payload["spectra"]
    else:
        raise RuntimeError("slim metadata must be a JSON list, JSONL, or {'spectra': [...]} object")
    if any(not isinstance(record, dict) for record in records):
        raise RuntimeError("slim metadata contains a non-object record")
    return list(records)


def load_slim_metadata(path: Path) -> tuple[dict[str, dict[str, object]], dict]:
    """Index official metadata by unique SpectrumID and USI."""
    records = _json_records(path)
    index: dict[str, dict[str, object]] = {}
    for row_number, record in enumerate(records):
        keys = {
            _lookup(record, "SpectrumID", "spectrum_id", "SPECTRUMID"),
            _lookup(record, "USI", "usi"),
        } - {""}
        if not keys:
            raise RuntimeError(f"slim metadata row {row_number} lacks SpectrumID and USI")
        for key in keys:
            if key in index and index[key] is not record:
                raise RuntimeError(f"slim metadata key is not one-to-one: {key}")
            index[key] = record
    return index, {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "unique_join_keys": len(index),
    }


def join_slim(fields: dict[str, str], index: dict[str, dict[str, object]]) -> dict[str, object] | None:
    keys = {
        _lookup(fields, "SPECTRUMID", "TITLE"),
        _lookup(fields, "USI"),
    } - {""}
    matches = {id(index[key]): index[key] for key in keys if key in index}
    if len(matches) > 1:
        raise RuntimeError(f"MGF SpectrumID/USI resolve to different slim records: {sorted(keys)}")
    return next(iter(matches.values())) if matches else None


def formula_exact_mass(formula: str) -> float | None:
    formula = formula.strip()
    parts = list(FORMULA_TOKEN.finditer(formula))
    if not parts or "".join(match.group(0) for match in parts) != formula:
        return None
    table = Chem.GetPeriodicTable()
    total = 0.0
    for match in parts:
        symbol = match.group(1)
        count = int(match.group(2) or "1")
        atomic_number = table.GetAtomicNumber(symbol)
        if atomic_number <= 0 or count <= 0:
            return None
        total += table.GetMostCommonIsotopeMass(atomic_number) * count
    return total if math.isfinite(total) and total > 0 else None


def _raw_ik(fields: dict[str, object]) -> str:
    candidates = [
        _lookup(fields, "INCHIKEY", "INCHIAUX", "INCHI_AUX"),
        _lookup(fields, "InChIKey_smiles"),
        _lookup(fields, "InChIKey_inchi"),
    ]
    valid = {value.upper() for value in candidates if IK_FULL.fullmatch(value.upper())}
    if len(valid) > 1 and len({value[:14] for value in valid}) > 1:
        raise ValueError("conflicting official InChIKeys")
    return sorted(valid)[0] if valid else ""


@lru_cache(maxsize=500_000)
def _chemistry_from_text(
    smiles: str,
    inchi: str,
    raw_ik: str,
    annotated_formula: str,
    basis: str,
) -> Chemistry | None:
    """Parse repeated library annotations once (replicate spectra are common)."""
    mol_smiles = Chem.MolFromSmiles(smiles) if smiles else None
    mol_inchi = Chem.MolFromInchi(inchi.strip('"')) if inchi.startswith("InChI=") else None
    if mol_smiles is not None and mol_inchi is not None:
        if Chem.MolToInchiKey(mol_smiles)[:14] != Chem.MolToInchiKey(mol_inchi)[:14]:
            raise ValueError("SMILES/InChI identity conflict")
    mol = mol_smiles if mol_smiles is not None else mol_inchi
    if mol is not None:
        computed_key = Chem.MolToInchiKey(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        if raw_ik and raw_ik[:14] != computed_key[:14]:
            raise ValueError("official/computed InChIKey conflict")
        if annotated_formula and annotated_formula != formula:
            raise ValueError("official/computed formula conflict")
        return Chemistry(
            inchikey=computed_key,
            ik14=computed_key[:14],
            formula=formula,
            exact_mass=float(Descriptors.ExactMolWt(mol)),
            canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
            basis=f"{basis}:structure",
        )
    formula = annotated_formula
    exact_mass = formula_exact_mass(formula) if formula else None
    if raw_ik and exact_mass is not None:
        return Chemistry(
            inchikey=raw_ik,
            ik14=raw_ik[:14],
            formula=formula,
            exact_mass=exact_mass,
            canonical_smiles="",
            basis=f"{basis}:official_key_formula",
        )
    return None


def chemistry_from_fields(fields: dict[str, object], basis: str) -> Chemistry | None:
    smiles = _lookup(fields, "SMILES", "Smiles")
    inchi = _lookup(fields, "INCHI", "InChI")
    raw_ik = _raw_ik(fields)
    annotated_formula = _lookup(fields, "FORMULA", "Formula_smiles", "Formula_inchi")
    return _chemistry_from_text(smiles, inchi, raw_ik, annotated_formula, basis)


def merge_chemistry(fields: dict[str, str], slim: dict[str, object] | None) -> Chemistry | None:
    mgf_chem = chemistry_from_fields(fields, "mgf")
    slim_chem = chemistry_from_fields(slim, "official_slim_json") if slim is not None else None
    if mgf_chem is not None and slim_chem is not None:
        if mgf_chem.ik14 != slim_chem.ik14 or mgf_chem.formula != slim_chem.formula:
            raise ValueError("MGF/slim chemistry conflict")
        return mgf_chem
    return mgf_chem if mgf_chem is not None else slim_chem


def normalize_polarity(value: str) -> str | None:
    lowered = value.strip().lower()
    if "pos" in lowered or lowered == "+":
        return "positive"
    if "neg" in lowered or lowered == "-":
        return "negative"
    return None


def normalize_adduct(raw: str, name: str) -> str | None:
    text = _clean(raw)
    candidates = BRACKETED_ADDUCT.findall(f"{text} {name}")
    if candidates:
        body, charge_count, sign = candidates[0]
        body = body.upper().replace(" ", "")
        suffix = f"{charge_count}{sign}" if charge_count else sign
        token = f"[{body}]{suffix}"
    else:
        compact = text.upper().replace(" ", "").strip("[]")
        aliases = {
            "M": "[M]+",
            "M+H": "[M+H]+", "M-H": "[M-H]-", "M+NA": "[M+NA]+",
            "M+K": "[M+K]+", "M+NH4": "[M+NH4]+", "M+CL": "[M+CL]-",
            "M+FA-H": "[M+FA-H]-", "M+H-H2O": "[M+H-H2O]+",
            "M+2H": "[M+2H]2+", "M-2H": "[M-2H]2-",
        }
        token = aliases.get(compact, text.upper().replace(" ", ""))
    aliases = {
        "[M]+": "[M]+", "[M]-": "[M]-", "[M]2+": "[M]2+", "[M]2-": "[M]2-",
        "[M+H]+": "[M+H]+", "[M-H]-": "[M-H]-", "[M+NA]+": "[M+Na]+",
        "[M+K]+": "[M+K]+", "[M+NH4]+": "[M+NH4]+", "[M+CL]-": "[M+Cl]-",
        "[M+FA-H]-": "[M+FA-H]-", "[M+H-H2O]+": "[M+H-H2O]+",
        "[M+2H]2+": "[M+2H]2+", "[M-2H]2-": "[M-2H]2-",
    }
    return aliases.get(token.upper())


def expected_precursor(exact_mass: float, adduct: str) -> tuple[float, str]:
    rules = {
        # Permanently charged structures are represented explicitly in GNPS2
        # as [M]+/[M]- rather than protonation/deprotonation.
        "[M]+": (exact_mass, "positive"),
        "[M]-": (exact_mass, "negative"),
        "[M]2+": (exact_mass / 2, "positive"),
        "[M]2-": (exact_mass / 2, "negative"),
        "[M+H]+": (exact_mass + PROTON, "positive"),
        "[M-H]-": (exact_mass - PROTON, "negative"),
        "[M+Na]+": (exact_mass + SODIUM, "positive"),
        "[M+K]+": (exact_mass + POTASSIUM, "positive"),
        "[M+NH4]+": (exact_mass + AMMONIUM, "positive"),
        "[M+Cl]-": (exact_mass + CHLORIDE, "negative"),
        "[M+FA-H]-": (exact_mass + FORMATE_MINUS_H, "negative"),
        "[M+H-H2O]+": (exact_mass + PROTON - WATER, "positive"),
        "[M+2H]2+": ((exact_mass + 2 * PROTON) / 2, "positive"),
        "[M-2H]2-": ((exact_mass - 2 * PROTON) / 2, "negative"),
    }
    if adduct not in rules:
        raise KeyError(adduct)
    return rules[adduct]


def _precursor(fields: dict[str, str], slim: dict[str, object] | None) -> float | None:
    value = _lookup(fields, "PEPMASS", "PRECURSOR_MZ")
    if not value and slim is not None:
        value = _lookup(slim, "Precursor_MZ", "precursor_mz")
    try:
        result = float(value.split()[0])
    except (ValueError, IndexError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _quality(fields: dict[str, str], slim: dict[str, object] | None) -> int | None:
    value = _lookup(fields, "LIBRARYQUALITY")
    if not value and slim is not None:
        value = _lookup(slim, "Library_Class", "library_class")
    try:
        quality = int(float(value))
    except ValueError:
        return None
    return quality if quality in {1, 2, 3} else None


def _record_token(source_id: str, fields: dict[str, str], source_row: int) -> str:
    spectrum_id = _lookup(fields, "SPECTRUMID", "TITLE")
    usi = _lookup(fields, "USI")
    return f"{source_id}|{spectrum_id}|{usi}|{source_row}"


def scan_source(
    source: SourceSpec,
    *,
    min_peaks: int,
    precursor_ppm: float,
    precursor_abs_da: float,
) -> tuple[list[dict], dict]:
    path = source.mgf
    if not path.is_file() or path.stat().st_size <= 0:
        return [], {
            "source_id": source.source_id,
            "path": str(path),
            "required": source.required,
            "status": "NOT_READY_EMPTY_OR_MISSING",
            "bytes": path.stat().st_size if path.exists() else None,
        }
    before = path.stat()
    slim_index: dict[str, dict[str, object]] = {}
    slim_report = None
    if source.slim_metadata is not None:
        if not source.slim_metadata.is_file():
            return [], {
                "source_id": source.source_id,
                "path": str(path),
                "required": source.required,
                "status": "NOT_READY_SLIM_METADATA_MISSING",
                "slim_metadata": str(source.slim_metadata),
            }
        slim_index, slim_report = load_slim_metadata(source.slim_metadata)

    counters: Counter[str] = Counter()
    rows: list[dict] = []
    seen_join_keys: set[str] = set()
    used_slim_records: set[int] = set()
    for source_row, (mgf_row, fields, peaks) in enumerate(strict_iter_mgf(path)):
        assert source_row == mgf_row
        counters["raw_records"] += 1
        spectrum_id = _lookup(fields, "SPECTRUMID", "TITLE")
        if spectrum_id:
            if spectrum_id in seen_join_keys:
                raise RuntimeError(f"duplicate MGF SpectrumID in {source.source_id}: {spectrum_id}")
            seen_join_keys.add(spectrum_id)
        slim = join_slim(fields, slim_index) if slim_index else None
        if slim is not None:
            used_slim_records.add(id(slim))
            counters["slim_metadata_joined"] += 1
        try:
            quality = _quality(fields, slim)
            if quality is None:
                counters["invalid_library_quality"] += 1
                continue
            counters[f"library_quality_{quality}"] += 1
            ms_level = _lookup(fields, "MSLEVEL") or (_lookup(slim, "ms_level") if slim else "")
            if ms_level and ms_level not in {"2", "2.0"}:
                counters["not_ms2"] += 1
                continue
            if len(peaks) < min_peaks:
                counters["too_few_peaks"] += 1
                continue
            chemistry = merge_chemistry(fields, slim)
            if chemistry is None:
                counters["no_verifiable_identity"] += 1
                continue
            counters[f"identity_basis_{chemistry.basis}"] += 1
            polarity = normalize_polarity(
                _lookup(fields, "IONMODE") or (_lookup(slim, "Ion_Mode", "ion_mode") if slim else "")
            )
            if polarity is None:
                counters["invalid_polarity"] += 1
                continue
            adduct = normalize_adduct(
                _lookup(fields, "ADDUCT") or (_lookup(slim, "Adduct", "adduct") if slim else ""),
                _lookup(fields, "NAME") or (_lookup(slim, "Compound_Name") if slim else ""),
            )
            if adduct is None:
                counters["unsupported_or_missing_adduct"] += 1
                continue
            precursor_mz = _precursor(fields, slim)
            if precursor_mz is None:
                counters["invalid_precursor"] += 1
                continue
            expected_mz, expected_polarity = expected_precursor(chemistry.exact_mass, adduct)
            if polarity != expected_polarity:
                counters["polarity_adduct_conflict"] += 1
                continue
            error_da = abs(precursor_mz - expected_mz)
            error_ppm = error_da / expected_mz * 1e6
            if error_da > max(precursor_abs_da, expected_mz * precursor_ppm * 1e-6):
                counters["precursor_structure_mismatch"] += 1
                continue
            primary_hash = spectrum_hash(precursor_mz, peaks)
            secondary_hash = secondary_spectrum_hash(peaks)
        except ValueError as error:
            counters[f"chemistry_conflict:{error}"] += 1
            continue

        rows.append({
            "source_id": source.source_id,
            "source_row": source_row,
            "record_token": _record_token(source.source_id, fields, source_row),
            "spectrum_id": spectrum_id,
            "usi": _lookup(fields, "USI"),
            "library_quality": quality,
            "quality_tier": "Q12_PRIMARY" if quality in {1, 2} else "Q3_SEPARATE",
            "inchikey": chemistry.inchikey,
            "ik14": chemistry.ik14,
            "formula": chemistry.formula,
            "identity_basis": chemistry.basis,
            "precursor_mz": precursor_mz,
            "precursor_structure_ppm": (precursor_mz - expected_mz) / expected_mz * 1e6,
            "polarity": polarity,
            "adduct": adduct,
            "n_peaks": len(peaks),
            "spectrum_hash": primary_hash,
            "secondary_spectrum_hash": secondary_hash,
        })
        counters["eligible_before_overlap"] += 1

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while being scanned: {path}")
    digest = sha256_file(path)
    final = path.stat()
    if (after.st_size, after.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise RuntimeError(f"source changed while being hashed: {path}")
    identity_panel_status, identity_eligible_fraction = classify_identity_panel_status(counters)
    report = {
        "source_id": source.source_id,
        "path": str(path),
        "required": source.required,
        "status": "SCANNED",
        "bytes": final.st_size,
        "sha256": digest,
        "slim_metadata": slim_report,
        "slim_records_joined": len(used_slim_records),
        "identity_panel_status": identity_panel_status,
        "identity_eligible_fraction": identity_eligible_fraction,
        "counters": dict(sorted(counters.items())),
    }
    return rows, report


def load_consumed_registry(path: Path) -> tuple[set[str], set[str], dict[str, set[str]], list[dict], list[str]]:
    identities: set[str] = set()
    formulas: set[str] = set()
    hashes = {"primary_sha256": set(), "secondary_blake2b": set()}
    reports = []
    missing_required = []
    loaders = {"hdf5": load_hdf5, "csv": load_csv, "mgf": load_mgf}
    for source in load_exclusion_registry(path):
        name = str(source["name"])
        kind = str(source.get("kind", ""))
        source_path = Path(str(source.get("path", "")))
        required = bool(source.get("required", True))
        if kind not in loaders:
            raise RuntimeError(f"unsupported consumed source kind: {name} -> {kind}")
        if required and not source_path.is_file():
            missing_required.append(name)
            reports.append({
                "name": name, "kind": kind, "path": str(source_path),
                "required": required, "status": "missing",
            })
            continue
        loaded = loaders[kind](source_path, identities, formulas, hashes)
        loaded.update({"name": name, "kind": kind, "required": required})
        reports.append(loaded)
        if required and loaded.get("status") != "loaded":
            missing_required.append(name)
    return identities, formulas, hashes, reports, sorted(set(missing_required))


def clean_records(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    if not rows:
        return pd.DataFrame(), {}
    frame = pd.DataFrame(rows)
    counters: Counter[str] = Counter()
    # Any exact peak signature assigned to multiple identities is ambiguous.
    conflicting_primary = set(
        frame.groupby("spectrum_hash").ik14.nunique().loc[lambda values: values > 1].index
    )
    conflicting_secondary = set(
        frame.groupby("secondary_spectrum_hash").ik14.nunique().loc[lambda values: values > 1].index
    )
    conflict = frame.spectrum_hash.isin(conflicting_primary) | frame.secondary_spectrum_hash.isin(conflicting_secondary)
    counters["cross_identity_hash_conflict_records"] = int(conflict.sum())
    frame = frame.loc[~conflict].copy()
    before = len(frame)
    frame = (
        frame.sort_values(["ik14", "spectrum_hash", "source_id", "source_row"], kind="stable")
        .drop_duplicates(["ik14", "spectrum_hash"], keep="first")
    )
    counters["same_identity_exact_duplicates_removed"] = before - len(frame)
    formula_count = frame.groupby("ik14").formula.nunique()
    conflicting_identities = set(formula_count[formula_count > 1].index)
    conflict_formula = frame.ik14.isin(conflicting_identities)
    counters["identity_formula_conflict_records"] = int(conflict_formula.sum())
    frame = frame.loc[~conflict_formula].copy()
    frame = frame.sort_values(["source_id", "source_row"], kind="stable").reset_index(drop=True)
    frame.insert(0, "global_row", np.arange(len(frame), dtype=np.int64))
    return frame, dict(counters)


def apply_overlap(
    frame: pd.DataFrame,
    identities: set[str],
    formulas: set[str],
    hashes: dict[str, set[str]],
) -> pd.DataFrame:
    result = frame.copy()
    result["consumed_identity_overlap"] = result.ik14.isin(identities)
    result["consumed_formula_overlap"] = result.formula.isin(formulas)
    result["consumed_spectrum_overlap"] = (
        result.spectrum_hash.isin(hashes["primary_sha256"])
        | result.secondary_spectrum_hash.isin(hashes["secondary_blake2b"])
    )
    return result


def _split_query_reference(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    query_indices: list[int] = []
    reference_indices: list[int] = []
    cross_source_groups = 0
    within_source_groups = 0
    for _, group in frame.groupby(["polarity", "adduct", "ik14"], sort=True):
        if len(group) < 2:
            continue
        sources = sorted(group.source_id.unique())
        if len(sources) >= 2:
            # Deterministically reserve one complete source for references and
            # use the others as queries.  This is the stronger split whenever
            # independent source coverage exists.
            source_order = sorted(
                sources,
                key=lambda value: hashlib.sha256(f"{seed}|source|{value}".encode()).hexdigest(),
            )
            ref_source = source_order[0]
            reference_indices.extend(group.index[group.source_id == ref_source].tolist())
            query_indices.extend(group.index[group.source_id != ref_source].tolist())
            cross_source_groups += 1
        else:
            ordered = sorted(
                group.index.tolist(),
                key=lambda index: hashlib.sha256(
                    f"{seed}|record|{frame.at[index, 'record_token']}".encode()
                ).hexdigest(),
            )
            n_reference = max(1, len(ordered) // 2)
            reference_indices.extend(ordered[:n_reference])
            query_indices.extend(ordered[n_reference:])
            within_source_groups += 1
    return (
        frame.loc[query_indices].copy().sort_values(["polarity", "adduct", "precursor_mz", "record_token"]),
        frame.loc[reference_indices].copy().sort_values(["polarity", "adduct", "precursor_mz", "record_token"]),
        {
            "identity_adduct_groups_cross_source_split": cross_source_groups,
            "identity_adduct_groups_within_source_split": within_source_groups,
        },
    )


def forecast_panel(frame: pd.DataFrame, *, ppm: float, seed: int) -> dict:
    queries, references, split_report = _split_query_reference(frame, seed)
    n_q = n_e = n_p = near = 0
    query_identities: set[str] = set()
    query_formulas: set[str] = set()
    by_domain: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray]] = {}
    for domain, group in references.groupby(["polarity", "adduct"], sort=False):
        ordered = group.sort_values("precursor_mz", kind="stable")
        by_domain[domain] = (ordered, ordered.precursor_mz.to_numpy(float))
    for query in queries.itertuples(index=False):
        pool = by_domain.get((query.polarity, query.adduct))
        if pool is None:
            continue
        reference_frame, masses = pool
        delta = abs(float(query.precursor_mz)) * ppm * 1e-6
        left = int(np.searchsorted(masses, float(query.precursor_mz) - delta, side="left"))
        right = int(np.searchsorted(masses, float(query.precursor_mz) + delta, side="right"))
        window = reference_frame.iloc[left:right]
        if window.empty or query.ik14 not in set(window.ik14):
            continue
        candidate_groups = list(window.groupby("ik14", sort=True))
        if len(candidate_groups) < 2:
            continue
        n_q += 1
        n_e += len(candidate_groups)
        n_p += len(window)
        query_identities.add(str(query.ik14))
        query_formulas.add(str(query.formula))
        if any(identity != query.ik14 and group.formula.iloc[0] == query.formula for identity, group in candidate_groups):
            near += 1
    return {
        "N_Q_labelled_queries": n_q,
        "N_I_query_identity_clusters": len(query_identities),
        "N_F_query_formula_clusters": len(query_formulas),
        "N_E_query_candidate_molecule_edges": n_e,
        "N_P_query_reference_spectrum_comparisons": n_p,
        "near_queries": near,
        "reaches_10M_comparisons": n_p >= TEN_MILLION,
        "unit_warning": "N_P is spectrum comparisons, not labelled queries",
        **split_report,
    }


def _mix_uint64(values: np.ndarray) -> np.ndarray:
    """Stable SplitMix64 finalizer used only for deterministic pair sampling."""
    mixed = np.asarray(values, dtype=np.uint64).copy()
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


def _deterministic_take(
    positions: np.ndarray,
    reference_global_rows: np.ndarray,
    *,
    cap: int,
    seed: int,
    query_global_row: int,
    salt: int,
) -> np.ndarray:
    if cap <= 0 or len(positions) == 0:
        return np.empty(0, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    global_rows = reference_global_rows[positions].astype(np.uint64, copy=False)
    query_key = (
        int(query_global_row) * 0x9E3779B97F4A7C15
        + int(seed)
        + int(salt) * 0xD1B54A32D192ED03
    ) & ((1 << 64) - 1)
    keys = _mix_uint64(global_rows ^ np.uint64(query_key))
    order = np.lexsort((global_rows, keys))
    return positions[order[:cap]]


def forecast_pairwise_verification(
    frame: pd.DataFrame,
    *,
    seed: int,
    hard_ppm: float = DEFAULT_PAIR_HARD_PPM,
    hard_abs_da: float = DEFAULT_PAIR_HARD_ABS_DA,
    outer_abs_da: float = DEFAULT_PAIR_OUTER_ABS_DA,
    positive_cap: int = DEFAULT_PAIR_POSITIVE_CAP,
    negative_cap: int = DEFAULT_PAIR_NEGATIVE_CAP,
    return_indices: bool = False,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    """Forecast a separate labelled pairwise identity-verification axis.

    This does not alter the 10-ppm candidate-ranking panels. Positives are
    independent query/reference spectra with the same IK14. Negatives always
    share polarity and normalized adduct and are selected in three transparent
    precursor bands: 10 ppm first, then 0.05 Da, then the outer 2 Da window.
    Pair caps bound identity dominance and keep materialization practical.
    """
    if hard_ppm <= 0 or hard_abs_da <= 0 or outer_abs_da < hard_abs_da:
        raise ValueError("pairwise mass windows must satisfy 0 < ppm and 0 < hard_da <= outer_da")
    if positive_cap <= 0 or negative_cap <= 0:
        raise ValueError("pairwise per-query caps must be positive")

    queries, references, split_report = _split_query_reference(frame, seed)
    pools: dict[tuple[str, str], dict[str, object]] = {}
    for domain, group in references.groupby(["polarity", "adduct"], sort=False):
        ordered = group.sort_values(["precursor_mz", "global_row"], kind="stable")
        identities = ordered.ik14.astype(str).to_numpy()
        positive_positions: dict[str, np.ndarray] = {}
        for identity in np.unique(identities):
            positive_positions[str(identity)] = np.flatnonzero(identities == identity).astype(np.int64)
        pools[domain] = {
            "mass": ordered.precursor_mz.to_numpy(float),
            "ik14": identities,
            "formula": ordered.formula.astype(str).to_numpy(),
            "hash": ordered.spectrum_hash.astype(str).to_numpy(),
            "global_row": ordered.global_row.to_numpy(np.int64),
            "positive_positions": positive_positions,
        }

    query_rows_parts: list[np.ndarray] = []
    reference_rows_parts: list[np.ndarray] = []
    pair_type_parts: list[np.ndarray] = []
    n_q = n_positive = n_negative = 0
    negative_bands = Counter()
    query_identities: set[str] = set()
    query_formulas: set[str] = set()
    max_observed_pairs = 0

    for query in queries.itertuples(index=False):
        pool = pools.get((query.polarity, query.adduct))
        if pool is None:
            continue
        masses = pool["mass"]
        identities = pool["ik14"]
        spectrum_hashes = pool["hash"]
        global_rows = pool["global_row"]
        positive_positions = pool["positive_positions"].get(str(query.ik14), np.empty(0, dtype=np.int64))
        if len(positive_positions):
            positive_positions = positive_positions[
                spectrum_hashes[positive_positions] != str(query.spectrum_hash)
            ]
        positives = _deterministic_take(
            positive_positions,
            global_rows,
            cap=positive_cap,
            seed=seed,
            query_global_row=int(query.global_row),
            salt=1,
        )
        if len(positives) == 0:
            continue

        query_mass = float(query.precursor_mz)
        left = int(np.searchsorted(masses, query_mass - outer_abs_da, side="left"))
        right = int(np.searchsorted(masses, query_mass + outer_abs_da, side="right"))
        window_positions = np.arange(left, right, dtype=np.int64)
        negative_positions = window_positions[identities[window_positions] != str(query.ik14)]
        if len(negative_positions) == 0:
            continue
        differences = np.abs(masses[negative_positions] - query_mass)
        ppm_da = abs(query_mass) * hard_ppm * 1e-6
        band_masks = (
            differences <= ppm_da,
            (differences > ppm_da) & (differences <= hard_abs_da),
            (differences > hard_abs_da) & (differences <= outer_abs_da),
        )
        selected_negative_parts = []
        selected_type_parts = []
        remaining = negative_cap
        for band_index, mask in enumerate(band_masks):
            if remaining <= 0:
                break
            selected = _deterministic_take(
                negative_positions[mask],
                global_rows,
                cap=remaining,
                seed=seed,
                query_global_row=int(query.global_row),
                salt=10 + band_index,
            )
            if len(selected):
                selected_negative_parts.append(selected)
                selected_type_parts.append(np.full(len(selected), 2 + band_index, dtype=np.uint8))
                negative_bands[
                    {
                        0: "within_hard_ppm",
                        1: "outside_hard_ppm_within_hard_abs_da",
                        2: "outside_hard_abs_da_within_outer_abs_da",
                    }[band_index]
                ] += len(selected)
                remaining -= len(selected)
        if not selected_negative_parts:
            continue
        negatives = np.concatenate(selected_negative_parts)
        negative_types = np.concatenate(selected_type_parts)
        selected_references = np.concatenate([positives, negatives])
        selected_types = np.concatenate([
            np.ones(len(positives), dtype=np.uint8),
            negative_types,
        ])
        n_q += 1
        n_positive += len(positives)
        n_negative += len(negatives)
        max_observed_pairs = max(max_observed_pairs, len(selected_references))
        query_identities.add(str(query.ik14))
        query_formulas.add(str(query.formula))
        if return_indices:
            query_rows_parts.append(np.full(len(selected_references), int(query.global_row), dtype=np.int32))
            reference_rows_parts.append(global_rows[selected_references].astype(np.int32, copy=False))
            pair_type_parts.append(selected_types)

    n_pairs = n_positive + n_negative
    report = {
        "N_Q_labelled_queries": n_q,
        "N_I_query_identity_clusters": len(query_identities),
        "N_F_query_formula_clusters": len(query_formulas),
        "N_pairs_positive": n_positive,
        "N_pairs_negative": n_negative,
        "N_pairs_total": n_pairs,
        "negative_pair_bands": dict(sorted(negative_bands.items())),
        "reaches_10M_pairs": n_pairs >= TEN_MILLION,
        "positive_pairs_per_query_cap": positive_cap,
        "negative_pairs_per_query_cap": negative_cap,
        "maximum_observed_pairs_per_query": max_observed_pairs,
        "hard_negative_ppm": hard_ppm,
        "hard_negative_abs_da": hard_abs_da,
        "outer_negative_abs_da": outer_abs_da,
        "sampling_seed": seed,
        "duplicate_control": {
            "query_reference_pairs_unique": True,
            "query_and_reference_axes_disjoint": True,
            "positive_exact_spectrum_hashes_distinct": True,
            "same_reference_may_pair_with_different_queries": True,
        },
        "unit_warning": "N_pairs is labelled query-reference pairs, not labelled queries and not Top-k candidate comparisons",
        **split_report,
    }
    arrays = None
    if return_indices:
        arrays = {
            "query_row": np.concatenate(query_rows_parts) if query_rows_parts else np.empty(0, dtype=np.int32),
            "reference_row": np.concatenate(reference_rows_parts) if reference_rows_parts else np.empty(0, dtype=np.int32),
            "pair_type": np.concatenate(pair_type_parts) if pair_type_parts else np.empty(0, dtype=np.uint8),
        }
        if not (len(arrays["query_row"]) == len(arrays["reference_row"]) == len(arrays["pair_type"]) == n_pairs):
            raise RuntimeError("pairwise materialization count drift")
    return report, arrays


def load_source_manifest(path: Path) -> tuple[dict, list[SourceSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError(f"unexpected source manifest schema: {path}")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError("source manifest needs a non-empty sources list")
    specs = []
    seen = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise RuntimeError(f"sources[{index}] must be an object")
        source_id = str(raw.get("source_id", "")).strip()
        if not source_id or source_id in seen:
            raise RuntimeError(f"source_id is empty or duplicated: {source_id!r}")
        seen.add(source_id)
        mgf = Path(str(raw.get("mgf", "")))
        if not mgf.is_absolute():
            mgf = (path.parent / mgf).resolve()
        slim_raw = raw.get("slim_metadata")
        slim = None
        if slim_raw:
            slim = Path(str(slim_raw))
            if not slim.is_absolute():
                slim = (path.parent / slim).resolve()
        specs.append(SourceSpec(source_id, mgf, slim, bool(raw.get("required", True))))
    return payload, specs


def preflight(
    source_manifest: Path,
    exclusion_registry: Path,
    *,
    min_peaks: int = DEFAULT_MIN_PEAKS,
    precursor_ppm: float = 20.0,
    precursor_abs_da: float = 0.01,
    candidate_ppm: float = DEFAULT_PPM,
    pair_hard_ppm: float = DEFAULT_PAIR_HARD_PPM,
    pair_hard_abs_da: float = DEFAULT_PAIR_HARD_ABS_DA,
    pair_outer_abs_da: float = DEFAULT_PAIR_OUTER_ABS_DA,
    pair_positive_cap: int = DEFAULT_PAIR_POSITIVE_CAP,
    pair_negative_cap: int = DEFAULT_PAIR_NEGATIVE_CAP,
    seed: int = 20261102,
) -> tuple[dict, pd.DataFrame]:
    manifest, sources = load_source_manifest(source_manifest)
    all_rows: list[dict] = []
    source_reports = []
    for source in sources:
        rows, report = scan_source(
            source,
            min_peaks=min_peaks,
            precursor_ppm=precursor_ppm,
            precursor_abs_da=precursor_abs_da,
        )
        all_rows.extend(rows)
        source_reports.append(report)
    frame, cleaning = clean_records(all_rows)
    identities, formulas, hashes, exclusion_reports, missing_exclusions = load_consumed_registry(exclusion_registry)
    source_not_ready = [
        row["source_id"] for row in source_reports
        if row["required"] and row["status"] != "SCANNED"
    ]
    exclusion_complete = not missing_exclusions
    # Always mark overlap against every source that is locally available. A
    # missing required source still blocks panel release; it must not prevent
    # us from recording already-known contamination in the inventory.
    if not frame.empty:
        frame = apply_overlap(frame, identities, formulas, hashes)

    def forecasts(base: pd.DataFrame) -> dict[str, dict]:
        result = {}
        tiers = {
            "q12_primary": base.library_quality.isin([1, 2]),
            "q3_separate": base.library_quality.eq(3),
            "q123_expanded": base.library_quality.isin([1, 2, 3]),
        }
        for tier, quality_mask in tiers.items():
            selected = base[quality_mask].copy()
            result[tier] = {
                "eligible_records": len(selected),
                **forecast_panel(selected, ppm=candidate_ppm, seed=seed),
            }
        return result

    def pairwise_forecasts(base: pd.DataFrame) -> dict[str, dict]:
        result = {}
        tiers = {
            "q12_primary": base.library_quality.isin([1, 2]),
            "q3_separate": base.library_quality.eq(3),
            "q123_expanded": base.library_quality.isin([1, 2, 3]),
        }
        for tier, quality_mask in tiers.items():
            selected = base[quality_mask].copy()
            pair_report, _ = forecast_pairwise_verification(
                selected,
                seed=seed,
                hard_ppm=pair_hard_ppm,
                hard_abs_da=pair_hard_abs_da,
                outer_abs_da=pair_outer_abs_da,
                positive_cap=pair_positive_cap,
                negative_cap=pair_negative_cap,
            )
            result[tier] = {"eligible_records": len(selected), **pair_report}
        return result

    # This is a score-blind capacity forecast, not an external benchmark. It is
    # emitted even when consumed exclusions are unavailable so that users can
    # decide whether finishing the exclusion assets is computationally useful.
    pre_exclusion_forecasts = forecasts(frame) if not frame.empty else {}
    pre_exclusion_pairwise_forecasts = pairwise_forecasts(frame) if not frame.empty else {}
    available_exclusion_forecasts = {}
    available_exclusion_pairwise_forecasts = {}
    if not frame.empty:
        tiers = {
            "q12_primary": frame.library_quality.isin([1, 2]),
            "q3_separate": frame.library_quality.eq(3),
            "q123_expanded": frame.library_quality.isin([1, 2, 3]),
        }
        for tier, quality_mask in tiers.items():
            for disjointness, overlap_column in (
                ("identity_disjoint", "consumed_identity_overlap"),
                ("formula_disjoint", "consumed_formula_overlap"),
            ):
                selected = frame[
                    quality_mask
                    & ~frame[overlap_column]
                    & ~frame.consumed_spectrum_overlap
                ].copy()
                available_exclusion_forecasts[f"{tier}.{disjointness}"] = {
                    "eligible_records_after_overlap": len(selected),
                    **forecast_panel(selected, ppm=candidate_ppm, seed=seed),
                }
                pair_report, _ = forecast_pairwise_verification(
                    selected,
                    seed=seed,
                    hard_ppm=pair_hard_ppm,
                    hard_abs_da=pair_hard_abs_da,
                    outer_abs_da=pair_outer_abs_da,
                    positive_cap=pair_positive_cap,
                    negative_cap=pair_negative_cap,
                )
                available_exclusion_pairwise_forecasts[f"{tier}.{disjointness}"] = {
                    "eligible_records_after_overlap": len(selected),
                    **pair_report,
                }

    # The same calculations become releasable panels only after every required
    # consumed source is present. Before then they are incomplete upper bounds.
    panels = available_exclusion_forecasts if exclusion_complete else {}
    pairwise_panels = available_exclusion_pairwise_forecasts if exclusion_complete else {}

    if source_not_ready:
        status = "GNPS2_STANDARD_MEGA_PREFLIGHT_INCOMPLETE_SOURCES"
    elif not exclusion_complete:
        status = "GNPS2_STANDARD_MEGA_PREFLIGHT_BLOCKED_NOT_EXTERNAL"
    elif panels:
        status = "GNPS2_STANDARD_MEGA_PREFLIGHT_READY"
    else:
        status = "GNPS2_STANDARD_MEGA_PREFLIGHT_NO_ELIGIBLE_PANEL"
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "dataset_id": manifest.get("dataset_id"),
        "evaluation_role": "development",
        "truth_status": "metadata inspected; consumed for method development on first scored use",
        "allow_final_claim": False,
        "scores_accessed": False,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "exclusion_registry": str(exclusion_registry),
        "exclusion_registry_sha256": sha256_file(exclusion_registry),
        "parameters": {
            "min_peaks": min_peaks,
            "precursor_validation_ppm": precursor_ppm,
            "precursor_validation_abs_da": precursor_abs_da,
            "candidate_window_ppm": candidate_ppm,
            "pairwise_hard_negative_ppm": pair_hard_ppm,
            "pairwise_hard_negative_abs_da": pair_hard_abs_da,
            "pairwise_outer_negative_abs_da": pair_outer_abs_da,
            "pairwise_positive_pairs_per_query_cap": pair_positive_cap,
            "pairwise_negative_pairs_per_query_cap": pair_negative_cap,
            "split_seed": seed,
        },
        "sources": source_reports,
        "cleaning": cleaning,
        "inventory_counts": {
            "eligible_records_before_overlap": len(frame),
            "unique_IK14_before_overlap": int(frame.ik14.nunique()) if not frame.empty else 0,
            "unique_formulas_before_overlap": int(frame.formula.nunique()) if not frame.empty else 0,
            "quality_strata": (
                {str(key): int(value) for key, value in frame.library_quality.value_counts().sort_index().items()}
                if not frame.empty else {}
            ),
        },
        "overlap_exclusion": {
            "complete": exclusion_complete,
            "missing_required_sources": missing_exclusions,
            "loaded_identity_count": len(identities),
            "loaded_formula_count": len(formulas),
            "loaded_primary_spectrum_hash_count": len(hashes["primary_sha256"]),
            "loaded_secondary_spectrum_hash_count": len(hashes["secondary_blake2b"]),
            "available_overlap_record_counts": {
                "identity": int(frame.consumed_identity_overlap.sum()) if not frame.empty else 0,
                "formula": int(frame.consumed_formula_overlap.sum()) if not frame.empty else 0,
                "exact_spectrum": int(frame.consumed_spectrum_overlap.sum()) if not frame.empty else 0,
            },
            "sources": exclusion_reports,
        },
        "required_sources_not_ready": source_not_ready,
        "score_blind_pre_exclusion_forecasts": pre_exclusion_forecasts,
        "score_blind_pre_exclusion_pairwise_verification_forecasts": pre_exclusion_pairwise_forecasts,
        "pairwise_verification_boundary": (
            "A separate labelled identity-verification task. It does not enlarge or replace the "
            "10-ppm Top-k candidate panels. Negatives share polarity and normalized adduct; the "
            "outer 2-Da band supplies scale and is reported separately from 10-ppm/0.05-Da hard "
            "negatives. Pair count must never be reported as query count or Top-k comparisons."
        ),
        "score_blind_forecast_boundary": (
            "Capacity only: no model score was accessed. Until all required consumed-source "
            "exclusions are loaded, these counts are blocked_not_external and cannot support "
            "an external-performance claim."
        ),
        "score_blind_available_exclusion_forecasts": available_exclusion_forecasts,
        "score_blind_available_exclusion_pairwise_verification_forecasts": available_exclusion_pairwise_forecasts,
        "available_exclusion_forecast_boundary": (
            "These remove every locally available consumed identity/formula/exact-spectrum "
            "overlap, but remain incomplete upper bounds whenever required sources are missing."
        ),
        "panel_forecasts": panels,
        "pairwise_verification_panel_forecasts": pairwise_panels,
        "unit_definitions": {
            "N_Q": "labelled query spectra",
            "N_I": "unique query IK14 identity clusters",
            "N_F": "unique query formula clusters",
            "N_E": "query-candidate-molecule edges",
            "N_P": "query-reference-spectrum comparisons; never query count",
            "N_pairs": "labelled pairwise-verification query-reference pairs; never query count",
        },
        "quality_boundary": (
            "Q1/Q2 is the primary stratum. Q3 (including MCE-DRUG) is reported "
            "separately and may appear only in the explicitly expanded Q1/Q2/Q3 panel."
        ),
        "panel_build_ready": exclusion_complete and not source_not_ready and bool(panels),
    }
    return report, frame


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-peaks", type=int, default=DEFAULT_MIN_PEAKS)
    parser.add_argument("--precursor-validation-ppm", type=float, default=20.0)
    parser.add_argument("--precursor-validation-abs-da", type=float, default=0.01)
    parser.add_argument("--candidate-ppm", type=float, default=DEFAULT_PPM)
    parser.add_argument("--pairwise-hard-ppm", type=float, default=DEFAULT_PAIR_HARD_PPM)
    parser.add_argument("--pairwise-hard-abs-da", type=float, default=DEFAULT_PAIR_HARD_ABS_DA)
    parser.add_argument("--pairwise-outer-abs-da", type=float, default=DEFAULT_PAIR_OUTER_ABS_DA)
    parser.add_argument("--pairwise-positive-cap", type=int, default=DEFAULT_PAIR_POSITIVE_CAP)
    parser.add_argument("--pairwise-negative-cap", type=int, default=DEFAULT_PAIR_NEGATIVE_CAP)
    parser.add_argument(
        "--materialize-pre-exclusion-pairs",
        action="store_true",
        help="Write only integer row indices for the blocked score-blind Q1/Q2/Q3 pair axis.",
    )
    parser.add_argument("--split-seed", type=int, default=20261102)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite preflight output: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    report, frame = preflight(
        args.source_manifest,
        args.exclusion_registry,
        min_peaks=args.min_peaks,
        precursor_ppm=args.precursor_validation_ppm,
        precursor_abs_da=args.precursor_validation_abs_da,
        candidate_ppm=args.candidate_ppm,
        pair_hard_ppm=args.pairwise_hard_ppm,
        pair_hard_abs_da=args.pairwise_hard_abs_da,
        pair_outer_abs_da=args.pairwise_outer_abs_da,
        pair_positive_cap=args.pairwise_positive_cap,
        pair_negative_cap=args.pairwise_negative_cap,
        seed=args.split_seed,
    )
    pair_arrays = None
    if args.materialize_pre_exclusion_pairs and not frame.empty:
        pair_report, pair_arrays = forecast_pairwise_verification(
            frame[frame.library_quality.isin([1, 2, 3])].copy(),
            seed=args.split_seed,
            hard_ppm=args.pairwise_hard_ppm,
            hard_abs_da=args.pairwise_hard_abs_da,
            outer_abs_da=args.pairwise_outer_abs_da,
            positive_cap=args.pairwise_positive_cap,
            negative_cap=args.pairwise_negative_cap,
            return_indices=True,
        )
        expected = report["score_blind_pre_exclusion_pairwise_verification_forecasts"]["q123_expanded"]
        for field in ("N_Q_labelled_queries", "N_pairs_positive", "N_pairs_negative", "N_pairs_total"):
            if int(pair_report[field]) != int(expected[field]):
                raise RuntimeError(f"materialized pairwise axis drifted for {field}")
    with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / args.output_dir.name
        staging.mkdir()
        if pair_arrays is not None:
            pair_path = staging / "pre_exclusion_q123_pairwise_verification_indices.npz"
            np.savez_compressed(
                pair_path,
                query_row=pair_arrays["query_row"],
                reference_row=pair_arrays["reference_row"],
                label=(pair_arrays["pair_type"] == 1).astype(np.uint8),
                pair_type=pair_arrays["pair_type"],
            )
            report["pre_exclusion_pairwise_verification_artifact"] = {
                "path": pair_path.name,
                "bytes": pair_path.stat().st_size,
                "sha256": sha256_file(pair_path),
                "scope": "Q1/Q2/Q3 pre-exclusion; blocked_not_external until every required consumed source is excluded",
                "contains_spectra": False,
                "pair_type_codes": {
                    "1": "positive_same_IK14_distinct_spectrum",
                    "2": "negative_within_10ppm",
                    "3": "negative_outside_10ppm_within_hard_abs_da",
                    "4": "negative_outside_hard_abs_da_within_outer_abs_da",
                },
                "N_pairs": int(len(pair_arrays["query_row"])),
            }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not frame.empty:
            frame.to_csv(staging / "eligible_inventory.csv.gz", index=False, compression="gzip")
        staging.replace(args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output_dir),
        "panel_build_ready": report["panel_build_ready"],
        "required_sources_not_ready": report["required_sources_not_ready"],
        "missing_required_exclusions": report["overlap_exclusion"]["missing_required_sources"],
        "panel_forecasts": report["panel_forecasts"],
    }, indent=2))


if __name__ == "__main__":
    main()
