"""Shared, score-blind utilities for ChemAware transfer evaluation on MoNA."""
from __future__ import annotations

import hashlib
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def normalized_spectrum_signature(
    peaks_2_n: np.ndarray, precursor_mz: float, n_highest_peaks: int = 100,
) -> str:
    """Hash the exact normalized array presented to the DreaMS encoder.

    This deliberately removes intensity-scale duplicates.  It prevents repeated
    uploads of one spectrum from becoming a trivial query/positive pair.
    """
    raw = np.asarray(peaks_2_n, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] == 0:
        raise ValueError("spectrum must have shape (2, n) with at least one peak")
    highest = np.argsort(raw[1], kind="stable")[-n_highest_peaks:]
    highest = np.sort(highest)
    peaks = raw[:, highest].T.astype(np.float32, copy=True)
    if len(peaks) < n_highest_peaks:
        peaks = np.pad(peaks, ((0, n_highest_peaks - len(peaks)), (0, 0)))
    maximum = float(peaks[:, 1].max())
    if maximum > 0:
        peaks[:, 1] /= maximum
    precursor = np.asarray([[precursor_mz, 1.1]], dtype=np.float32)
    tokens = np.vstack((precursor, peaks))
    return hashlib.sha256(tokens.tobytes()).hexdigest()


def iter_mgf_records(
    path: Path, *, include_peaks: bool = False, compute_signature: bool = True,
    n_highest_peaks: int = 100,
) -> Iterator[dict[str, object]]:
    """Stream records using the same row convention as ``parse_mgf``.

    A row is counted only when it has a precursor and at least one numeric peak.
    This keeps panel row indices aligned with the existing MoNA embedding cache.
    """
    current: dict[str, object] | None = None
    peaks: list[tuple[float, float]] = []
    row = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                current = {}
                peaks = []
            elif line == "END IONS":
                # Match the repository's canonical MoNA parser exactly: zero or
                # missing precursor mass does not define an encoded library row.
                if current is not None and peaks and current.get("precursor_mz"):
                    array = np.asarray(peaks, dtype=np.float32).T
                    record = dict(current)
                    record["row"] = row
                    if compute_signature:
                        record["spectrum_hash"] = normalized_spectrum_signature(
                            array, float(record["precursor_mz"]), n_highest_peaks,
                        )
                    if include_peaks:
                        record["peaks"] = array
                    yield record
                    row += 1
                current = None
            elif current is not None and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip()
                if key == "PEPMASS":
                    try:
                        current["precursor_mz"] = float(value.split()[0])
                    except (ValueError, IndexError):
                        current["precursor_mz"] = None
                elif key == "SMILES":
                    current["smiles"] = value
                elif key == "INCHIKEY":
                    current["inchikey"] = value
                elif key == "NAME":
                    current["name"] = value
                elif key == "SOURCE":
                    current["source"] = value
                elif key in {"ADDUCT", "ION", "PRECURSORTYPE"}:
                    current["adduct"] = value
                elif key == "IONMODE":
                    current["ionmode"] = value
            elif current is not None:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except ValueError:
                        pass


def formula_from_structure(value: str) -> str:
    """Return a molecular formula from a SMILES or InChI field."""
    structure = html.unescape(str(value)).strip()
    if not structure or structure.lower() in {"n/a", "na", "none"}:
        return ""
    if structure.startswith("InChI="):
        molecule = Chem.MolFromInchi(structure)
    else:
        molecule = Chem.MolFromSmiles(structure)
    return rdMolDescriptors.CalcMolFormula(molecule) if molecule is not None else ""


def ppm_mask(values: np.ndarray, center: float, ppm: float) -> np.ndarray:
    return np.abs(values - center) <= max(abs(center), 1.0) * ppm * 1e-6


def prepare_metadata(
    records: list[dict[str, object]], development_identities: set[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Validate identities/structures and remove within-identity spectrum clones."""
    formula_cache: dict[str, str] = {}
    eligible: list[dict[str, object]] = []
    rejected_development = 0
    rejected_metadata = 0
    for record in records:
        identity = str(record.get("inchikey", ""))[:14]
        structure = str(record.get("smiles", ""))
        if structure not in formula_cache:
            formula_cache[structure] = formula_from_structure(structure)
        formula = formula_cache[structure]
        precursor_mz = float(record.get("precursor_mz", float("nan")))
        if identity in development_identities:
            rejected_development += 1
            continue
        if (
            len(identity) != 14 or not formula or not np.isfinite(precursor_mz)
            or len(str(record.get("spectrum_hash", ""))) != 64
        ):
            rejected_metadata += 1
            continue
        eligible.append({
            "row": int(record["row"]), "ik14": identity, "formula": formula,
            "precursor_mz": precursor_mz,
            "spectrum_hash": str(record["spectrum_hash"]),
        })

    # One normalized spectrum is allowed only once per molecular identity.
    deduplicated: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for record in eligible:
        key = (str(record["ik14"]), str(record["spectrum_hash"]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduplicated.append(record)
    return deduplicated, {
        "input_records": len(records),
        "rejected_development_identity_records": rejected_development,
        "rejected_invalid_metadata_records": rejected_metadata,
        "identity_spectrum_duplicates_removed": duplicates,
        "eligible_unique_records": len(deduplicated),
    }


def build_panel(
    records: list[dict[str, object]], *, ppm: float, queries_per_identity: int,
    references_per_candidate: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build a same-formula, mass-competitive retrieval panel without model scores."""
    if ppm <= 0 or queries_per_identity < 1 or references_per_candidate < 1:
        raise ValueError("invalid panel construction parameters")
    counts = Counter(str(record["ik14"]) for record in records)
    repeated = [record for record in records if counts[str(record["ik14"])] >= 2]
    identities_by_formula: dict[str, set[str]] = defaultdict(set)
    for record in repeated:
        identities_by_formula[str(record["formula"])].add(str(record["ik14"]))
    eligible = [
        record for record in repeated
        if len(identities_by_formula[str(record["formula"])]) >= 2
    ]

    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in eligible:
        grouped[str(record["formula"])][str(record["ik14"])].append(record)
    for identities in grouped.values():
        for identity_records in identities.values():
            identity_records.sort(key=lambda item: (float(item["precursor_mz"]), int(item["row"])))

    query_row: list[int] = []
    query_ik14: list[str] = []
    query_formula: list[str] = []
    query_spectrum_hash: list[str] = []
    query_ptr = [0]
    molecule_ik14: list[str] = []
    molecule_label: list[bool] = []
    molecule_ptr = [0]
    candidate_row: list[int] = []
    candidate_spectrum_hash: list[str] = []

    for formula in sorted(grouped):
        identities = grouped[formula]
        for identity in sorted(identities):
            source = identities[identity]
            if len(source) <= queries_per_identity:
                selected = source
            else:
                positions = np.unique(np.linspace(
                    0, len(source) - 1, queries_per_identity, dtype=int,
                ))
                selected = [source[int(position)] for position in positions]
            for query in selected:
                center = float(query["precursor_mz"])
                candidate_groups: list[tuple[str, list[dict[str, object]]]] = []
                for candidate_identity in sorted(identities):
                    available = identities[candidate_identity]
                    masses = np.asarray([float(item["precursor_mz"]) for item in available])
                    allowed = [
                        item for item, keep in zip(available, ppm_mask(masses, center, ppm), strict=True)
                        if keep and int(item["row"]) != int(query["row"])
                    ]
                    allowed.sort(key=lambda item: (
                        abs(float(item["precursor_mz"]) - center), int(item["row"]),
                    ))
                    if allowed:
                        candidate_groups.append((
                            candidate_identity, allowed[:references_per_candidate],
                        ))
                positive = [item for item in candidate_groups if item[0] == identity]
                negative = [item for item in candidate_groups if item[0] != identity]
                if len(positive) != 1 or not negative:
                    continue
                query_row.append(int(query["row"]))
                query_ik14.append(identity)
                query_formula.append(formula)
                query_spectrum_hash.append(str(query["spectrum_hash"]))
                for candidate_identity, references in positive + negative:
                    molecule_ik14.append(candidate_identity)
                    molecule_label.append(candidate_identity == identity)
                    candidate_row.extend(int(item["row"]) for item in references)
                    candidate_spectrum_hash.extend(str(item["spectrum_hash"]) for item in references)
                    molecule_ptr.append(len(candidate_row))
                query_ptr.append(len(molecule_ik14))

    panel = {
        "query_row": np.asarray(query_row, dtype=np.int64),
        "query_ik14": np.asarray(query_ik14, dtype="U14"),
        "query_formula": np.asarray(query_formula, dtype="U64"),
        "query_spectrum_hash": np.asarray(query_spectrum_hash, dtype="U64"),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_ik14": np.asarray(molecule_ik14, dtype="U14"),
        "molecule_label": np.asarray(molecule_label, dtype=bool),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "candidate_row": np.asarray(candidate_row, dtype=np.int64),
        "candidate_spectrum_hash": np.asarray(candidate_spectrum_hash, dtype="U64"),
    }
    validate_panel(panel)
    candidate_counts = np.diff(panel["query_ptr"])
    reference_counts = np.diff(panel["molecule_ptr"])
    summary: dict[str, object] = {
        "queries": len(query_row),
        "query_identities": len(set(query_ik14)),
        "candidate_identities": len(set(molecule_ik14)),
        "formulas": len(set(query_formula)),
        "candidate_molecules": len(molecule_ik14),
        "candidate_spectra": len(candidate_row),
        "candidates_per_query": {
            "median": float(np.median(candidate_counts)) if len(candidate_counts) else 0.0,
            "p90": float(np.quantile(candidate_counts, 0.9)) if len(candidate_counts) else 0.0,
            "maximum": int(candidate_counts.max()) if len(candidate_counts) else 0,
        },
        "references_per_candidate": {
            "median": float(np.median(reference_counts)) if len(reference_counts) else 0.0,
            "p90": float(np.quantile(reference_counts, 0.9)) if len(reference_counts) else 0.0,
            "maximum": int(reference_counts.max()) if len(reference_counts) else 0,
        },
    }
    return panel, summary


def validate_panel(panel: dict[str, np.ndarray]) -> None:
    queries = len(panel["query_row"])
    if len(panel["query_ptr"]) != queries + 1:
        raise RuntimeError("query pointer length mismatch")
    if len(panel["molecule_ptr"]) != len(panel["molecule_ik14"]) + 1:
        raise RuntimeError("molecule pointer length mismatch")
    if len(panel["candidate_row"]) != len(panel["candidate_spectrum_hash"]):
        raise RuntimeError("candidate row/hash mismatch")
    for query in range(queries):
        left, right = map(int, panel["query_ptr"][query:query + 2])
        labels = panel["molecule_label"][left:right]
        if len(labels) < 2 or np.sum(labels) != 1 or not bool(labels[0]):
            raise RuntimeError(f"query {query} lacks a unique-first positive")
        positive_left, positive_right = map(int, panel["molecule_ptr"][left:left + 2])
        positive_hashes = set(map(str, panel["candidate_spectrum_hash"][positive_left:positive_right]))
        if str(panel["query_spectrum_hash"][query]) in positive_hashes:
            raise RuntimeError(f"query {query} has an identical positive spectrum clone")
        if int(panel["query_row"][query]) in set(map(
            int, panel["candidate_row"][positive_left:positive_right],
        )):
            raise RuntimeError(f"query {query} appears as its own positive reference")
