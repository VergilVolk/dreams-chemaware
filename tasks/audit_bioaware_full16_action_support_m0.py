#!/usr/bin/env python
"""Audit full-16-panel BioAware action-discovery support before model fitting.

This is a deliberately model-free M0 stage.  It reconstructs the exact
strict-10-ppm, same-adduct candidate sets for every MetDNA3 Level-1 row, then
asks whether the *available data* can support a source-held-out BioAware
action study.  Rhea and KEGG are used only as graph topology here: no
embedding is read, no score is fitted, and no action is selected.

For every query, source-level seed pools exclude the query truth identity and
every observed seed sharing the truth formula.  Truth-vs-wrong topology fields
are therefore labelled as headroom diagnostics; they are not deployable
features and must never enter a later model without cross-fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCES = ("BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma")
EXPECTED_SEPARATIONS = ("hilic", "rplc")
EXPECTED_POLARITIES = ("negative", "positive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: Iterable[object]) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, bytearray, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=object,
    )


def finite_quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


class CandidateIndex:
    """Mass-sorted per-adduct index with molecule-level aggregation."""

    def __init__(
        self,
        precursor_mz: np.ndarray,
        adduct: np.ndarray,
        ik14: np.ndarray,
        formula: np.ndarray,
        reference_source: np.ndarray | None = None,
    ) -> None:
        if reference_source is None:
            reference_source = np.full(len(precursor_mz), "unspecified", dtype=object)
        if not (
            len(precursor_mz) == len(adduct) == len(ik14) == len(formula) == len(reference_source)
        ):
            raise ValueError("candidate index arrays must have identical lengths")
        self._by_adduct: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        valid = (
            np.isfinite(precursor_mz)
            & (np.char.str_len(ik14.astype(str)) == 14)
            & (ik14.astype(str) != "nan")
            & (adduct.astype(str) != "nan")
        )
        for value in sorted(set(adduct[valid].astype(str))):
            positions = np.flatnonzero(valid & (adduct.astype(str) == value))
            order = np.argsort(precursor_mz[positions], kind="stable")
            positions = positions[order]
            self._by_adduct[value] = (
                precursor_mz[positions].astype(float, copy=False),
                ik14[positions].astype(object, copy=False),
                formula[positions].astype(object, copy=False),
                reference_source[positions].astype(object, copy=False),
            )

    @property
    def supported_adducts(self) -> list[str]:
        return sorted(self._by_adduct)

    def query(self, mz: float, adduct: str, ppm: float) -> list[dict[str, object]]:
        payload = self._by_adduct.get(str(adduct))
        if payload is None or not np.isfinite(mz) or mz <= 0:
            return []
        masses, identities, formulas, sources = payload
        tolerance = float(mz) * float(ppm) * 1e-6
        left = int(np.searchsorted(masses, mz - tolerance, side="left"))
        right = int(np.searchsorted(masses, mz + tolerance, side="right"))
        grouped: dict[str, list[int]] = defaultdict(list)
        for position in range(left, right):
            grouped[str(identities[position])].append(position)
        result = []
        for identity in sorted(grouped):
            positions = grouped[identity]
            observed_formulas = {
                str(formulas[position])
                for position in positions
                if str(formulas[position]) not in {"", "nan", "None"}
            }
            if len(observed_formulas) > 1:
                raise RuntimeError(
                    f"candidate identity {identity} has conflicting formulas in one mass window: "
                    f"{sorted(observed_formulas)}"
                )
            result.append(
                {
                    "candidate_ik14": identity,
                    "candidate_formula": next(iter(observed_formulas), ""),
                    "candidate_spectra": int(len(positions)),
                    "candidate_reference_sources": "|".join(
                        sorted({str(sources[position]) for position in positions})
                    ),
                    "candidate_nearest_mass_error_ppm": float(
                        min(abs(float(masses[position]) - mz) for position in positions) / mz * 1e6
                    ),
                }
            )
        return result


def load_candidate_index(
    hdf5_path: Path,
    negative_manifest_path: Path,
    negative_report_path: Path,
    negative_integrity_report_path: Path,
    negative_approved_rows_path: Path,
) -> tuple[CandidateIndex, dict[str, object]]:
    """Combine positive HDF5 candidates with validated MONA [M-H]- rows."""
    with h5py.File(hdf5_path, "r") as handle:
        required = {"precursor_mz", "adduct", "INCHIKEY", "FORMULA"}
        missing = required - set(handle.keys())
        if missing:
            raise RuntimeError(f"reference HDF5 lacks fields: {sorted(missing)}")
        mz = np.asarray(handle["precursor_mz"][:], dtype=float)
        adduct = decode(handle["adduct"][:])
        ik14 = np.asarray([value[:14].upper() for value in decode(handle["INCHIKEY"][:])])
        formula = decode(handle["FORMULA"][:])
    hdf5_valid = (
        np.isfinite(mz)
        & (mz > 0)
        & (np.char.str_len(ik14.astype(str)) == 14)
        & (adduct.astype(str) != "nan")
    )

    negative_report = json.loads(negative_report_path.read_text(encoding="utf-8"))
    if negative_report.get("status") != "bioaware_metdna3_external_negative_dreams_complete":
        raise RuntimeError("frozen negative benchmark report status mismatch")
    if negative_report.get("library_audit", {}).get("chemically_filtered_for_m_h") is not True:
        raise RuntimeError("negative benchmark is not chemically filtered for [M-H]-")
    integrity = json.loads(negative_integrity_report_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "mona_negative_library_chemical_integrity_complete":
        raise RuntimeError("MONA-negative chemical-integrity report status mismatch")
    if integrity.get("declared_adduct_scope") != "[M-H]-":
        raise RuntimeError("MONA-negative chemical-integrity adduct scope mismatch")
    observed_manifest_hash = sha256(negative_manifest_path)
    if (
        negative_report.get("provenance", {}).get("library_manifest_sha256")
        != observed_manifest_hash
    ):
        raise RuntimeError("MONA-negative manifest does not match frozen negative benchmark")
    if integrity.get("provenance", {}).get("manifest_sha256") != observed_manifest_hash:
        raise RuntimeError("MONA-negative manifest does not match chemical-integrity audit")
    approved_hash = sha256(negative_approved_rows_path)
    if integrity.get("provenance", {}).get("approved_rows_sha256") != approved_hash:
        raise RuntimeError("approved MONA [M-H]- rows do not match chemical-integrity audit")
    if negative_report.get("provenance", {}).get("approved_library_rows_sha256") != approved_hash:
        raise RuntimeError("approved MONA [M-H]- rows do not match filtered benchmark")
    negative = pd.read_csv(negative_manifest_path)
    required = {"inchikey", "precursor_mz"}
    if not required.issubset(negative.columns):
        raise RuntimeError(
            f"MONA-negative manifest lacks fields: {sorted(required-set(negative.columns))}"
        )
    negative_mz_all = pd.to_numeric(negative.precursor_mz, errors="coerce").to_numpy(float)
    negative_ik14_all = (
        negative.inchikey.fillna("").astype(str).str[:14].str.upper().to_numpy(object)
    )
    negative_valid = (
        np.isfinite(negative_mz_all)
        & (negative_mz_all >= 50.0)
        & (negative_mz_all <= 1500.0)
        & (np.char.str_len(negative_ik14_all.astype(str)) == 14)
    )
    approved_rows = np.load(negative_approved_rows_path, allow_pickle=False)
    if (
        approved_rows.ndim != 1
        or not np.issubdtype(approved_rows.dtype, np.integer)
        or len(approved_rows) == 0
        or int(approved_rows.min()) < 0
        or int(approved_rows.max()) >= len(negative)
        or len(np.unique(approved_rows)) != len(approved_rows)
    ):
        raise RuntimeError("approved MONA [M-H]- row index is invalid")
    approved_mask = np.zeros(len(negative), dtype=bool)
    approved_mask[approved_rows.astype(np.int64)] = True
    negative_valid &= approved_mask
    negative_mz = negative_mz_all[negative_valid]
    negative_ik14 = negative_ik14_all[negative_valid]
    negative_adduct = np.full(len(negative_mz), "[M-H]-", dtype=object)
    # The validated manifest has no molecular-formula column.  Missing is
    # explicit; formula is not inferred from structure in this model-free M0.
    negative_formula = np.full(len(negative_mz), "", dtype=object)

    combined_mz = np.concatenate([mz[hdf5_valid], negative_mz])
    combined_adduct = np.concatenate([adduct[hdf5_valid], negative_adduct])
    combined_ik14 = np.concatenate([ik14[hdf5_valid], negative_ik14])
    combined_formula = np.concatenate([formula[hdf5_valid], negative_formula])
    combined_source = np.concatenate(
        [
            np.full(int(hdf5_valid.sum()), "massspecgym_hdf5", dtype=object),
            np.full(len(negative_mz), "mona_negative_manifest", dtype=object),
        ]
    )
    index = CandidateIndex(
        combined_mz,
        combined_adduct,
        combined_ik14,
        combined_formula,
        combined_source,
    )
    return index, {
        "reference_rows": int(len(combined_mz)),
        "identities": int(len(set(combined_ik14))),
        "formula_values_available": int(
            len({str(value) for value in combined_formula if str(value) not in {"", "nan"}})
        ),
        "supported_adducts": index.supported_adducts,
        "reference_sources": {
            "massspecgym_hdf5": {
                "input_rows": int(len(mz)),
                "valid_rows": int(hdf5_valid.sum()),
                "identities": int(len(set(ik14[hdf5_valid]))),
                "adducts": sorted(set(adduct[hdf5_valid].astype(str))),
            },
            "mona_negative_manifest": {
                "input_rows": int(len(negative)),
                "valid_rows": int(negative_valid.sum()),
                "chemically_approved_rows": int(len(approved_rows)),
                "identities": int(len(set(negative_ik14))),
                "assumed_adduct": "[M-H]-",
                "formula_metadata_available": False,
                "frozen_negative_benchmark_queries": int(negative_report.get("queries", 0)),
                "frozen_negative_benchmark_identities": int(negative_report.get("identities", 0)),
                "frozen_negative_benchmark_formulas": int(negative_report.get("formulas", 0)),
                "scope_limit": (
                    "MGF declares negative ion mode but has no per-record adduct; only external "
                    "[M-H]- queries may use this source"
                ),
            },
        },
    }


def load_external_truth(manifest_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    report_path = manifest_root / "report.json"
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_metdna3_external_manifest_frozen":
        raise RuntimeError("external manifest status mismatch")
    unit_ids = sorted(report.get("units", {}))
    if len(unit_ids) != 8:
        raise RuntimeError(f"expected exactly 8 manifest units, got {len(unit_ids)}")
    parts = []
    hashes: dict[str, str] = {}
    for unit_id in unit_ids:
        path = manifest_root / unit_id / "external_level1.csv.gz"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        unit_report = report["units"][unit_id]
        observed_hash = sha256(path)
        if unit_report.get("truth_sha256") != observed_hash:
            raise RuntimeError(f"frozen truth hash mismatch: {unit_id}")
        required = {
            "peak_name", "mz", "formula", "ik14", "adduct", "polarity",
            "separation", "unit_id", "panel_id", "sample_type",
        }
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{path} lacks fields: {sorted(required-set(frame.columns))}")
        if set(frame.unit_id.astype(str)) != {unit_id}:
            raise RuntimeError(f"unit_id mismatch in {path}")
        expected_unit = (
            frame.sample_type.astype(str) + "__" + frame.separation.astype(str)
        )
        expected_panel = expected_unit + "__" + frame.polarity.astype(str)
        if not frame.unit_id.astype(str).equals(expected_unit):
            raise RuntimeError(f"row-level source/separation unit mismatch in {path}")
        if not frame.panel_id.astype(str).equals(expected_panel):
            raise RuntimeError(f"row-level panel label mismatch in {path}")
        adduct_sign = frame.adduct.astype(str).str[-1]
        expected_sign = frame.polarity.astype(str).map({"positive": "+", "negative": "-"})
        if expected_sign.isna().any() or not adduct_sign.equals(expected_sign):
            raise RuntimeError(f"polarity/adduct sign mismatch in {path}")
        observed_panels = {
            str(key): int(value)
            for key, value in frame.panel_id.value_counts().sort_index().items()
        }
        if int(unit_report.get("level1_rows", -1)) != len(frame):
            raise RuntimeError(f"frozen Level-1 row count mismatch: {unit_id}")
        if int(unit_report.get("level1_identities", -1)) != frame.ik14.nunique():
            raise RuntimeError(f"frozen Level-1 identity count mismatch: {unit_id}")
        if unit_report.get("panels") != observed_panels:
            raise RuntimeError(f"frozen panel count mismatch: {unit_id}")
        frame = frame.copy()
        frame["manifest_row"] = np.arange(len(frame), dtype=np.int64)
        frame["query_id"] = [
            f"{unit_id}|{panel}|{position:05d}"
            for panel, position in zip(frame.panel_id.astype(str), frame.manifest_row)
        ]
        parts.append(frame)
        hashes[unit_id] = observed_hash
    truth = pd.concat(parts, ignore_index=True)
    sources = sorted(set(truth.sample_type.astype(str)))
    panels = sorted(set(truth.panel_id.astype(str)))
    units = sorted(set(truth.unit_id.astype(str)))
    expected_panels = {
        f"{source}__{separation}__{polarity}"
        for source in EXPECTED_SOURCES
        for separation in EXPECTED_SEPARATIONS
        for polarity in EXPECTED_POLARITIES
    }
    if len(sources) != 4 or set(sources) != set(EXPECTED_SOURCES):
        raise RuntimeError(f"expected exactly four biological sources, got {sources}")
    if len(panels) != 16 or set(panels) != expected_panels:
        raise RuntimeError(f"expected exactly 16 panels, got {panels}")
    if len(units) != 8:
        raise RuntimeError(f"expected exactly 8 source-separation units, got {units}")
    if truth.query_id.duplicated().any():
        raise RuntimeError("external query IDs are not unique")
    if truth.ik14.astype(str).str.len().ne(14).any():
        raise RuntimeError("external truth contains invalid IK14")
    return truth, hashes


def undirected_adjacency(edges: pd.DataFrame, left: str, right: str) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for a, b in edges[[left, right]].itertuples(index=False, name=None):
        a, b = str(a)[:14].upper(), str(b)[:14].upper()
        if len(a) != 14 or len(b) != 14 or a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    return dict(adjacency)


def load_networks(rhea_path: Path, kegg_path: Path) -> tuple[dict[str, dict[str, set[str]]], dict[str, object]]:
    rhea = pd.read_csv(rhea_path)
    required = {"identity_a", "identity_b", "relation_type"}
    if not required.issubset(rhea.columns):
        raise RuntimeError(f"Rhea relation table lacks fields: {sorted(required-set(rhea.columns))}")
    rhea = rhea.loc[rhea.relation_type.astype(str) == "reaction_direction_unknown"].copy()
    rhea_adjacency = undirected_adjacency(rhea, "identity_a", "identity_b")

    kegg = pd.read_csv(kegg_path)
    required = {"ik14_a", "ik14_b"}
    if not required.issubset(kegg.columns):
        raise RuntimeError(f"KEGG edge table lacks fields: {sorted(required-set(kegg.columns))}")
    kegg_adjacency = undirected_adjacency(kegg, "ik14_a", "ik14_b")

    union: dict[str, set[str]] = defaultdict(set)
    for graph in (rhea_adjacency, kegg_adjacency):
        for node, neighbours in graph.items():
            union[node].update(neighbours)
    graphs = {"rhea": rhea_adjacency, "kegg": kegg_adjacency, "union": dict(union)}
    report = {}
    for name, graph in graphs.items():
        report[name] = {
            "nodes": int(len(graph)),
            "undirected_edges": int(sum(len(neighbours) for neighbours in graph.values()) // 2),
            "degree": finite_quantiles(len(neighbours) for neighbours in graph.values()),
        }
    return graphs, report


def strict_seed_pool(
    source: str,
    truth_identity: str,
    truth_formula: str,
    source_identities: dict[str, set[str]],
    identity_formulas: dict[str, set[str]],
) -> set[str]:
    return {
        identity
        for identity in source_identities[source]
        if identity != truth_identity and truth_formula not in identity_formulas.get(identity, set())
    }


def topology_outcome(truth_score: int, wrong_scores: list[int]) -> str:
    if not wrong_scores:
        raise RuntimeError("topology outcome requires at least one wrong candidate")
    maximum = max(wrong_scores)
    if truth_score > maximum:
        return "truth_advantaged"
    if truth_score < maximum:
        return "wrong_advantaged"
    return "tied"


def build_candidate_tables(
    truth: pd.DataFrame,
    index: CandidateIndex,
    graphs: dict[str, dict[str, set[str]]],
    ppm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_identities = {
        str(source): set(group.ik14.astype(str))
        for source, group in truth.groupby("sample_type", sort=True)
    }
    identity_formulas: dict[str, set[str]] = defaultdict(set)
    for identity, formula in truth[["ik14", "formula"]].itertuples(index=False, name=None):
        identity_formulas[str(identity)].add(str(formula))

    query_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    for row in truth.itertuples(index=False):
        candidates = index.query(float(row.mz), str(row.adduct), ppm)
        identities = {str(candidate["candidate_ik14"]) for candidate in candidates}
        covered = str(row.ik14) in identities
        ambiguous = covered and len(identities) >= 2
        base_record = {
            "query_id": str(row.query_id),
            "source": str(row.sample_type),
            "unit_id": str(row.unit_id),
            "panel_id": str(row.panel_id),
            "separation": str(row.separation),
            "polarity": str(row.polarity),
            "peak_name": str(row.peak_name),
            "precursor_mz": float(row.mz),
            "adduct": str(row.adduct),
            "truth_ik14": str(row.ik14),
            "truth_formula": str(row.formula),
            "truth_present": bool(covered),
            "candidate_molecules": int(len(identities)),
            "ambiguous": bool(ambiguous),
        }
        if not ambiguous:
            query_records.append(base_record)
            continue

        seeds = strict_seed_pool(
            str(row.sample_type), str(row.ik14), str(row.formula),
            source_identities, identity_formulas,
        )
        query_candidates: list[dict[str, object]] = []
        for candidate in candidates:
            identity = str(candidate["candidate_ik14"])
            record = dict(base_record)
            record.update(candidate)
            record["is_truth"] = identity == str(row.ik14)
            record["strict_source_seed_identities"] = int(len(seeds))
            for name, graph in graphs.items():
                neighbours = graph.get(identity, set())
                record[f"{name}_node"] = identity in graph
                record[f"{name}_degree"] = int(len(neighbours))
                record[f"{name}_strict_seed_support"] = int(len(neighbours & seeds))
            query_candidates.append(record)
            candidate_records.append(record)

        query_record = dict(base_record)
        query_record["strict_source_seed_identities"] = int(len(seeds))
        for name in graphs:
            truth_rows = [item for item in query_candidates if item["is_truth"]]
            wrong_rows = [item for item in query_candidates if not item["is_truth"]]
            if len(truth_rows) != 1 or not wrong_rows:
                raise RuntimeError(f"invalid molecule aggregation for query {row.query_id}")
            truth_score = int(truth_rows[0][f"{name}_strict_seed_support"])
            wrong_scores = [int(item[f"{name}_strict_seed_support"]) for item in wrong_rows]
            scores = [truth_score, *wrong_scores]
            outcome = topology_outcome(truth_score, wrong_scores)
            query_record[f"{name}_any_candidate_node"] = any(
                bool(item[f"{name}_node"]) for item in query_candidates
            )
            query_record[f"{name}_any_candidate_supported"] = any(score > 0 for score in scores)
            query_record[f"{name}_truth_support"] = truth_score
            query_record[f"{name}_max_wrong_support"] = max(wrong_scores)
            query_record[f"{name}_support_discriminative"] = len(set(scores)) > 1
            query_record[f"{name}_outcome"] = outcome
            query_record[f"{name}_truth_advantaged"] = outcome == "truth_advantaged"
            query_record[f"{name}_wrong_advantaged"] = outcome == "wrong_advantaged"
        query_records.append(query_record)

    queries = pd.DataFrame(query_records)
    candidates = pd.DataFrame(candidate_records)
    if queries.query_id.duplicated().any():
        raise RuntimeError("query table has duplicate IDs")
    if not candidates.empty and candidates.duplicated(["query_id", "candidate_ik14"]).any():
        raise RuntimeError("candidate table was not aggregated to unique query/identity")
    return queries, candidates


def network_summary(ambiguous: pd.DataFrame, network: str) -> dict[str, object]:
    outcomes = ambiguous[f"{network}_outcome"].value_counts().to_dict()
    edge_bearing = ambiguous.loc[ambiguous[f"{network}_any_candidate_supported"]]
    return {
        "queries_with_any_candidate_node": int(ambiguous[f"{network}_any_candidate_node"].sum()),
        "queries_with_strict_seed_support": int(ambiguous[f"{network}_any_candidate_supported"].sum()),
        "identities_with_strict_seed_support": int(edge_bearing.truth_ik14.nunique()),
        "formulas_with_strict_seed_support": int(edge_bearing.truth_formula.nunique()),
        "support_discriminative_queries": int(ambiguous[f"{network}_support_discriminative"].sum()),
        "truth_advantaged_queries": int(outcomes.get("truth_advantaged", 0)),
        "wrong_advantaged_queries": int(outcomes.get("wrong_advantaged", 0)),
        "tied_queries": int(outcomes.get("tied", 0)),
        "truth_advantage_headroom_rate_all_ambiguous": float(
            ambiguous[f"{network}_truth_advantaged"].mean()
        ),
        "truth_advantage_rate_among_supported": float(
            edge_bearing[f"{network}_truth_advantaged"].mean()
        ) if len(edge_bearing) else 0.0,
    }


def subgroup_summary(frame: pd.DataFrame) -> dict[str, object]:
    ambiguous = frame.loc[frame.ambiguous].copy()
    result: dict[str, object] = {
        "level1_rows": int(len(frame)),
        "level1_identities": int(frame.truth_ik14.nunique()),
        "level1_formulas": int(frame.truth_formula.nunique()),
        "truth_covered_rows": int(frame.truth_present.sum()),
        "truth_covered_identities": int(frame.loc[frame.truth_present, "truth_ik14"].nunique()),
        "ambiguous_rows": int(len(ambiguous)),
        "ambiguous_identities": int(ambiguous.truth_ik14.nunique()),
        "ambiguous_formulas": int(ambiguous.truth_formula.nunique()),
        "candidate_molecules": finite_quantiles(ambiguous.candidate_molecules),
    }
    if len(ambiguous):
        result["network"] = {
            name: network_summary(ambiguous, name) for name in ("rhea", "kegg", "union")
        }
    else:
        result["network"] = {}
    return result


def source_overlap(ambiguous: pd.DataFrame) -> pd.DataFrame:
    records = []
    by_source = {
        str(source): group for source, group in ambiguous.groupby("source", sort=True)
    }
    for source_a in EXPECTED_SOURCES:
        for source_b in EXPECTED_SOURCES:
            a = by_source.get(source_a, ambiguous.iloc[0:0])
            b = by_source.get(source_b, ambiguous.iloc[0:0])
            records.append(
                {
                    "source_a": source_a,
                    "source_b": source_b,
                    "identity_overlap": int(len(set(a.truth_ik14) & set(b.truth_ik14))),
                    "formula_overlap": int(len(set(a.truth_formula) & set(b.truth_formula))),
                }
            )
    return pd.DataFrame(records)


def loso_report(ambiguous: pd.DataFrame, full_truth: pd.DataFrame) -> pd.DataFrame:
    records = []
    for held_source in EXPECTED_SOURCES:
        held = ambiguous.loc[ambiguous.source == held_source]
        held_full = full_truth.loc[full_truth.source == held_source]
        train = ambiguous.loc[ambiguous.source != held_source]
        # Purge against the complete held-source Level-1 universe, not only its
        # ambiguous subset.  Otherwise an uncovered/nonambiguous held feature
        # can leak its identity or formula into the action-discovery fold.
        held_identities = set(held_full.truth_ik14)
        held_formulas = set(held_full.truth_formula)
        purged = train.loc[
            ~train.truth_ik14.isin(held_identities)
            & ~train.truth_formula.isin(held_formulas)
        ]
        edge_bearing = purged.loc[purged.union_any_candidate_supported]
        records.append(
            {
                "held_source": held_source,
                "held_rows": int(len(held)),
                "held_identities": int(held.truth_ik14.nunique()),
                "held_formulas": int(held.truth_formula.nunique()),
                "held_full_level1_rows_used_for_purge": int(len(held_full)),
                "held_full_level1_identities_used_for_purge": int(held_full.truth_ik14.nunique()),
                "held_full_level1_formulas_used_for_purge": int(held_full.truth_formula.nunique()),
                "train_rows_before_purge": int(len(train)),
                "train_rows_after_identity_formula_purge": int(len(purged)),
                "train_identities_after_purge": int(purged.truth_ik14.nunique()),
                "train_formulas_after_purge": int(purged.truth_formula.nunique()),
                "train_sources_after_purge": int(purged.source.nunique()),
                "train_panels_after_purge": int(purged.panel_id.nunique()),
                "train_union_supported_rows_after_purge": int(len(edge_bearing)),
                "train_union_supported_identities_after_purge": int(edge_bearing.truth_ik14.nunique()),
                "train_union_supported_formulas_after_purge": int(edge_bearing.truth_formula.nunique()),
            }
        )
    return pd.DataFrame(records)


def write_json_atomic(path: Path, body: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_manifest_v1",
    )
    parser.add_argument(
        "--reference-hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--negative-library-manifest", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv",
    )
    parser.add_argument(
        "--negative-library-report", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/report.json",
    )
    parser.add_argument(
        "--negative-library-integrity-report", type=Path,
        default=ROOT / "data/validation/mona_negative_library_chemical_integrity_v1/report.json",
    )
    parser.add_argument(
        "--negative-approved-rows", type=Path,
        default=ROOT / "data/validation/mona_negative_library_chemical_integrity_v1/approved_m_h_library_rows.npy",
    )
    parser.add_argument(
        "--rhea-relations", type=Path,
        default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/identity_pairs.csv.gz",
    )
    parser.add_argument(
        "--kegg-edges", type=Path,
        default=ROOT / "data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/bioaware_full16_action_support_m0_v6_20260906",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    args = parser.parse_args()

    for path in (
        args.manifest_root / "report.json", args.reference_hdf5,
        args.negative_library_manifest, args.negative_library_report,
        args.negative_library_integrity_report, args.negative_approved_rows,
        args.rhea_relations, args.kegg_edges,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if not np.isclose(args.ppm, 10.0, rtol=0.0, atol=1e-12):
        raise ValueError("formal M0 protocol is pinned to exactly 10 ppm")

    truth, truth_hashes = load_external_truth(args.manifest_root)
    index, library_report = load_candidate_index(
        args.reference_hdf5,
        args.negative_library_manifest,
        args.negative_library_report,
        args.negative_library_integrity_report,
        args.negative_approved_rows,
    )
    graphs, graph_report = load_networks(args.rhea_relations, args.kegg_edges)
    queries, candidates = build_candidate_tables(truth, index, graphs, args.ppm)
    ambiguous = queries.loc[queries.ambiguous].copy()
    if ambiguous.empty:
        raise RuntimeError("strict candidate construction produced no ambiguous queries")

    overlap = source_overlap(ambiguous)
    loso = loso_report(ambiguous, queries)
    source_reports = {
        str(source): subgroup_summary(group)
        for source, group in queries.groupby("source", sort=True)
    }
    panel_reports = {
        str(panel): subgroup_summary(group)
        for panel, group in queries.groupby("panel_id", sort=True)
    }
    polarity_reports = {
        str(polarity): subgroup_summary(group)
        for polarity, group in queries.groupby("polarity", sort=True)
    }
    supported_panels = sorted(
        panel for panel, payload in panel_reports.items()
        if int(payload["truth_covered_rows"]) > 0
    )
    unsupported_panels = sorted(set(panel_reports) - set(supported_panels))
    supported_polarities = sorted(
        polarity for polarity, payload in polarity_reports.items()
        if int(payload["truth_covered_rows"]) > 0
    )
    truth_adduct_counts = {
        str(key): int(value)
        for key, value in queries.adduct.value_counts().sort_index().items()
    }
    covered_adduct_counts = {
        str(key): int(value)
        for key, value in queries.loc[
            queries.truth_present, "adduct"
        ].value_counts().sort_index().items()
    }
    unsupported_truth_adducts = sorted(
        set(queries.adduct.astype(str)) - set(index.supported_adducts)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    queries_path = args.output_dir / "queries.csv.gz"
    candidates_path = args.output_dir / "candidate_identities.csv.gz"
    overlap_path = args.output_dir / "source_overlap.csv"
    loso_path = args.output_dir / "source_loso.csv"
    queries.to_csv(queries_path, index=False, compression="gzip")
    candidates.to_csv(candidates_path, index=False, compression="gzip")
    overlap.to_csv(overlap_path, index=False)
    loso.to_csv(loso_path, index=False)

    overall = subgroup_summary(queries)
    per_source_union_identities = {
        source: int(
            ambiguous.loc[
                (ambiguous.source == source) & ambiguous.union_any_candidate_supported,
                "truth_ik14",
            ].nunique()
        )
        for source in EXPECTED_SOURCES
    }
    gates = {
        "exactly_four_biological_sources": int(queries.source.nunique()) == 4,
        "exactly_sixteen_panels": int(queries.panel_id.nunique()) == 16,
        "candidate_library_supports_both_polarities": (
            any(str(value).endswith("+") for value in index.supported_adducts)
            and any(str(value).endswith("-") for value in index.supported_adducts)
        ),
        "all_truth_adducts_supported": len(unsupported_truth_adducts) == 0,
        "every_panel_has_truth_coverage": all(
            int(panel_reports[panel]["truth_covered_rows"]) > 0
            for panel in sorted(panel_reports)
        ),
        "ambiguous_identities_ge_600": int(overall["ambiguous_identities"]) >= 600,
        "ambiguous_formulas_ge_450": int(overall["ambiguous_formulas"]) >= 450,
        "each_source_ambiguous_identities_ge_100": all(
            int(source_reports[source]["ambiguous_identities"]) >= 100
            for source in EXPECTED_SOURCES
        ),
        "each_source_union_supported_identities_ge_100": all(
            value >= 100 for value in per_source_union_identities.values()
        ),
        "each_loso_train_identities_ge_400_after_purge": bool(
            (loso.train_identities_after_purge >= 400).all()
        ),
        "each_loso_train_formulas_ge_300_after_purge": bool(
            (loso.train_formulas_after_purge >= 300).all()
        ),
    }
    report = {
        "schema_version": "bioaware_full16_m0_v1",
        "status": "bioaware_full16_action_support_m0_complete",
        "formal": True,
        "model_fitted": False,
        "embedding_values_read": False,
        "opened_external_manifest_is_development_data": True,
        "manifest": {
            "sources": int(queries.source.nunique()),
            "units": int(queries.unit_id.nunique()),
            "panels": int(queries.panel_id.nunique()),
            "level1_rows": int(len(queries)),
            "level1_identities": int(queries.truth_ik14.nunique()),
            "level1_formulas": int(queries.truth_formula.nunique()),
        },
        "candidate_library": library_report,
        "candidate_protocol_evaluability": {
            "supported_panels": supported_panels,
            "unsupported_panels": unsupported_panels,
            "supported_polarities": supported_polarities,
            "truth_rows_by_adduct": truth_adduct_counts,
            "truth_covered_rows_by_adduct": covered_adduct_counts,
            "unsupported_truth_adducts": unsupported_truth_adducts,
            "unsupported_truth_rows": int(
                queries.adduct.astype(str).isin(unsupported_truth_adducts).sum()
            ),
            "warning": (
                "Unsupported adduct rows are unevaluable, not zero-effect observations. "
                "MONA-negative is valid only for the explicitly assumed [M-H]- scope."
            ),
        },
        "strict_10ppm_same_adduct": overall,
        "by_source": source_reports,
        "by_polarity": polarity_reports,
        "by_panel": panel_reports,
        "network_graphs": graph_report,
        "per_source_union_supported_identities": per_source_union_identities,
        "leave_one_biological_source_out": loso.to_dict(orient="records"),
        "source_overlap": overlap.to_dict(orient="records"),
        "gates": gates,
        "pass_to_reaction_specific_action_discovery": bool(all(gates.values())),
        "contracts": {
            "candidate_protocol": f"strict-{args.ppm:g}ppm same-adduct; molecule aggregated",
            "biological_outer_unit": "source; both LC modes and both polarities remain together",
            "seed_pool": "same-source Level-1 identities excluding truth identity and every same-formula seed",
            "seed_context_limit": (
                "source-level Level-1 union is an optimistic topology upper bound, not a per-sample "
                "deployable seed set"
            ),
            "identity_and_formula_purge": "both applied to every held-source training set",
            "Rhea_and_KEGG_use": "undirected topology audit only",
            "truth_advantage_fields": "headroom diagnostics only; forbidden as deployable input",
            "embeddings": "not read",
            "model_fit": "none",
            "P2b": "forbidden",
            "phenotype": "forbidden",
        },
        "provenance": {
            "manifest_report_sha256": sha256(args.manifest_root / "report.json"),
            "truth_file_sha256": truth_hashes,
            "reference_hdf5_sha256": sha256(args.reference_hdf5),
            "negative_library_manifest_sha256": sha256(args.negative_library_manifest),
            "negative_library_report_sha256": sha256(args.negative_library_report),
            "negative_library_integrity_report_sha256": sha256(
                args.negative_library_integrity_report
            ),
            "negative_approved_rows_sha256": sha256(args.negative_approved_rows),
            "rhea_relations_sha256": sha256(args.rhea_relations),
            "kegg_edges_sha256": sha256(args.kegg_edges),
            "script_sha256": sha256(Path(__file__)),
            "queries_sha256": sha256(queries_path),
            "candidate_identities_sha256": sha256(candidates_path),
            "source_overlap_sha256": sha256(overlap_path),
            "source_loso_sha256": sha256(loso_path),
        },
        "claim_limit": (
            "M0 establishes only candidate ambiguity, source-held-out sample size, and graph-topology "
            "opportunity. Source-level seeds overestimate per-sample opportunity, and unsupported adduct "
            "rows are unevaluable rather than zero-effect. Truth-vs-wrong support is oracle headroom. "
            "No reaction-specific ranking gain, "
            "embedding improvement, external confirmation, or biological mechanism is established."
        ),
    }
    write_json_atomic(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
