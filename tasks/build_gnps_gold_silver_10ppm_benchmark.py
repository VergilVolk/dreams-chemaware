#!/usr/bin/env python
"""Build a labelled GNPS Gold/Silver 10-ppm retrieval benchmark.

The benchmark is intentionally metadata-only during construction: no model
embedding, similarity or prediction is consulted.  It follows the mathematical
core of the DreaMS NIST20 Figure-4b protocol (same IK14 is positive; a different
IK14 in a 10-ppm precursor window is negative) while also materialising a
molecule-level retrieval graph.

This is *not* an exact NIST20 reproduction.  GNPS belongs to the same public
data ecosystem as DreaMS pretraining, so the report explicitly limits claims to
an identity-disjoint labelled GNPS transfer/stress benchmark.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
PROTON = 1.007276466621
STATUS = "gnps_gold_silver_10ppm_benchmark_v1_sealed"

RDLogger.DisableLog("rdApp.*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "n/a", "na", "none", "null"} else text


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip()


def iter_mgf(path: Path):
    fields: dict[str, str] | None = None
    peaks: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
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
                if len(parts) < 2:
                    continue
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0:
                    peaks.append((mz, intensity))


def iter_mgf_headers(path: Path):
    fields: dict[str, str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
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


@lru_cache(maxsize=None)
def structure_from_text(inchi: str, smiles: str) -> tuple[str, str, str, float, bool] | None:
    mol_inchi = Chem.MolFromInchi(inchi) if inchi else None
    mol_smiles = Chem.MolFromSmiles(smiles) if smiles else None
    if mol_inchi is None and mol_smiles is None:
        return None
    if mol_inchi is not None and mol_smiles is not None:
        if Chem.MolToInchiKey(mol_inchi)[:14] != Chem.MolToInchiKey(mol_smiles)[:14]:
            return None
    mol = mol_inchi if mol_inchi is not None else mol_smiles
    assert mol is not None
    inchikey = Chem.MolToInchiKey(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    exact_mass = float(Descriptors.ExactMolWt(mol))
    stereo_consistent = mol_inchi is not None and mol_smiles is not None and (
        Chem.MolToInchiKey(mol_inchi) == Chem.MolToInchiKey(mol_smiles)
    )
    return inchikey, canonical, formula, exact_mass, stereo_consistent


def structure_from_fields(fields: dict[str, str]):
    return structure_from_text(clean_text(fields.get("INCHI")), clean_text(fields.get("SMILES")))


def peak_hash(precursor_mz: float, peaks: list[tuple[float, float]]) -> str:
    ordered = sorted(peaks)
    maximum = max(intensity for _, intensity in ordered)
    digest = hashlib.sha256()
    digest.update(f"{precursor_mz:.5f}|".encode())
    for mz, intensity in ordered:
        digest.update(f"{mz:.4f}:{intensity / maximum:.6f};".encode())
    return digest.hexdigest()


def load_exclusions(hdf5_path: Path, mona_paths: list[Path]) -> tuple[set[str], set[str], dict]:
    identities: set[str] = set()
    formulas: set[str] = set()
    by_source: dict[str, dict[str, int]] = {}
    with h5py.File(hdf5_path, "r") as handle:
        hdf_identities = {
            decode_text(value)[:14] for value in handle["INCHIKEY"][:] if len(decode_text(value)) >= 14
        }
        hdf_formulas = {
            decode_text(value) for value in handle["FORMULA"][:] if decode_text(value)
        }
    identities.update(hdf_identities)
    formulas.update(hdf_formulas)
    by_source[str(hdf5_path)] = {"ik14": len(hdf_identities), "formulas": len(hdf_formulas)}

    for path in mona_paths:
        source_identities: set[str] = set()
        source_formulas: set[str] = set()
        for fields in iter_mgf_headers(path):
            raw_ik = clean_text(fields.get("INCHIKEY"))
            if len(raw_ik) >= 14:
                source_identities.add(raw_ik[:14])
            smiles = clean_text(fields.get("SMILES"))
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    source_identities.add(Chem.MolToInchiKey(mol)[:14])
                    source_formulas.add(rdMolDescriptors.CalcMolFormula(mol))
        identities.update(source_identities)
        formulas.update(source_formulas)
        by_source[str(path)] = {"ik14": len(source_identities), "formulas": len(source_formulas)}
    return identities, formulas, by_source


def write_mgf_record(handle, fields: dict[str, str], record: dict, peaks: list[tuple[float, float]]) -> None:
    handle.write("BEGIN IONS\n")
    handle.write(f"NAME={clean_text(fields.get('NAME'))}\n")
    handle.write(f"SMILES={record['smiles']}\n")
    handle.write(f"INCHIKEY={record['inchikey']}\n")
    handle.write(f"PEPMASS={record['precursor_mz']:.8f}\n")
    handle.write("ADDUCT=[M+H]+\n")
    handle.write("IONMODE=positive\n")
    handle.write(f"LIBRARYQUALITY={record['library_quality']}\n")
    handle.write(f"SPECTRUMID={record['spectrum_id']}\n")
    handle.write(f"USI={record['usi']}\n")
    handle.write(f"FILENAME={record['filename']}\n")
    handle.write(f"SOURCE_ROW={record['source_row']}\n")
    for mz, intensity in sorted(peaks):
        handle.write(f"{mz:.6f} {intensity:.6f}\n")
    handle.write("END IONS\n")


def scan_gnps(
    path: Path,
    output_mgf: Path,
    excluded_identities: set[str],
    excluded_formulas: set[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen_identity_hash: set[tuple[str, str]] = set()
    hash_identities: dict[str, set[str]] = defaultdict(set)
    counters: dict[str, int] = defaultdict(int)
    output_mgf.parent.mkdir(parents=True, exist_ok=True)
    with output_mgf.open("w", encoding="utf-8", newline="\n") as writer:
        for source_row, (fields, peaks) in enumerate(iter_mgf(path)):
            counters["raw_records"] += 1
            if source_row and source_row % args.progress_every == 0:
                print(
                    f"[GNPS benchmark] {source_row:,} raw; {len(rows):,} accepted; "
                    f"{counters['identity_overlap']:,} identity-overlap",
                    flush=True,
                )
            try:
                quality = int(clean_text(fields.get("LIBRARYQUALITY")) or "0")
            except ValueError:
                quality = 0
            if quality not in (1, 2):
                counters["quality_rejected"] += 1
                continue
            ionmode = clean_text(fields.get("IONMODE")).lower()
            if "pos" not in ionmode:
                counters["non_positive"] += 1
                continue
            mslevel = clean_text(fields.get("MSLEVEL"))
            if mslevel and mslevel not in {"2", "2.0"}:
                counters["non_ms2"] += 1
                continue
            if len(peaks) < args.min_peaks:
                counters["too_few_peaks"] += 1
                continue
            try:
                precursor_mz = float(clean_text(fields.get("PEPMASS")).split()[0])
            except (ValueError, IndexError):
                counters["invalid_precursor"] += 1
                continue
            if not math.isfinite(precursor_mz) or precursor_mz <= 0:
                counters["invalid_precursor"] += 1
                continue
            structure = structure_from_fields(fields)
            if structure is None:
                counters["invalid_or_conflicting_structure"] += 1
                continue
            inchikey, smiles, formula, exact_mass, stereo_consistent = structure
            ik14 = inchikey[:14]
            expected = exact_mass + PROTON
            error_da = abs(precursor_mz - expected)
            error_ppm = error_da / max(expected, 1e-12) * 1e6
            if error_da > max(args.precursor_abs_da, expected * args.precursor_validation_ppm * 1e-6):
                counters["not_validated_m_plus_h"] += 1
                continue
            if ik14 in excluded_identities:
                counters["identity_overlap"] += 1
                continue
            digest = peak_hash(precursor_mz, peaks)
            if (ik14, digest) in seen_identity_hash:
                counters["same_identity_exact_duplicates"] += 1
                continue
            seen_identity_hash.add((ik14, digest))
            hash_identities[digest].add(ik14)
            record = {
                "row": len(rows),
                "source_row": source_row,
                "inchikey": inchikey,
                "ik14": ik14,
                "smiles": smiles,
                "formula": formula,
                "exact_mass": exact_mass,
                "precursor_mz": precursor_mz,
                "precursor_structure_ppm": error_ppm,
                "library_quality": quality,
                "quality_label": "gold" if quality == 1 else "silver",
                "n_peaks": len(peaks),
                "spectrum_hash": digest,
                "formula_disjoint": formula not in excluded_formulas,
                "stereo_cross_field_consistent": stereo_consistent,
                "filename": clean_text(fields.get("FILENAME")),
                "spectrum_id": clean_text(fields.get("SPECTRUMID")),
                "usi": clean_text(fields.get("USI")),
                "instrument": clean_text(fields.get("SOURCE_INSTRUMENT")),
            }
            write_mgf_record(writer, fields, record, peaks)
            rows.append(record)
    conflicting_hashes = {digest for digest, values in hash_identities.items() if len(values) > 1}
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("GNPS filters produced no spectra")
    frame["conflicting_spectrum_annotation"] = frame.spectrum_hash.isin(conflicting_hashes)
    counters["cross_identity_conflicting_hashes"] = len(conflicting_hashes)
    counters["accepted_spectra"] = len(frame)
    counters["accepted_identities"] = int(frame.ik14.nunique())
    counters["accepted_formulas"] = int(frame.formula.nunique())
    return frame, dict(counters)


def ppm_bounds(center: float, ppm: float) -> tuple[float, float]:
    delta = max(abs(center), 1.0) * ppm * 1e-6
    return center - delta, center + delta


def build_panel(frame: pd.DataFrame, args: argparse.Namespace, label: str) -> tuple[dict[str, np.ndarray], dict]:
    work = frame[~frame.conflicting_spectrum_annotation].copy()
    if label == "formula_disjoint":
        work = work[work.formula_disjoint].copy()
    work = work.sort_values(["precursor_mz", "ik14", "row"]).reset_index(drop=True)
    mz = work.precursor_mz.to_numpy(float)
    global_rows = work.row.to_numpy(np.int64)
    ik14 = work.ik14.astype(str).to_numpy()
    formula = work.formula.astype(str).to_numpy()
    filename = work.filename.astype(str).to_numpy()
    quality = work.library_quality.to_numpy(np.int16)
    n_peaks = work.n_peaks.to_numpy(np.int32)

    by_identity: dict[str, np.ndarray] = {
        str(identity): group.index.to_numpy(np.int64)
        for identity, group in work.groupby("ik14", sort=True)
        if len(group) >= 2
    }
    query_row: list[int] = []
    query_ik14: list[str] = []
    query_formula: list[str] = []
    query_precursor_mz: list[float] = []
    query_ptr = [0]
    molecule_ik14: list[str] = []
    molecule_formula: list[str] = []
    molecule_label: list[bool] = []
    molecule_same_formula: list[bool] = []
    molecule_ptr = [0]
    candidate_row: list[int] = []
    independent_positive: list[bool] = []
    near_query: list[bool] = []
    queries_per_identity: dict[str, int] = defaultdict(int)

    for identity in sorted(by_identity):
        candidates_for_query = sorted(
            by_identity[identity],
            key=lambda i: (quality[i], -n_peaks[i], int(global_rows[i])),
        )
        for query_local in candidates_for_query:
            if queries_per_identity[identity] >= args.queries_per_identity:
                break
            center = float(mz[query_local])
            low, high = ppm_bounds(center, args.ppm)
            left = int(np.searchsorted(mz, low, side="left"))
            right = int(np.searchsorted(mz, high, side="right"))
            window = np.arange(left, right, dtype=np.int64)
            window = window[window != query_local]
            if not len(window):
                continue
            grouped: dict[str, list[int]] = defaultdict(list)
            for local in window:
                grouped[str(ik14[local])].append(int(local))
            positives = grouped.pop(identity, [])
            if not positives or not grouped:
                continue
            distinct_file = [
                i for i in positives
                if filename[query_local] and filename[i] and filename[i] != filename[query_local]
            ]
            positive_pool = distinct_file if distinct_file else positives
            if args.require_independent_positive and not distinct_file:
                continue
            negative_groups = sorted(grouped.items())
            if len(negative_groups) < args.min_negative_identities:
                continue
            if len(negative_groups) > args.max_negative_identities:
                negative_groups = sorted(
                    negative_groups,
                    key=lambda item: hashlib.sha256(
                        f"{args.seed}|{identity}|{int(global_rows[query_local])}|{item[0]}".encode()
                    ).hexdigest(),
                )[: args.max_negative_identities]
                negative_groups.sort()
            groups = [(identity, positive_pool)] + negative_groups
            query_row.append(int(global_rows[query_local]))
            query_ik14.append(identity)
            query_formula.append(str(formula[query_local]))
            query_precursor_mz.append(center)
            independent_positive.append(bool(distinct_file))
            near = False
            for candidate_identity, local_rows in groups:
                ranked = sorted(
                    local_rows,
                    key=lambda i: (abs(float(mz[i]) - center), quality[i], -n_peaks[i], int(global_rows[i])),
                )[: args.references_per_identity]
                candidate_formula = str(formula[ranked[0]])
                is_positive = candidate_identity == identity
                is_same_formula = candidate_formula == str(formula[query_local])
                near = near or (is_same_formula and not is_positive)
                molecule_ik14.append(candidate_identity)
                molecule_formula.append(candidate_formula)
                molecule_label.append(is_positive)
                molecule_same_formula.append(is_same_formula)
                candidate_row.extend(int(global_rows[i]) for i in ranked)
                molecule_ptr.append(len(candidate_row))
            query_ptr.append(len(molecule_ik14))
            near_query.append(near)
            queries_per_identity[identity] += 1

    arrays = {
        "query_row": np.asarray(query_row, dtype=np.int64),
        "query_ik14": np.asarray(query_ik14, dtype="U14"),
        "query_formula": np.asarray(query_formula, dtype="U96"),
        "query_precursor_mz": np.asarray(query_precursor_mz, dtype=np.float64),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_ik14": np.asarray(molecule_ik14, dtype="U14"),
        "molecule_formula": np.asarray(molecule_formula, dtype="U96"),
        "molecule_label": np.asarray(molecule_label, dtype=bool),
        "molecule_same_formula": np.asarray(molecule_same_formula, dtype=bool),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "candidate_row": np.asarray(candidate_row, dtype=np.int64),
        "independent_positive": np.asarray(independent_positive, dtype=bool),
        "near_query": np.asarray(near_query, dtype=bool),
    }
    if not len(query_row):
        raise RuntimeError(f"{label} graph produced no queries")
    for query_index in range(len(query_row)):
        left, right = arrays["query_ptr"][query_index:query_index + 2]
        labels = arrays["molecule_label"][left:right]
        if len(labels) < 2 or int(labels.sum()) != 1 or not bool(labels[0]):
            raise RuntimeError(f"{label} query {query_index} lacks unique-first positive")
    candidate_counts = np.diff(arrays["query_ptr"])
    reference_counts = np.diff(arrays["molecule_ptr"])
    positive_references = int(sum(
        arrays["molecule_ptr"][i + 1] - arrays["molecule_ptr"][i]
        for i, value in enumerate(arrays["molecule_label"]) if value
    ))
    negative_references = len(candidate_row) - positive_references
    stats = {
        "queries": len(query_row),
        "query_identities": len(set(query_ik14)),
        "query_formulas": len(set(query_formula)),
        "candidate_identities": len(set(molecule_ik14)),
        "candidate_molecules": len(molecule_ik14),
        "candidate_spectra": len(candidate_row),
        "positive_pairs": positive_references,
        "negative_pairs": negative_references,
        "pooled_pairs": positive_references + negative_references,
        "independent_positive_fraction": float(np.mean(independent_positive)),
        "near_queries": int(np.sum(near_query)),
        "near_query_fraction": float(np.mean(near_query)),
        "candidate_identities_per_query": {
            "median": float(np.median(candidate_counts)),
            "p90": float(np.quantile(candidate_counts, 0.9)),
            "maximum": int(candidate_counts.max()),
        },
        "references_per_identity": {
            "median": float(np.median(reference_counts)),
            "p90": float(np.quantile(reference_counts, 0.9)),
            "maximum": int(reference_counts.max()),
        },
    }
    return arrays, stats


def expand_pair_ledger(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    query_index: list[int] = []
    reference_row: list[int] = []
    label: list[bool] = []
    same_formula: list[bool] = []
    for q in range(len(panel["query_row"])):
        mol_left, mol_right = panel["query_ptr"][q:q + 2]
        for molecule in range(int(mol_left), int(mol_right)):
            ref_left, ref_right = panel["molecule_ptr"][molecule:molecule + 2]
            count = int(ref_right - ref_left)
            query_index.extend([q] * count)
            reference_row.extend(map(int, panel["candidate_row"][ref_left:ref_right]))
            label.extend([bool(panel["molecule_label"][molecule])] * count)
            same_formula.extend([bool(panel["molecule_same_formula"][molecule])] * count)
    return {
        "query_index": np.asarray(query_index, dtype=np.int64),
        "reference_row": np.asarray(reference_row, dtype=np.int64),
        "label": np.asarray(label, dtype=bool),
        "same_formula": np.asarray(same_formula, dtype=bool),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnps", type=Path, default=ROOT / "data/reference/gnps/ALL_GNPS.mgf")
    parser.add_argument(
        "--massspecgym-hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--mona-mgf", type=Path, action="append",
        default=[ROOT / "data/models/mona_pos_full.mgf", ROOT / "data/models/mona_neg_full.mgf"],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--precursor-validation-ppm", type=float, default=30.0)
    parser.add_argument("--precursor-abs-da", type=float, default=0.01)
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--queries-per-identity", type=int, default=2)
    parser.add_argument("--references-per-identity", type=int, default=3)
    parser.add_argument("--min-negative-identities", type=int, default=1)
    parser.add_argument("--max-negative-identities", type=int, default=200)
    parser.add_argument("--require-independent-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-queries", type=int, default=5000)
    parser.add_argument("--minimum-pairs", type=int, default=100000)
    parser.add_argument("--minimum-formula-queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict:
    required = [args.gnps, args.massspecgym_hdf5, *args.mona_mgf]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite benchmark: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        excluded_identities, excluded_formulas, exclusion_sources = load_exclusions(
            args.massspecgym_hdf5, list(args.mona_mgf),
        )
        spectra_path = staging / "spectra.mgf"
        frame, scan_stats = scan_gnps(
            args.gnps, spectra_path, excluded_identities, excluded_formulas, args,
        )
        manifest_path = staging / "manifest.csv.gz"
        frame.to_csv(manifest_path, index=False, compression="gzip")

        identity_panel, identity_stats = build_panel(frame, args, "identity_disjoint")
        formula_panel, formula_stats = build_panel(frame, args, "formula_disjoint")
        identity_panel_path = staging / "panel_identity_disjoint.npz"
        formula_panel_path = staging / "panel_formula_disjoint.npz"
        np.savez_compressed(identity_panel_path, **identity_panel)
        np.savez_compressed(formula_panel_path, **formula_panel)
        identity_pairs_path = staging / "pairs_identity_disjoint.npz"
        formula_pairs_path = staging / "pairs_formula_disjoint.npz"
        np.savez_compressed(identity_pairs_path, **expand_pair_ledger(identity_panel))
        np.savez_compressed(formula_pairs_path, **expand_pair_ledger(formula_panel))

        used_rows = set(map(int, identity_panel["query_row"])) | set(map(int, identity_panel["candidate_row"]))
        identity_overlap = set(frame.loc[frame.row.isin(used_rows), "ik14"]) & excluded_identities
        formula_rows = set(map(int, formula_panel["query_row"])) | set(map(int, formula_panel["candidate_row"]))
        formula_overlap = set(frame.loc[frame.row.isin(formula_rows), "formula"]) & excluded_formulas
        gates = {
            "construction_uses_no_model_scores": True,
            "only_gold_silver": bool(frame.library_quality.isin([1, 2]).all()),
            "only_validated_m_plus_h_positive_ms2": True,
            "no_exact_duplicate_within_identity": bool(~frame.duplicated(["ik14", "spectrum_hash"]).any()),
            "conflicting_cross_identity_spectra_excluded_from_panels": bool(
                not frame.loc[frame.row.isin(used_rows), "conflicting_spectrum_annotation"].any()
            ),
            "massspecgym_and_mona_identity_overlap_is_zero": len(identity_overlap) == 0,
            "formula_disjoint_panel_overlap_is_zero": len(formula_overlap) == 0,
            "identity_panel_has_unique_first_positive": True,
            "formula_panel_has_unique_first_positive": True,
            "independent_acquisition_positive_required": bool(args.require_independent_positive),
            "identity_query_scale": identity_stats["queries"] >= args.minimum_queries,
            "identity_pair_scale": identity_stats["pooled_pairs"] >= args.minimum_pairs,
            "formula_query_scale": formula_stats["queries"] >= args.minimum_formula_queries,
        }
        report = {
            "status": STATUS if all(gates.values()) else "gnps_gold_silver_10ppm_benchmark_v1_gate_failed",
            "formal": bool(all(gates.values())),
            "exact_nist20_replication": False,
            "protocol": {
                "positive": "same first 14 InChIKey characters (IK14)",
                "negative": f"different IK14 within {args.ppm:g} ppm observed precursor m/z",
                "ion_adduct": "positive [M+H]+ only, inferred from structure exact mass",
                "quality": "GNPS LibraryQuality 1 (Gold) or 2 (Silver)",
                "query_positive_independence": "different source FILENAME required",
                "retrieval_scoring": "one positive molecule first; aggregate replicate references per molecule; ties count against positive",
                "near_subset": "negative molecule has the same molecular formula as the query",
                "construction_model_blind": True,
            },
            "scan": scan_stats,
            "identity_disjoint": identity_stats,
            "formula_disjoint": formula_stats,
            "exclusions": {
                "excluded_ik14": len(excluded_identities),
                "excluded_formulas": len(excluded_formulas),
                "sources": exclusion_sources,
                "identity_overlap_in_published_panel": len(identity_overlap),
                "formula_overlap_in_strict_panel": len(formula_overlap),
            },
            "gates": gates,
            "claim_limit": (
                "Large labelled GNPS Gold/Silver identity-disjoint transfer and robustness benchmark. "
                "It uses the NIST20 pairwise label/window mathematics but is not NIST20 and does not "
                "prove pretraining-corpus novelty because GNPS and DreaMS pretraining share a public-data ecosystem."
            ),
            "required_metrics": [
                "Recall@1/2/3/5/10/20", "MRR", "mean_rank", "median_rank",
                "macro_query_AUROC/AUPRC", "micro_candidate_AUROC/AUPRC",
                "pooled_pairwise_AUROC/AUPRC", "positive_vs_best_negative_margin",
                "Top1-Top2_gap", "corrected/introduced/risk_net",
                "near_subset_metrics", "formula_cluster_paired_CI",
            ],
            "parameters": {
                key: ([str(item) for item in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "provenance": {
                "gnps_sha256": sha256_file(args.gnps),
                "massspecgym_hdf5_sha256": sha256_file(args.massspecgym_hdf5),
                "mona_mgf_sha256": {str(path): sha256_file(path) for path in args.mona_mgf},
                "spectra_sha256": sha256_file(spectra_path),
                "manifest_sha256": sha256_file(manifest_path),
                "panel_identity_sha256": sha256_file(identity_panel_path),
                "panel_formula_sha256": sha256_file(formula_panel_path),
                "pairs_identity_sha256": sha256_file(identity_pairs_path),
                "pairs_formula_sha256": sha256_file(formula_pairs_path),
                "builder_sha256": sha256_file(Path(__file__)),
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (staging / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as handle:
            for name in (
                "spectra.mgf", "manifest.csv.gz", "panel_identity_disjoint.npz",
                "panel_formula_disjoint.npz", "pairs_identity_disjoint.npz",
                "pairs_formula_disjoint.npz", "report.json",
            ):
                handle.write(f"{sha256_file(staging / name)}  {name}\n")
        if not all(gates.values()):
            raise RuntimeError(f"GNPS benchmark gates failed: {gates}")
        staging.replace(args.output_dir)
        return report
    except Exception:
        failed = args.output_dir.parent / f"{args.output_dir.name}.failed"
        if failed.exists():
            shutil.rmtree(failed)
        staging.replace(failed)
        raise


def main() -> None:
    report = build(arguments())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
