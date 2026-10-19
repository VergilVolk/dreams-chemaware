#!/usr/bin/env python
"""Build a score-blind MultiMS2 external retrieval benchmark.

The primary endpoint is positive-mode [M+H]+ CID 60 eV query retrieval
against a fixed CID 40 eV reference library in a complete 10-ppm mass
window.  EAD 16 eV -> CID 40 eV is emitted only as a prespecified OOD stress
panel.  Identity- and formula-disjoint panels are defined against every
required source in the consumed-source registry before any model is scored.

Observable panels deliberately omit query identities and labels.  Truth is
written to separate files for a later, one-time evaluation step.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_enveda180_scoreblind_manifest import (
    PROTON,
    clean,
    collision_energy,
    iter_mgf,
    load_csv,
    load_exclusion_registry,
    load_hdf5,
    load_mgf,
    secondary_spectrum_hash,
    sha256_file,
    spectrum_hash,
    structure,
)


SOURCE_COMMIT = "659bd9b608408a75ecbb4b03295ed4c03ba52539"
SOURCE_BLOB_SHA256 = "4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"
PRIMARY_ADDUCT = "[M+H]+"
PRIMARY_MODE = "positive"
PANEL_SPECS = {
    "cid_identity_disjoint": ("CID", 60.0, "identity"),
    "cid_formula_disjoint": ("CID", 60.0, "formula"),
    "ead_identity_disjoint_stress": ("EAD", 16.0, "identity"),
    "ead_formula_disjoint_stress": ("EAD", 16.0, "formula"),
}


def ppm_window(value: float, ppm: float) -> tuple[float, float]:
    delta = abs(value) * ppm * 1e-6
    return value - delta, value + delta


def load_consumed_registry(path: Path) -> tuple[set[str], set[str], dict[str, set[str]], list[dict]]:
    identities: set[str] = set()
    formulas: set[str] = set()
    hashes = {"primary_sha256": set(), "secondary_blake2b": set()}
    loaders = {"hdf5": load_hdf5, "csv": load_csv, "mgf": load_mgf}
    reports = []
    for source in load_exclusion_registry(path):
        name = str(source["name"])
        kind = str(source.get("kind", ""))
        source_path = Path(str(source.get("path", "")))
        required = bool(source.get("required", True))
        if kind not in loaders:
            raise RuntimeError(f"unsupported consumed-source kind: {name} -> {kind}")
        if required and not source_path.is_file():
            raise RuntimeError(f"required consumed source missing: {name} -> {source_path}")
        report = loaders[kind](source_path, identities, formulas, hashes)
        report.update({"name": name, "kind": kind, "required": required})
        reports.append(report)
    failed = [row["name"] for row in reports if row["required"] and row["status"] != "loaded"]
    if failed:
        raise RuntimeError(f"required consumed sources not loaded: {failed}")
    return identities, formulas, hashes, reports


def source_records(
    mgf: Path,
    consumed_identities: set[str],
    consumed_formulas: set[str],
    consumed_hashes: dict[str, set[str]],
    min_peaks: int,
    precursor_ppm: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    counters = Counter()
    first_hash_identity: dict[str, str] = {}
    conflicting_hashes: set[str] = set()
    seen_same_identity: set[tuple[str, str]] = set()

    for source_row, (fields, peaks) in enumerate(iter_mgf(mgf)):
        counters["raw_records"] += 1
        if len(peaks) < min_peaks:
            counters["too_few_peaks"] += 1
            continue
        raw_key = clean(fields.get("INCHIAUX", fields.get("INCHIKEY")))
        parsed = structure(clean(fields.get("SMILES")), raw_key)
        if parsed is None:
            counters["invalid_or_conflicting_structure"] += 1
            continue
        inchikey, smiles, formula, exact_mass, _ = parsed
        annotated_formula = clean(fields.get("FORMULA"))
        if annotated_formula and annotated_formula != formula:
            counters["formula_structure_mismatch"] += 1
            continue
        adduct = clean(fields.get("ADDUCT"))
        mode = clean(fields.get("IONMODE")).lower()
        if adduct != PRIMARY_ADDUCT or mode != PRIMARY_MODE:
            counters["outside_primary_ion_domain"] += 1
            continue
        method = clean(fields.get("FRAGMENTATION_METHOD")).upper()
        energy = collision_energy(fields.get("COLLISION_ENERGY", ""))
        if method not in {"CID", "EAD"} or energy is None:
            counters["unsupported_or_missing_fragmentation"] += 1
            continue
        try:
            precursor_mz = float(clean(fields.get("PRECURSOR_MZ", fields.get("PEPMASS"))).split()[0])
        except (ValueError, IndexError):
            counters["invalid_precursor"] += 1
            continue
        expected_mz = exact_mass + PROTON
        ppm_error = (precursor_mz - expected_mz) / expected_mz * 1e6
        if not math.isfinite(ppm_error) or abs(ppm_error) > precursor_ppm:
            counters["precursor_structure_mismatch"] += 1
            continue
        digest = spectrum_hash(precursor_mz, peaks)
        secondary = secondary_spectrum_hash(peaks)
        ik14 = inchikey[:14]
        if (ik14, digest) in seen_same_identity:
            counters["same_identity_exact_duplicate"] += 1
            continue
        seen_same_identity.add((ik14, digest))
        previous = first_hash_identity.setdefault(digest, ik14)
        if previous != ik14:
            conflicting_hashes.add(digest)
        rows.append({
            "source_row": source_row,
            "ik14": ik14,
            "inchikey": inchikey,
            "smiles": smiles,
            "formula": formula,
            "precursor_mz": precursor_mz,
            "precursor_structure_ppm": ppm_error,
            "adduct": adduct,
            "ion_mode": mode,
            "fragmentation_method": method,
            "collision_energy": energy,
            "n_peaks": len(peaks),
            "spectrum_hash": digest,
            "spectrum_hash_secondary": secondary,
            "consumed_identity_overlap": ik14 in consumed_identities,
            "consumed_formula_overlap": formula in consumed_formulas,
            "consumed_spectrum_overlap": (
                digest in consumed_hashes["primary_sha256"]
                or secondary in consumed_hashes["secondary_blake2b"]
            ),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("MultiMS2 produced no eligible [M+H]+ spectra")
    conflict_rows = set(frame.loc[frame.spectrum_hash.isin(conflicting_hashes), "source_row"].astype(int))
    frame["cross_identity_source_hash_conflict"] = frame.source_row.isin(conflict_rows)
    frame = frame[~frame.cross_identity_source_hash_conflict].copy()
    frame = frame.sort_values("source_row", kind="stable").reset_index(drop=True)
    frame.insert(0, "eligible_row", np.arange(len(frame), dtype=np.int64))
    counters["eligible_primary_domain_records"] = len(frame)
    counters["consumed_identity_overlap_records"] = int(frame.consumed_identity_overlap.sum())
    counters["consumed_formula_overlap_records"] = int(frame.consumed_formula_overlap.sum())
    counters["consumed_spectrum_overlap_records"] = int(frame.consumed_spectrum_overlap.sum())
    counters["cross_identity_source_hash_conflict_records"] = len(conflict_rows)
    return frame, dict(counters)


def choose_condition(frame: pd.DataFrame, method: str, energy: float) -> pd.DataFrame:
    selected = frame[
        (frame.fragmentation_method == method)
        & np.isclose(frame.collision_energy.astype(float), energy)
    ].copy()
    # Observable quality rule, fixed before scoring: most peaks, then source row.
    return (
        selected.sort_values(["ik14", "n_peaks", "source_row"], ascending=[True, False, True], kind="stable")
        .drop_duplicates("ik14", keep="first")
        .reset_index(drop=True)
    )


def build_panel(
    frame: pd.DataFrame,
    query_method: str,
    query_energy: float,
    disjointness: str,
    ppm: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    if disjointness == "identity":
        pool = frame[~frame.consumed_identity_overlap & ~frame.consumed_spectrum_overlap].copy()
    elif disjointness == "formula":
        pool = frame[~frame.consumed_formula_overlap & ~frame.consumed_spectrum_overlap].copy()
    else:
        raise ValueError(f"unsupported disjointness: {disjointness}")
    references = choose_condition(pool, "CID", 40.0)
    queries = choose_condition(pool, query_method, query_energy)
    queries = queries[queries.ik14.isin(set(references.ik14))].copy()
    references = references.sort_values(["precursor_mz", "ik14"], kind="stable").reset_index(drop=True)
    masses = references.precursor_mz.to_numpy(float)

    query_rows = []
    query_tokens = []
    query_mz = []
    candidate_rows = []
    candidate_ids = []
    candidate_formulas = []
    candidate_smiles = []
    candidate_mz = []
    query_ptr = [0]
    truth_ids = []
    truth_formulas = []
    labels = []
    near = []

    for row in queries.sort_values(["ik14", "source_row"], kind="stable").itertuples(index=False):
        low, high = ppm_window(float(row.precursor_mz), ppm)
        left = int(np.searchsorted(masses, low, side="left"))
        right = int(np.searchsorted(masses, high, side="right"))
        candidates = references.iloc[left:right].copy()
        if len(candidates) < 2 or int((candidates.ik14 == row.ik14).sum()) != 1:
            continue
        token = hashlib.sha256(
            f"MultiMS2|{query_method}|{query_energy:g}|{disjointness}|{int(row.source_row)}".encode()
        ).hexdigest()[:24]
        query_rows.append(int(row.source_row))
        query_tokens.append(token)
        query_mz.append(float(row.precursor_mz))
        truth_ids.append(str(row.ik14))
        truth_formulas.append(str(row.formula))
        local_labels = (candidates.ik14 == row.ik14).to_numpy(bool)
        labels.extend(local_labels.tolist())
        near.append(bool(np.any((candidates.formula == row.formula).to_numpy(bool) & ~local_labels)))
        candidate_rows.extend(candidates.source_row.astype(int).tolist())
        candidate_ids.extend(candidates.ik14.astype(str).tolist())
        candidate_formulas.extend(candidates.formula.astype(str).tolist())
        candidate_smiles.extend(candidates.smiles.astype(str).tolist())
        candidate_mz.extend(candidates.precursor_mz.astype(float).tolist())
        query_ptr.append(len(candidate_rows))

    observable = {
        "schema": np.asarray("multims2_scoreblind_observable_panel_v1"),
        "query_token": np.asarray(query_tokens, dtype=np.str_),
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "query_precursor_mz": np.asarray(query_mz, dtype=np.float64),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "candidate_row": np.asarray(candidate_rows, dtype=np.int64),
        "candidate_ik14": np.asarray(candidate_ids, dtype=np.str_),
        "candidate_formula": np.asarray(candidate_formulas, dtype=np.str_),
        "candidate_smiles": np.asarray(candidate_smiles, dtype=np.str_),
        "candidate_precursor_mz": np.asarray(candidate_mz, dtype=np.float64),
        "query_fragmentation_method": np.asarray(query_method),
        "query_collision_energy": np.asarray(query_energy, dtype=np.float64),
        "reference_fragmentation_method": np.asarray("CID"),
        "reference_collision_energy": np.asarray(40.0, dtype=np.float64),
        "disjointness": np.asarray(disjointness),
        "ppm": np.asarray(ppm, dtype=np.float64),
    }
    truth = {
        "schema": np.asarray("multims2_sealed_truth_v1"),
        "query_token": np.asarray(query_tokens, dtype=np.str_),
        "query_ik14": np.asarray(truth_ids, dtype=np.str_),
        "query_formula": np.asarray(truth_formulas, dtype=np.str_),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "candidate_label": np.asarray(labels, dtype=np.int8),
        "near_query": np.asarray(near, dtype=bool),
    }
    if any(name in observable for name in ("query_ik14", "candidate_label", "near_query")):
        raise RuntimeError("truth leaked into observable MultiMS2 panel")
    for left, right in zip(truth["query_ptr"][:-1], truth["query_ptr"][1:]):
        if int(truth["candidate_label"][left:right].sum()) != 1:
            raise RuntimeError("MultiMS2 query does not have exactly one positive")
    report = {
        "query_method": query_method,
        "query_energy": query_energy,
        "reference_method": "CID",
        "reference_energy": 40.0,
        "disjointness": disjointness,
        "eligible_pool_identities": int(pool.ik14.nunique()),
        "reference_identities": int(references.ik14.nunique()),
        "condition_paired_identities": int(queries.ik14.nunique()),
        "competitive_queries": len(query_rows),
        "candidate_identities": len(candidate_rows),
        "near_queries": int(np.sum(near)),
    }
    return observable, truth, report


def write_sanitized_mgf(
    source_mgf: Path,
    selected_source_rows: set[int],
    destination: Path,
) -> dict[int, int]:
    row_map = {}
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for source_row, (fields, peaks) in enumerate(iter_mgf(source_mgf)):
            if source_row not in selected_source_rows:
                continue
            output_row = len(row_map)
            row_map[source_row] = output_row
            precursor = clean(fields.get("PRECURSOR_MZ", fields.get("PEPMASS"))).split()[0]
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE=MULTIMS2_BLIND_{output_row}\n")
            handle.write(f"PEPMASS={precursor}\n")
            handle.write(f"IONMODE={PRIMARY_MODE}\n")
            handle.write(f"ADDUCT={PRIMARY_ADDUCT}\n")
            handle.write(f"FRAGMENTATION_METHOD={clean(fields.get('FRAGMENTATION_METHOD')).upper()}\n")
            handle.write(f"COLLISION_ENERGY={collision_energy(fields.get('COLLISION_ENERGY', ''))}\n")
            for mz, intensity in peaks:
                handle.write(f"{mz:.8f} {intensity:.8g}\n")
            handle.write("END IONS\n\n")
    if set(row_map) != selected_source_rows:
        missing = sorted(selected_source_rows - set(row_map))[:10]
        raise RuntimeError(f"selected MultiMS2 rows missing during sanitization: {missing}")
    return row_map


def remap_rows(panel: dict[str, np.ndarray], row_map: dict[int, int]) -> None:
    for field in ("query_row", "candidate_row"):
        panel[field] = np.asarray([row_map[int(value)] for value in panel[field]], dtype=np.int64)


def checksums(directory: Path) -> None:
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    with (directory / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mgf", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=SOURCE_BLOB_SHA256)
    parser.add_argument("--min-peaks", type=int, default=10)
    parser.add_argument("--precursor-ppm", type=float, default=30.0)
    parser.add_argument("--candidate-ppm", type=float, default=10.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite MultiMS2 benchmark: {args.output}")
    if not args.source_mgf.is_file():
        raise FileNotFoundError(args.source_mgf)
    actual_source_hash = sha256_file(args.source_mgf)
    if actual_source_hash != args.expected_source_sha256.lower():
        raise RuntimeError(
            f"MultiMS2 source hash mismatch: expected={args.expected_source_sha256} actual={actual_source_hash}"
        )
    identities, formulas, consumed_hashes, source_reports = load_consumed_registry(args.exclusion_registry)
    frame, source_counts = source_records(
        args.source_mgf, identities, formulas, consumed_hashes,
        args.min_peaks, args.precursor_ppm,
    )

    panels = {}
    truths = {}
    panel_reports = {}
    for name, (method, energy, disjointness) in PANEL_SPECS.items():
        panel, truth, report = build_panel(frame, method, energy, disjointness, args.candidate_ppm)
        panels[name], truths[name], panel_reports[name] = panel, truth, report
    if panel_reports["cid_identity_disjoint"]["competitive_queries"] == 0:
        raise RuntimeError("primary MultiMS2 identity panel is empty after full consumed-source exclusion")

    selected_rows = set()
    for panel in panels.values():
        selected_rows.update(map(int, panel["query_row"]))
        selected_rows.update(map(int, panel["candidate_row"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".multims2_external.", dir=args.output.parent))
    try:
        observable_dir = staging / "observable"
        sealed_dir = staging / "sealed_truth_do_not_open_during_scoring"
        observable_dir.mkdir()
        sealed_dir.mkdir()
        manifest = sealed_dir / "eligible_records.csv.gz"
        frame.to_csv(manifest, index=False, compression="gzip", quoting=csv.QUOTE_MINIMAL)
        row_map = write_sanitized_mgf(
            args.source_mgf, selected_rows, observable_dir / "spectra_scoreblind.mgf"
        )
        for name in PANEL_SPECS:
            remap_rows(panels[name], row_map)
            observable_path = observable_dir / f"panel_{name}.npz"
            truth_path = sealed_dir / f"panel_{name}_truth.npz"
            np.savez_compressed(observable_path, **panels[name])
            np.savez_compressed(truth_path, **truths[name])
            panel_reports[name].update({
                "observable_path": str(observable_path.relative_to(staging)),
                "observable_sha256": sha256_file(observable_path),
                "truth_path": str(truth_path.relative_to(staging)),
                "truth_sha256": sha256_file(truth_path),
            })
        report = {
            "status": "MULTIMS2_EXTERNAL_SCOREBLIND_BENCHMARK_COMPLETE",
            "schema": "multims2_external_scoreblind_benchmark_v1",
            "source": {
                "dataset": "MultiMS2 / MSV000099369",
                "repository": "https://github.com/zamboni-lab/MultiMS2",
                "commit": SOURCE_COMMIT,
                "mgf": str(args.source_mgf),
                "mgf_sha256": actual_source_hash,
            },
            "scientific_role": {
                "evaluation_role": "new_external_confirmation",
                "primary_endpoint": "CID 60 eV query -> CID 40 eV reference; positive [M+H]+; complete 10-ppm candidate window",
                "secondary_endpoint": "EAD 16 eV query -> CID 40 eV reference; prespecified OOD stress test",
                "selection_scores_opened": False,
                "model_loaded": False,
                "enveda_accessed": False,
            },
            "exclusion_registry": str(args.exclusion_registry),
            "exclusion_registry_sha256": sha256_file(args.exclusion_registry),
            "consumed_namespace": {
                "identities": len(identities),
                "formulas": len(formulas),
                "primary_spectrum_hashes": len(consumed_hashes["primary_sha256"]),
                "secondary_spectrum_hashes": len(consumed_hashes["secondary_blake2b"]),
                "sources": source_reports,
            },
            "filters": {
                "adduct": PRIMARY_ADDUCT,
                "ion_mode": PRIMARY_MODE,
                "min_peaks": args.min_peaks,
                "precursor_structure_ppm": args.precursor_ppm,
                "candidate_window_ppm": args.candidate_ppm,
                "representative_rule": "maximum peak count, then minimum source row; fixed before scoring",
            },
            "source_counts": source_counts,
            "eligible_spectra": len(frame),
            "eligible_identities": int(frame.ik14.nunique()),
            "scoreblind_spectra": len(row_map),
            "panels": panel_reports,
            "truth_separation": {
                "observable_contains_query_identity": False,
                "observable_contains_labels": False,
                "scoring_input_directory": "observable",
                "sealed_directory": "sealed_truth_do_not_open_during_scoring",
                "required_scoring_behavior": "score observable artifacts first; open sealed truth once only after all method outputs are complete",
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        checksums(staging)
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
