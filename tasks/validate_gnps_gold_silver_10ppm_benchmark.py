#!/usr/bin/env python
"""Independently validate the sealed GNPS Gold/Silver 10-ppm benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_gnps_gold_silver_10ppm_benchmark import (
    STATUS,
    expand_pair_ledger,
    iter_mgf_headers,
    load_exclusions,
    ppm_bounds,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument(
        "--massspecgym-hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--mona-mgf", type=Path, action="append",
        default=[ROOT / "data/models/mona_pos_full.mgf", ROOT / "data/models/mona_neg_full.mgf"],
    )
    return parser.parse_args()


def validate_checksums(root: Path) -> int:
    lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in lines:
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        actual = sha256_file(root / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {name}: {actual} != {expected}")
        checked += 1
    return checked


def validate_mgf(root: Path, frame: pd.DataFrame) -> int:
    count = 0
    for count, fields in enumerate(iter_mgf_headers(root / "spectra.mgf"), start=1):
        row = count - 1
        if row >= len(frame):
            raise RuntimeError("MGF has more rows than the manifest")
        expected = frame.iloc[row]
        if fields.get("INCHIKEY", "") != str(expected.inchikey):
            raise RuntimeError(f"MGF/manifest identity mismatch at row {row}")
        if fields.get("FILENAME", "") != str(expected.filename):
            raise RuntimeError(f"MGF/manifest filename mismatch at row {row}")
        if abs(float(fields["PEPMASS"]) - float(expected.precursor_mz)) > 1e-7:
            raise RuntimeError(f"MGF/manifest precursor mismatch at row {row}")
        if fields.get("ADDUCT") != "[M+H]+" or fields.get("IONMODE") != "positive":
            raise RuntimeError(f"non-[M+H]+ row in sealed MGF: {row}")
    if count != len(frame):
        raise RuntimeError(f"MGF/manifest row count mismatch: {count} != {len(frame)}")
    return count


def require_pointer(name: str, pointer: np.ndarray, terminal: int) -> None:
    if pointer.ndim != 1 or len(pointer) == 0 or int(pointer[0]) != 0:
        raise RuntimeError(f"{name} is not a zero-origin pointer")
    if np.any(np.diff(pointer) < 0) or int(pointer[-1]) != terminal:
        raise RuntimeError(f"{name} has invalid monotonicity or terminal")


def validate_panel(
    root: Path,
    frame: pd.DataFrame,
    name: str,
    excluded_identities: set[str],
    excluded_formulas: set[str],
    ppm: float,
) -> dict:
    panel_path = root / f"panel_{name}.npz"
    pairs_path = root / f"pairs_{name}.npz"
    with np.load(panel_path, allow_pickle=False) as body:
        panel = {key: np.asarray(body[key]) for key in body.files}
    with np.load(pairs_path, allow_pickle=False) as body:
        pairs = {key: np.asarray(body[key]) for key in body.files}
    n_queries = len(panel["query_row"])
    if not (
        len(panel["query_ik14"]) == len(panel["query_formula"])
        == len(panel["query_precursor_mz"]) == n_queries
    ):
        raise RuntimeError(f"{name} query arrays disagree")
    require_pointer("query_ptr", panel["query_ptr"], len(panel["molecule_ik14"]))
    require_pointer("molecule_ptr", panel["molecule_ptr"], len(panel["candidate_row"]))
    if np.any(panel["query_row"] < 0) or np.any(panel["query_row"] >= len(frame)):
        raise RuntimeError(f"{name} query row out of bounds")
    if np.any(panel["candidate_row"] < 0) or np.any(panel["candidate_row"] >= len(frame)):
        raise RuntimeError(f"{name} candidate row out of bounds")
    used = np.unique(np.concatenate([panel["query_row"], panel["candidate_row"]]))
    used_frame = frame.iloc[used]
    if used_frame.conflicting_spectrum_annotation.astype(bool).any():
        raise RuntimeError(f"{name} includes a cross-identity spectrum conflict")
    if set(used_frame.ik14.astype(str)) & excluded_identities:
        raise RuntimeError(f"{name} includes an excluded identity")
    if name == "formula_disjoint" and set(used_frame.formula.astype(str)) & excluded_formulas:
        raise RuntimeError("formula-disjoint panel includes an excluded formula")

    near_count = 0
    for q in range(n_queries):
        query_row = int(panel["query_row"][q])
        query = frame.iloc[query_row]
        if str(query.ik14) != str(panel["query_ik14"][q]):
            raise RuntimeError(f"{name} query IK mismatch at {q}")
        if str(query.formula) != str(panel["query_formula"][q]):
            raise RuntimeError(f"{name} query formula mismatch at {q}")
        mol_left, mol_right = map(int, panel["query_ptr"][q:q + 2])
        labels = panel["molecule_label"][mol_left:mol_right]
        if len(labels) < 2 or int(labels.sum()) != 1 or not bool(labels[0]):
            raise RuntimeError(f"{name} lacks unique-first positive at query {q}")
        observed_near = False
        for molecule in range(mol_left, mol_right):
            ref_left, ref_right = map(int, panel["molecule_ptr"][molecule:molecule + 2])
            rows = panel["candidate_row"][ref_left:ref_right]
            if not len(rows):
                raise RuntimeError(f"{name} empty reference group at molecule {molecule}")
            references = frame.iloc[rows]
            molecule_ik = str(panel["molecule_ik14"][molecule])
            molecule_formula = str(panel["molecule_formula"][molecule])
            if set(references.ik14.astype(str)) != {molecule_ik}:
                raise RuntimeError(f"{name} candidate identity mismatch at molecule {molecule}")
            if set(references.formula.astype(str)) != {molecule_formula}:
                raise RuntimeError(f"{name} candidate formula mismatch at molecule {molecule}")
            expected_label = molecule_ik == str(query.ik14)
            if bool(panel["molecule_label"][molecule]) != expected_label:
                raise RuntimeError(f"{name} incorrect IK14 label at molecule {molecule}")
            expected_same_formula = molecule_formula == str(query.formula)
            if bool(panel["molecule_same_formula"][molecule]) != expected_same_formula:
                raise RuntimeError(f"{name} incorrect same-formula label at molecule {molecule}")
            observed_near = observed_near or (expected_same_formula and not expected_label)
            low, high = ppm_bounds(float(query.precursor_mz), ppm)
            values = references.precursor_mz.to_numpy(float)
            if np.any(values < low - 1e-12) or np.any(values > high + 1e-12):
                raise RuntimeError(f"{name} reference outside {ppm:g}-ppm window at query {q}")
            if expected_label:
                if any(not str(value) for value in references.filename):
                    raise RuntimeError(f"{name} positive missing source filename at query {q}")
                if any(str(value) == str(query.filename) for value in references.filename):
                    raise RuntimeError(f"{name} positive reuses query source file at query {q}")
        if bool(panel["near_query"][q]) != observed_near:
            raise RuntimeError(f"{name} near-query label mismatch at query {q}")
        near_count += int(observed_near)

    expected_pairs = expand_pair_ledger(panel)
    if set(expected_pairs) != set(pairs):
        raise RuntimeError(f"{name} pair ledger fields disagree")
    for key in expected_pairs:
        if not np.array_equal(expected_pairs[key], pairs[key]):
            raise RuntimeError(f"{name} pair ledger mismatch: {key}")
    return {
        "queries": n_queries,
        "candidate_molecules": len(panel["molecule_ik14"]),
        "pairs": len(pairs["label"]),
        "positive_pairs": int(pairs["label"].sum()),
        "negative_pairs": int((~pairs["label"]).sum()),
        "near_queries": near_count,
        "used_spectra": len(used),
    }


def main() -> None:
    args = arguments()
    report = json.loads((args.benchmark / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != STATUS or report.get("formal") is not True:
        raise RuntimeError("benchmark report is not formally sealed")
    checksum_count = validate_checksums(args.benchmark)
    frame = pd.read_csv(args.benchmark / "manifest.csv.gz", low_memory=False).fillna("")
    if not np.array_equal(frame.row.to_numpy(np.int64), np.arange(len(frame), dtype=np.int64)):
        raise RuntimeError("manifest rows are not contiguous")
    if not frame.library_quality.isin([1, 2]).all():
        raise RuntimeError("manifest contains non-Gold/Silver spectra")
    if frame.duplicated(["ik14", "spectrum_hash"]).any():
        raise RuntimeError("manifest contains an exact same-identity spectrum duplicate")
    mgf_rows = validate_mgf(args.benchmark, frame)
    excluded_identities, excluded_formulas, _ = load_exclusions(
        args.massspecgym_hdf5, list(args.mona_mgf),
    )
    panels = {
        name: validate_panel(
            args.benchmark, frame, name, excluded_identities, excluded_formulas,
            float(report["parameters"]["ppm"]),
        )
        for name in ("identity_disjoint", "formula_disjoint")
    }
    output = {
        "status": "gnps_gold_silver_10ppm_benchmark_independent_validation_pass",
        "checksum_files": checksum_count,
        "manifest_rows": len(frame),
        "mgf_rows": mgf_rows,
        "excluded_identities_recomputed": len(excluded_identities),
        "excluded_formulas_recomputed": len(excluded_formulas),
        "panels": panels,
    }
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
