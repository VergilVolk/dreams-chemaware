"""Reconstruct the missing GNPS identity-disjoint panel file, certified by
reproducing the existing formula-disjoint panel byte-for-byte first.

The frozen evaluator needs benchmark/panel_<name>.npz with molecule-level
fields.  Locally only panel_formula_disjoint.npz exists.  We reconstruct the
panel from (a) pairs_<panel>.npz (query_index, reference_row, label,
same_formula), (b) manifest.csv.gz (row -> ik14/formula), and (c) the frozen
per-query table queries_<panel>_official_dreams.csv.gz (query_row/ik14/
formula/near).  The exact same algorithm is first run on the formula panel
and asserted field-by-field against the real file; only a perfect
reproduction authorizes writing the identity panel.

Usage:
  python GLM_reconstruct_gnps_identity_panel.py --benchmark <dir> --run <2349091 dir> --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_panel(pairs: dict, manifest: pd.DataFrame, queries: pd.DataFrame):
    query_index = pairs["query_index"].astype(np.int64)
    reference_row = pairs["reference_row"].astype(np.int64)
    label = pairs["label"].astype(np.int8)
    same_formula = pairs["same_formula"].astype(bool)

    ik14_by_row = manifest["ik14"].astype(str).to_numpy()
    formula_by_row = manifest["formula"].astype(str).to_numpy()
    filename_by_row = manifest["filename"].astype(str).to_numpy()
    precursor_by_row = manifest["precursor_mz"].to_numpy(np.float64)

    q = queries.sort_values("query_index", kind="stable").reset_index(drop=True)
    n_queries = len(q)
    assert np.array_equal(q["query_index"].to_numpy(np.int64),
                          np.arange(n_queries))

    query_row = q["query_row"].to_numpy(np.int64)
    query_ik14 = q["query_ik14"].astype(str).to_numpy()
    query_formula = q["query_formula"].astype(str).to_numpy()
    near_query = q["near"].astype(bool).to_numpy()

    query_ptr = np.zeros(n_queries + 1, dtype=np.int64)
    molecule_ptr = [0]
    molecule_rows: list[int] = []
    molecule_label: list[int] = []
    molecule_ik14: list[str] = []
    molecule_formula: list[str] = []
    molecule_same_formula: list[bool] = []
    candidate_row: list[int] = []

    order = np.argsort(query_index, kind="stable")
    boundaries = np.searchsorted(query_index[order], np.arange(n_queries + 1))
    independent_positive = np.zeros(n_queries, dtype=bool)
    for query in range(n_queries):
        local = order[boundaries[query]:boundaries[query + 1]]
        seen: dict[str, int] = {}
        query_file = filename_by_row[int(query_row[query])]
        positive_files: set[str] = set()
        for pair in local:
            row = int(reference_row[pair])
            key = ik14_by_row[row]
            if int(label[pair]):
                positive_files.add(filename_by_row[row])
            if key not in seen:
                seen[key] = len(molecule_rows)
                molecule_rows.append(query)
                molecule_label.append(int(label[pair]))
                molecule_ik14.append(key)
                molecule_formula.append(formula_by_row[row])
                molecule_same_formula.append(bool(same_formula[pair]))
                molecule_ptr.append(molecule_ptr[-1])
            candidate_row.append(row)
            molecule_ptr[-1] += 1
        independent_positive[query] = len(positive_files - {query_file}) > 0
        query_ptr[query + 1] = len(molecule_rows)
    return {
        "query_row": query_row,
        "query_ik14": query_ik14,
        "query_formula": query_formula,
        "query_precursor_mz": precursor_by_row[query_row],
        "near_query": near_query,
        "independent_positive": independent_positive,
        "query_ptr": query_ptr,
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "molecule_label": np.asarray(molecule_label, dtype=np.int8),
        "molecule_ik14": np.asarray(molecule_ik14, dtype=object).astype(str),
        "molecule_formula": np.asarray(molecule_formula, dtype=object).astype(str),
        "molecule_same_formula": np.asarray(molecule_same_formula, dtype=bool),
        "candidate_row": np.asarray(candidate_row, dtype=np.int64),
    }


def compare(rebuilt: dict, real_path: Path, panel: str) -> list[str]:
    mismatched = []
    with np.load(real_path, allow_pickle=False) as body:
        real = {key: np.asarray(body[key]) for key in body.files}
    if set(real) != set(rebuilt):
        mismatched.append(f"field set differs: {set(real) ^ set(rebuilt)}")
        return mismatched
    for key, value in rebuilt.items():
        other = real[key]
        if value.dtype.kind != other.dtype.kind:
            # only complain if values differ after cast
            try:
                same = np.array_equal(value.astype(other.dtype), other)
            except Exception:
                same = False
        else:
            same = np.array_equal(value, other)
        if not same:
            if value.shape != other.shape:
                mismatched.append(f"{panel}/{key}: shape {value.shape} vs {other.shape}")
            else:
                bad = np.flatnonzero(~(value == other))[:5] if value.shape else []
                mismatched.append(f"{panel}/{key}: differs at {bad.tolist()}")
    return mismatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz",
                           usecols=["row", "ik14", "formula", "filename",
                                    "precursor_mz"])
    if not np.array_equal(manifest["row"].to_numpy(np.int64),
                          np.arange(len(manifest))):
        raise RuntimeError("manifest row registry is not contiguous")

    # certification on the formula panel
    with np.load(args.benchmark / "pairs_formula_disjoint.npz") as body:
        pairs_formula = {key: np.asarray(body[key]) for key in body.files}
    queries_formula = pd.read_csv(
        args.run / "evaluation/queries_formula_disjoint_official_dreams.csv.gz",
        low_memory=False)
    rebuilt = build_panel(pairs_formula, manifest, queries_formula)
    mismatched = compare(rebuilt, args.benchmark / "panel_formula_disjoint.npz",
                         "formula_disjoint")
    if mismatched:
        print("RECONSTRUCTION CERTIFICATION FAILED — identity panel NOT written:",
              file=__import__("sys").stderr)
        for item in mismatched:
            print("  ", item, file=__import__("sys").stderr)
        raise SystemExit(1)

    with np.load(args.benchmark / "pairs_identity_disjoint.npz") as body:
        pairs_identity = {key: np.asarray(body[key]) for key in body.files}
    queries_identity = pd.read_csv(
        args.run / "evaluation/queries_identity_disjoint_official_dreams.csv.gz",
        low_memory=False)
    identity = build_panel(pairs_identity, manifest, queries_identity)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "panel_identity_disjoint.npz", **identity)
    np.savez_compressed(args.output / "panel_formula_disjoint.npz", **rebuilt)
    report = {
        "status": "GLM_GNPS_IDENTITY_PANEL_RECONSTRUCTED_CERTIFIED",
        "certification": "the same algorithm reproduces the frozen "
                         "panel_formula_disjoint.npz field-by-field",
        "identity_queries": int(len(identity["query_row"])),
        "identity_molecules": int(len(identity["molecule_label"])),
        "identity_candidate_rows": int(len(identity["candidate_row"])),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=True),
                                             encoding="utf-8")
    print(json.dumps(report, indent=True))


if __name__ == "__main__":
    main()
