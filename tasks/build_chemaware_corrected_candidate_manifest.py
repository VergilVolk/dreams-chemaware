"""Build the corrected, metadata-only ChemAware candidate manifest.

The input E1 pools already encode P3-disjoint, same-adduct, strict query-centred
10 ppm positive and negative spectrum candidates with tolerance-rounded
same-spectrum pairs excluded.  This builder groups candidate spectra by IK14
for every eligible anchor.  It does not compute DreaMS embeddings, raw spectral
scores, teacher labels, or model features, and therefore does not authorize or
start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOLS = (
    ROOT / "data/e1/chemaware_control_train_mh_triplet_pool_10ppm_p3disjoint_v3.npz",
    ROOT / "data/e1/chemaware_control_train_mna_triplet_pool_10ppm_p3disjoint_v3.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointer_counts(pointer: np.ndarray) -> np.ndarray:
    if pointer.ndim != 1 or len(pointer) < 2 or pointer[0] != 0 or np.any(np.diff(pointer) < 0):
        raise RuntimeError("invalid candidate pointer")
    return np.diff(pointer)


def decode(dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--pools", nargs="+", type=Path, default=list(DEFAULT_POOLS))
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    required = [args.data]
    for pool in args.pools:
        required.extend((pool, pool.with_suffix(".json")))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    with h5py.File(args.data, "r") as handle:
        total_rows = len(handle["INCHIKEY"])
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"])], dtype=str)
        formula = decode(handle["FORMULA"])
        adduct = decode(handle["adduct"])
        fold = decode(handle["fold"])
        membership = decode(handle["SIMULATION_CHALLENGE"])

    query_rows: list[int] = []
    query_ik14: list[str] = []
    query_formula: list[str] = []
    query_adduct: list[str] = []
    query_membership: list[str] = []
    query_ptr = [0]
    molecule_ptr = [0]
    molecule_query: list[int] = []
    molecule_label: list[int] = []
    molecule_ik14: list[str] = []
    molecule_formula: list[str] = []
    pair_candidate_row: list[int] = []
    pool_audit = []
    seen_queries: set[int] = set()

    for pool_path in args.pools:
        report_file = pool_path.with_suffix(".json")
        report = json.loads(report_file.read_text(encoding="utf-8"))
        if report.get("simulation_challenge_semantics") != (
            "spectrum-simulation benchmark subset membership; never used as provenance or a filter"
        ):
            raise RuntimeError(f"pool has obsolete membership semantics: {pool_path}")
        if report.get("window_protocol") != "query_centred_ppm_positive_and_negative":
            raise RuntimeError(f"pool is not query-centred ppm: {pool_path}")
        if float(report.get("mass_window_ppm", -1)) != 10.0:
            raise RuntimeError(f"pool is not strict 10 ppm: {pool_path}")
        if report.get("allow_list", {}).get("schema") != "train_primary_all.rows":
            raise RuntimeError(f"pool does not use corrected allow-list: {pool_path}")
        if int(report.get("allow_list", {}).get("p3_query_overlap", -1)) != 0:
            raise RuntimeError(f"pool does not prove P3 disjointness: {pool_path}")
        if report.get("candidate_caps") != {"positive": 0, "negative": 0}:
            raise RuntimeError(f"pool candidate lists were capped: {pool_path}")

        with np.load(pool_path) as pool:
            anchors = pool["anchor_idx"].astype(np.int64)
            positive_ptr = pool["positive_ptr"].astype(np.int64)
            positive_idx = pool["positive_idx"].astype(np.int64)
            negative_ptr = pool["negative_ptr"].astype(np.int64)
            negative_idx = pool["negative_idx"].astype(np.int64)
        if len(anchors) != len(pointer_counts(positive_ptr)) or len(anchors) != len(pointer_counts(negative_ptr)):
            raise RuntimeError(f"pool pointer/anchor mismatch: {pool_path}")
        if positive_ptr[-1] != len(positive_idx) or negative_ptr[-1] != len(negative_idx):
            raise RuntimeError(f"pool pointers do not span candidates: {pool_path}")
        all_pool_rows = np.concatenate((anchors, positive_idx, negative_idx))
        if np.any((all_pool_rows < 0) | (all_pool_rows >= total_rows)):
            raise RuntimeError(f"pool contains out-of-range rows: {pool_path}")

        for local_query, anchor in enumerate(anchors):
            anchor = int(anchor)
            if anchor in seen_queries:
                raise RuntimeError(f"query row appears in multiple pools: {anchor}")
            seen_queries.add(anchor)
            p_left, p_right = map(int, positive_ptr[local_query:local_query + 2])
            n_left, n_right = map(int, negative_ptr[local_query:local_query + 2])
            positive = np.unique(positive_idx[p_left:p_right])
            negative = negative_idx[n_left:n_right]
            truth = str(ik14[anchor])
            if len(positive) == 0 or np.any(ik14[positive] != truth):
                raise RuntimeError(f"invalid positive candidates for query {anchor}")
            if len(negative) == 0 or np.any(ik14[negative] == truth):
                raise RuntimeError(f"invalid negative candidates for query {anchor}")
            candidate_rows = np.concatenate((positive, negative))
            if np.any(adduct[candidate_rows] != adduct[anchor]):
                raise RuntimeError(f"cross-adduct candidate for query {anchor}")
            if fold[anchor] != "train" or np.any(fold[candidate_rows] != "train"):
                raise RuntimeError(f"non-train row in query group {anchor}")

            grouped: dict[str, list[int]] = defaultdict(list)
            for row in negative:
                grouped[str(ik14[int(row)])].append(int(row))
            identities = [truth, *sorted(grouped)]
            references = {truth: [int(row) for row in positive]}
            references.update({identity: sorted(set(rows)) for identity, rows in grouped.items()})
            query_index = len(query_rows)
            for identity in identities:
                rows = references[identity]
                pair_candidate_row.extend(rows)
                molecule_ptr.append(len(pair_candidate_row))
                molecule_query.append(query_index)
                molecule_label.append(int(identity == truth))
                molecule_ik14.append(identity)
                molecule_formula.append(str(formula[rows[0]]))
            query_ptr.append(len(molecule_label))
            query_rows.append(anchor)
            query_ik14.append(truth)
            query_formula.append(str(formula[anchor]))
            query_adduct.append(str(adduct[anchor]))
            query_membership.append(str(membership[anchor]))

        pool_audit.append({
            "path": str(pool_path.resolve()),
            "sha256": sha256(pool_path),
            "report_sha256": sha256(report_file),
            "adduct": report["adduct"],
            "queries": int(len(anchors)),
            "positive_edges": int(len(positive_idx)),
            "negative_edges": int(len(negative_idx)),
        })

    query_ptr_array = np.asarray(query_ptr, dtype=np.int64)
    molecule_ptr_array = np.asarray(molecule_ptr, dtype=np.int64)
    molecule_label_array = np.asarray(molecule_label, dtype=np.int8)
    if query_ptr_array[-1] != len(molecule_label_array):
        raise RuntimeError("query pointer does not span molecules")
    if molecule_ptr_array[-1] != len(pair_candidate_row):
        raise RuntimeError("molecule pointer does not span candidate spectra")
    for left, right in zip(query_ptr_array[:-1], query_ptr_array[1:]):
        labels = molecule_label_array[left:right]
        if len(labels) < 2 or labels[0] != 1 or labels.sum() != 1:
            raise RuntimeError("each query must have exactly one first-position positive molecule")

    arrays = {
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "query_ik14": np.asarray(query_ik14, dtype="U14"),
        "query_formula": np.asarray(query_formula, dtype=str),
        "query_adduct": np.asarray(query_adduct, dtype=str),
        "query_simulation_challenge_membership": np.asarray(query_membership, dtype=str),
        "query_ptr": query_ptr_array,
        "molecule_ptr": molecule_ptr_array,
        "molecule_query": np.asarray(molecule_query, dtype=np.int64),
        "molecule_label": molecule_label_array,
        "molecule_ik14": np.asarray(molecule_ik14, dtype="U14"),
        "molecule_formula": np.asarray(molecule_formula, dtype=str),
        "pair_candidate_row": np.asarray(pair_candidate_row, dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    temporary = args.output.with_name(args.output.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(args.output)
    molecules_per_query = np.diff(query_ptr_array)
    refs_per_molecule = np.diff(molecule_ptr_array)
    report = {
        "status": "chemaware_corrected_candidate_manifest_built",
        "formal_training_authorized": False,
        "data_contract": "train_primary_all_p3_disjoint_v1",
        "candidate_contract": (
            "same-adduct query-centred strict-10ppm; tolerance-rounded same-spectrum "
            "pairs excluded upstream; positive identity first; all candidate spectra retained"
        ),
        "counts": {
            "queries": int(len(query_rows)),
            "query_identities": int(len(set(query_ik14))),
            "query_formulas": int(len(set(query_formula))),
            "candidate_molecules": int(len(molecule_label)),
            "candidate_spectrum_edges": int(len(pair_candidate_row)),
            "membership": {
                str(value): int(np.sum(np.asarray(query_membership) == value))
                for value in sorted(set(query_membership))
            },
            "adduct": {
                str(value): int(np.sum(np.asarray(query_adduct) == value))
                for value in sorted(set(query_adduct))
            },
        },
        "molecules_per_query": {
            "min": int(molecules_per_query.min()),
            "median": float(np.median(molecules_per_query)),
            "p90": float(np.quantile(molecules_per_query, .9)),
            "max": int(molecules_per_query.max()),
            "mean": float(molecules_per_query.mean()),
        },
        "reference_spectra_per_molecule": {
            "min": int(refs_per_molecule.min()),
            "median": float(np.median(refs_per_molecule)),
            "p90": float(np.quantile(refs_per_molecule, .9)),
            "max": int(refs_per_molecule.max()),
            "mean": float(refs_per_molecule.mean()),
        },
        "empty_query_formulas": int(np.sum(np.asarray(query_formula) == "")),
        "simulation_challenge_semantics": (
            "spectrum-simulation benchmark subset membership; not provenance; not filtered"
        ),
        "pools": pool_audit,
        "provenance": {
            "hdf5_path": str(args.data.resolve()),
            "hdf5_sha256": sha256(args.data),
            "manifest_sha256": sha256(args.output),
            "script_sha256": sha256(Path(__file__)),
        },
        "next_gate": (
            "Add official DreaMS and frozen raw-spectral scores with independent hash audit; "
            "then freeze formula/scaffold folds and three matched training arms before training."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
