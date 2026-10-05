"""Export the complete official-pool candidate registry for chemical scoring.

This is a metadata-only bridge from the corrected candidate manifest to the
existing MassBank rule scorer.  It exports every manifest query whose anchor
is in the complete official MassSpecGym strict-10-ppm train pool.  Candidate
truth labels are deliberately absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from export_chemaware_sirius_source_panel import (
    connectivity_smiles,
    smiles_formula,
    text,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-train-pool", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.official_train_pool, allow_pickle=False) as loaded:
        official_anchors = set(map(int, loaded["anchor_idx"]))
    selected_queries = np.flatnonzero(np.isin(manifest["query_row"], list(official_anchors)))
    if not len(selected_queries):
        raise RuntimeError("manifest and official train pool do not overlap")

    query_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    structure_cache: dict[str, str] = {}
    total_queries = len(selected_queries)
    with h5py.File(args.data, "r") as data:
        total_rows = len(data["spectrum"])
        for index, query_value in enumerate(selected_queries):
            query = int(query_value)
            if index % 10000 == 0:
                print(
                    f"[chem-full-registry] query {index}/{total_queries} "
                    f"(identities {len(structure_cache)})",
                    flush=True,
                )
            spectrum_row = int(manifest["query_row"][query])
            if spectrum_row < 0 or spectrum_row >= total_rows:
                raise RuntimeError(f"query row outside HDF5: {spectrum_row}")
            left, right = map(int, manifest["query_ptr"][query:query + 2])
            feature_id = f"q{query}_r{spectrum_row}"
            spectrum = np.asarray(data["spectrum"][spectrum_row], dtype=np.float32)
            peaks = int(np.sum((spectrum[0] > 0) & (spectrum[1] > 0)))
            candidate_formulas = list(map(str, manifest["molecule_formula"][left:right]))
            query_records.append({
                "feature_id": feature_id,
                "manifest_query": query,
                "spectrum_row": spectrum_row,
                "adduct": text(data["adduct"][spectrum_row]),
                "precursor_mz": f"{float(data['precursor_mz'][spectrum_row]):.8f}",
                "instrument": text(data["INSTRUMENT_TYPE"][spectrum_row]),
                "instrument_profile": "not_required_by_massbank_scorer",
                "collision_energy": "",
                "peaks": peaks,
                "candidate_count": right - left,
                "candidate_formula_count": len(set(candidate_formulas)),
                "candidate_formulas": ";".join(sorted(set(candidate_formulas))),
            })
            for local_candidate, molecule in enumerate(range(left, right)):
                ref_left, ref_right = map(
                    int, manifest["molecule_ptr"][molecule:molecule + 2],
                )
                rows = np.asarray(
                    manifest["pair_candidate_row"][ref_left:ref_right], dtype=np.int64,
                )
                if not len(rows):
                    raise RuntimeError(f"query {query} candidate {local_candidate} has no references")
                ik14 = str(manifest["molecule_ik14"][molecule])
                formula = str(manifest["molecule_formula"][molecule])
                if ik14 not in structure_cache:
                    # The corrected manifest already proves that every row in
                    # this molecule block has this IK14.  Connectivity SMILES
                    # is stereo-invariant, so one deterministic representative
                    # is sufficient for the rule scorer and avoids serializing
                    # millions of unused reference-row identifiers.
                    representative = connectivity_smiles(
                        text(data["smiles"][int(rows[0])])
                    )
                    if smiles_formula(representative) != formula:
                        raise RuntimeError(
                            f"candidate structure/formula mismatch: {ik14} {formula}"
                        )
                    structure_cache[ik14] = representative
                candidate_records.append({
                    "feature_id": feature_id,
                    "manifest_query": query,
                    "local_candidate": local_candidate,
                    "ik14": ik14,
                    "formula": formula,
                    "smiles": structure_cache[ik14],
                })

    if len({int(row["manifest_query"]) for row in query_records}) != len(query_records):
        raise RuntimeError("query registry repeats queries")
    grouped: dict[int, list[int]] = {}
    for row in candidate_records:
        grouped.setdefault(int(row["manifest_query"]), []).append(int(row["local_candidate"]))
    if any(values != list(range(len(values))) for values in grouped.values()):
        raise RuntimeError("candidate registry is not contiguous per query")
    report = {
        "status": "CHEMAWARE_FULL_MASSSPECGYM_CANDIDATE_REGISTRY_COMPLETE",
        "truth_fields_exported": False,
        "official_pool_anchors": len(official_anchors),
        "manifest_queries_exported": len(query_records),
        "manifest_queries_not_in_official_pool": int(len(manifest["query_row"]) - len(query_records)),
        "candidate_molecules_exported": len(candidate_records),
        "candidate_identities": len(structure_cache),
        "coverage_contract": "every corrected-manifest query present in the complete official train pool",
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_train_pool_sha256": sha256(args.official_train_pool),
            "data_sha256": sha256(args.data),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_full_msg_registry_", dir=args.output.parent))
    try:
        with (temporary / "query_registry.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(query_records[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(query_records)
        with (temporary / "candidate_ledger.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidate_records[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(candidate_records)
        (temporary / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
