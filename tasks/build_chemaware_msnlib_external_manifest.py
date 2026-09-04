"""Build a score-blind MSnLib external-retrieval candidate manifest.

The builder reserves post-GeMS, MassSpecGym/MoNA-identity-disjoint MSnLib
spectra for future evaluation.  It selects one reference spectrum per
IK14/adduct and, when possible, a query from a distinct raw acquisition.  It
then creates same-adduct, strict-ppm candidate sets without running DreaMS or
using any model score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parent.parent
DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
SUPPORTED_ADDUCTS = {"[M+H]+", "[M+Na]+", "[M-H]-"}
MERGE_PRIORITY = {
    "SINGLE_BEST_SCAN": 0,
    "SAME_ENERGY": 1,
    "ALL_ENERGIES": 2,
    "SINGLE_SCAN": 3,
    "ALL_MSN_TO_PSEUDO_MS2": 4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def valid_full_ik(value: str) -> bool:
    parts = value.split("-")
    return len(parts) == 3 and len(parts[0]) == 14 and len(parts[1]) == 10 and len(parts[2]) == 1


def ik14(value: str) -> str:
    return value.split("-", 1)[0] if valid_full_ik(value) else ""


def first_text(value: Any) -> str:
    if isinstance(value, list):
        return first_text(value[0]) if value else ""
    return "" if value is None else str(value)


def raw_file_id(row: dict[str, Any]) -> str:
    raw = first_text(row.get("raw_file_name"))
    if raw:
        return Path(raw).stem
    for key in ("source_scan_usi", "usi"):
        value = first_text(row.get(key))
        if value.startswith("mzspec:"):
            parts = value.split(":")
            if len(parts) >= 4:
                return parts[2]
    feature = first_text(row.get("feature_id") or row.get("featurelist_feature_id"))
    return feature.split(".mzML", 1)[0] if ".mzML" in feature else feature


def acquisition_date(row: dict[str, Any], raw_id: str) -> str:
    for value in (
        raw_id,
        row.get("raw_file_name"),
        row.get("source_scan_usi"),
        row.get("usi"),
        row.get("feature_id"),
        row.get("featurelist_feature_id"),
    ):
        values = value if isinstance(value, list) else [value]
        for item in values:
            match = DATE_RE.search("" if item is None else str(item))
            if match:
                return match.group(1)
    return ""


def canonical_spectrum_hash(mz: np.ndarray, intensity: np.ndarray) -> str:
    """Scale-invariant rounded hash of the 128 most intense valid peaks."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if not len(mz):
        return "empty"
    if len(mz) > 128:
        chosen = np.argsort(-intensity, kind="stable")[:128]
        mz, intensity = mz[chosen], intensity[chosen]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    intensity = intensity / intensity.max()
    payload = np.stack((np.round(mz, 2), np.round(intensity, 4)), axis=1)
    return hashlib.blake2b(payload.tobytes(), digest_size=16).hexdigest()


def json_spectrum_hash(row: dict[str, Any]) -> str:
    peaks = np.asarray(row.get("peaks") or [], dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[1] < 2:
        return "empty"
    return canonical_spectrum_hash(peaks[:, 0], peaks[:, 1])


def resolve_structure(row: dict[str, Any]) -> tuple[str, str, str]:
    mol = None
    smiles = str(row.get("smiles") or "").strip()
    inchi = str(row.get("inchi") or "").strip()
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None and inchi:
        mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return "", "", ""
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
    return canonical, scaffold, rdMolDescriptors.CalcMolFormula(mol)


def massspecgym_sets(path: Path) -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    hashes: set[str] = set()
    with h5py.File(path, "r") as handle:
        raw_ik = handle["INCHIKEY"].asstr()[:]
        spectra = handle["spectrum"]
        for index, value in enumerate(raw_ik):
            value = str(value)
            if len(value) == 14:
                identities.add(value)
            elif valid_full_ik(value):
                identities.add(ik14(value))
            else:
                raise RuntimeError(f"unexpected MassSpecGym identity: {value!r}")
            spectrum = np.asarray(spectra[index])
            hashes.add(canonical_spectrum_hash(spectrum[0], spectrum[1]))
    return identities, hashes


def mona_sets(paths: list[Path]) -> tuple[set[str], set[str]]:
    identities: set[str] = set()
    hashes: set[str] = set()
    for path in paths:
        with h5py.File(path, "r") as handle:
            for raw in handle["data"]:
                row = json.loads(decode(raw))
                identity = str(row[1])
                if len(identity) == 14:
                    identities.add(identity)
                hashes.add(
                    canonical_spectrum_hash(
                        np.asarray(row[2], dtype=np.float64),
                        np.asarray(row[4], dtype=np.float64),
                    )
                )
    return identities, hashes


def rank_record(row: dict[str, Any]) -> tuple[Any, ...]:
    explained = row["quality_explained_intensity"]
    purity = row["precursor_purity"]
    return (
        MERGE_PRIORITY.get(row["merge_type"], 99),
        -explained if np.isfinite(explained) else 1.0,
        -purity if np.isfinite(purity) else 1.0,
        -row["num_peaks"],
        row["file"],
        row["line"],
    )


def describe(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    if not len(array):
        return {"min": 0, "median": 0.0, "p90": 0.0, "max": 0, "mean": 0.0}
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, .9)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msnlib-dir", type=Path, default=ROOT / "data/msnlib")
    parser.add_argument(
        "--massspecgym", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--mona-pos", type=Path, default=ROOT / "data/models/mona_pos_full.hdf5",
    )
    parser.add_argument(
        "--mona-neg", type=Path, default=ROOT / "data/models/mona_neg_full.hdf5",
    )
    parser.add_argument("--post-gems-after", default="20221130")
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_msnlib_external_manifest_v2",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    msg_ik14, msg_hashes = massspecgym_sets(args.massspecgym)
    mona_ik14, mona_hashes = mona_sets([args.mona_pos, args.mona_neg])
    internal_hashes = msg_hashes | mona_hashes

    records: list[dict[str, Any]] = []
    rejection = Counter()
    invalid_json = 0
    input_hashes: dict[str, str] = {}
    structure_cache: dict[str, tuple[str, str, str]] = {}
    files = sorted(args.msnlib_dir.glob("*_ms2.json"))
    for file_index, path in enumerate(files):
        name_match = re.match(r"(\d{8})_(.+)_(pos|neg)_ms2\.json$", path.name)
        if not name_match:
            raise RuntimeError(f"unexpected filename: {path.name}")
        release, library, filename_polarity = name_match.groups()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                full = str(row.get("inchikey") or "").strip()
                identity = ik14(full)
                raw_id = raw_file_id(row)
                date = acquisition_date(row, raw_id)
                adduct = str(row.get("adduct") or "").strip()
                peaks = row.get("peaks") or []
                num_peaks = int(row.get("num_peaks") or len(peaks))
                precursor_mz = row.get("precursor_mz")
                checks = {
                    "invalid_inchikey": not identity,
                    "not_post_gems": not date or date <= args.post_gems_after,
                    "unsupported_adduct": adduct not in SUPPORTED_ADDUCTS,
                    "too_few_peaks": num_peaks < 5,
                    "chimeric_qc_not_passed": str(row.get("quality_chimeric") or "") != "PASSED",
                    "massspecgym_identity_overlap": identity in msg_ik14,
                    "mona_identity_overlap": identity in mona_ik14,
                    "missing_precursor_mz": precursor_mz is None,
                    "missing_raw_file_id": not raw_id,
                }
                failed = [key for key, value in checks.items() if value]
                if failed:
                    rejection.update(failed)
                    continue
                if full not in structure_cache:
                    structure_cache[full] = resolve_structure(row)
                canonical_smiles, murcko_scaffold, structure_formula = structure_cache[full]
                if not canonical_smiles:
                    rejection["invalid_or_missing_structure"] += 1
                    continue
                spectrum_hash = json_spectrum_hash(row)
                if spectrum_hash == "empty":
                    rejection["empty_spectrum_hash"] += 1
                    continue
                if spectrum_hash in internal_hashes:
                    rejection["internal_spectrum_hash_overlap"] += 1
                    continue
                records.append(
                    {
                        "record_id": len(records),
                        "file_index": file_index,
                        "file": path.name,
                        "line": line_number,
                        "release": release,
                        "library": library,
                        "filename_polarity": filename_polarity,
                        "raw_file_id": raw_id,
                        "acquisition_date": date,
                        "full_inchikey": full,
                        "ik14": identity,
                        "formula": str(row.get("formula") or "").strip(),
                        "structure_formula": structure_formula,
                        "canonical_smiles": canonical_smiles,
                        "murcko_scaffold": murcko_scaffold,
                        "adduct": adduct,
                        "precursor_mz": float(precursor_mz),
                        "merge_type": str(row.get("merge_type") or "missing"),
                        "collision_energy": json.dumps(
                            row.get("collision_energy"), ensure_ascii=False, separators=(",", ":")
                        ),
                        "num_peaks": num_peaks,
                        "quality_explained_intensity": float(
                            row.get("quality_explained_intensity", np.nan)
                        ),
                        "precursor_purity": float(row.get("precursor_purity", np.nan)),
                        "spectrum_hash": spectrum_hash,
                    }
                )
        input_hashes[path.name] = sha256(path)

    by_identity_adduct: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_identity_adduct[(row["ik14"], row["adduct"])].append(row)

    references: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for key in sorted(by_identity_adduct):
        values = sorted(by_identity_adduct[key], key=rank_record)
        reference = values[0]
        ref_row = dict(reference)
        ref_row["role"] = "reference"
        ref_row["role_id"] = len(references)
        references.append(ref_row)
        alternatives = [
            row for row in values
            if row["raw_file_id"] != reference["raw_file_id"]
            and row["spectrum_hash"] != reference["spectrum_hash"]
        ]
        if not alternatives:
            continue
        same_merge = [row for row in alternatives if row["merge_type"] == reference["merge_type"]]
        query = sorted(same_merge or alternatives, key=rank_record)[0]
        query_row = dict(query)
        query_row["role"] = "query"
        query_row["role_id"] = len(queries)
        query_row["true_reference_role_id"] = ref_row["role_id"]
        query_row["same_merge_type_as_true_reference"] = (
            query["merge_type"] == reference["merge_type"]
        )
        queries.append(query_row)

    ref_by_adduct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        ref_by_adduct[row["adduct"]].append(row)
    candidate_rows: list[dict[str, Any]] = []
    query_summaries: list[dict[str, Any]] = []
    for query in queries:
        candidates = []
        for reference in ref_by_adduct[query["adduct"]]:
            ppm_error = abs(reference["precursor_mz"] - query["precursor_mz"]) / query["precursor_mz"] * 1e6
            if ppm_error <= args.ppm:
                candidates.append((reference, ppm_error))
        true = [item for item in candidates if item[0]["ik14"] == query["ik14"]]
        distinct_identities = {item[0]["ik14"] for item in candidates}
        if len(true) != 1 or len(distinct_identities) < 2:
            continue
        same_formula_negatives = sum(
            1 for reference, _ in candidates
            if reference["ik14"] != query["ik14"]
            and reference["formula"] == query["formula"]
        )
        query_id = len(query_summaries)
        for reference, ppm_error in sorted(
            candidates, key=lambda item: (item[0]["precursor_mz"], item[0]["ik14"])
        ):
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "query_record_id": query["record_id"],
                    "reference_role_id": reference["role_id"],
                    "reference_record_id": reference["record_id"],
                    "label": int(reference["ik14"] == query["ik14"]),
                    "same_formula": int(reference["formula"] == query["formula"]),
                    "ppm_error": ppm_error,
                }
            )
        query_summaries.append(
            {
                **query,
                "query_id": query_id,
                "candidate_identities": len(distinct_identities),
                "same_formula_negative_identities": same_formula_negatives,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "eligible_records.csv.gz"
    references_path = output_dir / "reference_library.csv.gz"
    queries_path = output_dir / "queries.csv.gz"
    candidates_path = output_dir / "candidate_edges.csv.gz"
    pd.DataFrame(records).to_csv(records_path, index=False)
    pd.DataFrame(references).to_csv(references_path, index=False)
    pd.DataFrame(query_summaries).to_csv(queries_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_path, index=False)

    candidate_counts = [row["candidate_identities"] for row in query_summaries]
    same_formula_counts = [row["same_formula_negative_identities"] for row in query_summaries]
    report = {
        "status": "chemaware_msnlib_external_manifest_built",
        "formal_training_authorized": False,
        "model_scores_read": False,
        "split_assigned": False,
        "selection_contract": {
            "post_gems_boundary_exclusive": args.post_gems_after,
            "identity_granularity": "IK14 connectivity block",
            "identity_exclusions": ["MassSpecGym", "local MoNA pos+neg"],
            "spectrum_hash_contract": {
                "max_peaks": 128,
                "selection": "highest intensity before sorting by m/z",
                "intensity_normalization": "divide by spectrum maximum",
                "mz_round_decimals": 2,
                "intensity_round_decimals": 4,
                "digest": "BLAKE2b-128",
            },
            "spectrum_hash_exclusions": ["MassSpecGym", "local MoNA pos+neg"],
            "quality": "quality_chimeric PASSED and at least 5 peaks",
            "supported_adducts": sorted(SUPPORTED_ADDUCTS),
            "reference_selection": "deterministic quality/merge priority; one per IK14/adduct",
            "query_selection": (
                "distinct raw-file identifier and distinct rounded spectrum hash; prefer the "
                "same merge type as the true reference"
            ),
            "candidate_generation": f"same adduct and <= {args.ppm:g} ppm precursor-m/z error",
        },
        "counts": {
            "input_files": len(files),
            "invalid_json_lines_skipped": invalid_json,
            "eligible_spectrum_records": len(records),
            "eligible_ik14_adduct_groups": len(by_identity_adduct),
            "reference_spectra": len(references),
            "cross_acquisition_query_candidates_before_mass_window": len(queries),
            "mass_competitive_queries": len(query_summaries),
            "mass_competitive_query_ik14": len({row["ik14"] for row in query_summaries}),
            "mass_competitive_query_formula": len({row["formula"] for row in query_summaries}),
            "mass_competitive_query_murcko_scaffold": len(
                {row["murcko_scaffold"] for row in query_summaries}
            ),
            "candidate_edges": len(candidate_rows),
            "same_formula_negative_edges": int(
                sum(row["same_formula"] and not row["label"] for row in candidate_rows)
            ),
            "queries_with_same_formula_negative": int(sum(value > 0 for value in same_formula_counts)),
            "queries_same_merge_type_as_true_reference": int(
                sum(row["same_merge_type_as_true_reference"] for row in query_summaries)
            ),
        },
        "query_difficulty": {
            "candidate_identities": describe(candidate_counts),
            "same_formula_negative_identities": describe(same_formula_counts),
            "adduct_queries": dict(sorted(Counter(row["adduct"] for row in query_summaries).items())),
            "library_queries": dict(sorted(Counter(row["library"] for row in query_summaries).items())),
            "acquisition_date_queries": dict(
                sorted(Counter(row["acquisition_date"] for row in query_summaries).items())
            ),
        },
        "rejection_reasons_nonexclusive": dict(sorted(rejection.items())),
        "claim_limit": (
            "This freezes a score-blind candidate universe only. It is not yet a benchmark: "
            "formula/scaffold-disjoint splits, structural-difficulty labels, tie policy, and "
            "minimum sample-size gates remain to be frozen before model scoring."
        ),
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "massspecgym_sha256": sha256(args.massspecgym),
            "mona_pos_sha256": sha256(args.mona_pos),
            "mona_neg_sha256": sha256(args.mona_neg),
            "input_sha256": input_hashes,
            "eligible_records_sha256": sha256(records_path),
            "reference_library_sha256": sha256(references_path),
            "queries_sha256": sha256(queries_path),
            "candidate_edges_sha256": sha256(candidates_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
