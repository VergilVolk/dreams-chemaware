#!/usr/bin/env python
"""Build an ambiguity-free DreaMS query/reference cache for MetDNA3 development.

Level-1 labels select and evaluate queries, but never enter spectra or scores.
One external MS2 spectrum shared by distinct truth identities is excluded.
Candidates are frozen MassSpecGym structures in the same adduct and 10 ppm
window; references and queries are subsequently encoded by one official model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation._inference import preprocess_spectrum  # noqa: E402
from metdna3_mzml import iter_ms2_spectra, iter_spectrum_metadata  # noqa: E402


def decode(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1"),
    )
    parser.add_argument(
        "--preflight", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_ms2_preflight.json"),
    )
    parser.add_argument(
        "--mzml-dir", type=Path,
        default=Path("data/external/metdna3_2025/mzml/development"),
    )
    parser.add_argument(
        "--reference-hdf5", type=Path,
        default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v1"),
    )
    parser.add_argument("--match-ppm", type=float, default=15.0)
    parser.add_argument("--match-rt-sec", type=float, default=25.0)
    parser.add_argument("--candidate-ppm", type=float, default=10.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--minimum-identities", type=int, default=75)
    parser.add_argument("--minimum-queries", type=int, default=100)
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--query-prefix", default=None)
    args = parser.parse_args()

    truth_path = args.development_dir / args.truth_name
    split_path = args.development_dir / "identity_splits.csv.gz"
    required = [truth_path, split_path, args.preflight, args.reference_hdf5]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if not preflight.get("formal") or not preflight.get("pass_to_dreams_ranking", preflight.get("pass_to_dreams_development_ranking")):
        raise RuntimeError("formal MetDNA3 MS2 preflight did not pass")
    if preflight.get("scope", args.scope) != args.scope:
        raise RuntimeError("preflight/cache scope mismatch")

    truth = pd.read_csv(truth_path).reset_index(names="truth_row")
    paths = sorted(args.mzml_dir.glob("*.mzML"))
    metadata: dict[str, list[dict]] = {"positive": [], "negative": []}
    for path in paths:
        polarity = "positive" if "_pos_" in path.name else "negative"
        for row in iter_spectrum_metadata(path):
            if row["ms_level"] != 2 or row["n_peaks"] < 2:
                continue
            if not math.isfinite(row["precursor_mz"]) or not math.isfinite(row["rt_sec"]):
                continue
            row["source_file"] = path.name
            row["spectrum_key"] = f"{path.name}|{row['spectrum_id']}"
            metadata[polarity].append(row)

    options_by_truth: dict[int, tuple[object, list[tuple]]] = {}
    unmatched_truth_rows: list[int] = []
    compatible_identities: dict[str, set[str]] = {}
    for row in truth.itertuples(index=False):
        candidates = []
        for spectrum in metadata[str(row.polarity)]:
            ppm = abs(float(spectrum["precursor_mz"]) - float(row.mz)) / float(row.mz) * 1e6
            rt_delta = abs(float(spectrum["rt_sec"]) - float(row.rt))
            if ppm <= args.match_ppm and rt_delta <= args.match_rt_sec:
                metric = (ppm / args.match_ppm) ** 2 + (rt_delta / args.match_rt_sec) ** 2
                candidates.append((metric, spectrum["spectrum_key"], spectrum, ppm, rt_delta))
                compatible_identities.setdefault(spectrum["spectrum_key"], set()).add(
                    str(row.ik14)
                )
        if not candidates:
            if args.scope != "external":
                raise RuntimeError(f"preflight mismatch: no MS2 for truth row {row.truth_row}")
            unmatched_truth_rows.append(int(row.truth_row))
            continue
        options_by_truth[int(row.truth_row)] = (row, candidates)

    preflight_matched = int(preflight.get("combined", {}).get("matched_level1_rows", -1))
    if preflight_matched >= 0 and len(options_by_truth) != preflight_matched:
        raise RuntimeError(
            "preflight/cache matched-row count differs: "
            f"cache={len(options_by_truth)} preflight={preflight_matched}"
        )
    if args.scope == "external" and len(options_by_truth) + len(unmatched_truth_rows) != len(truth):
        raise RuntimeError("external matched and explicitly unmatched truth rows do not reconcile")

    assignments: list[dict] = []
    no_exclusive_spectrum = 0
    for row, candidates in options_by_truth.values():
        exclusive = [
            item for item in candidates
            if compatible_identities[item[1]] == {str(row.ik14)}
        ]
        if not exclusive:
            no_exclusive_spectrum += 1
            continue
        _, _, best, ppm, rt_delta = min(exclusive, key=lambda item: (item[0], item[1]))
        assignment_row = {
            "truth_row": int(row.truth_row), "truth_ik14": str(row.ik14),
            "truth_formula": str(row.formula), "adduct": str(row.adduct),
            "feature_mz": float(row.mz), "feature_rt_sec": float(row.rt),
            "polarity": str(row.polarity), "source_file": best["source_file"],
            "spectrum_id": best["spectrum_id"], "spectrum_key": best["spectrum_key"],
            "observed_precursor_mz": float(best["precursor_mz"]),
            "observed_rt_sec": float(best["rt_sec"]), "match_ppm": float(ppm),
            "match_rt_sec": float(rt_delta),
        }
        for optional in ("panel_id", "sample_type", "separation", "unit_id"):
            if hasattr(row, optional):
                assignment_row[optional] = str(getattr(row, optional))
        assignments.append(assignment_row)
    assignment = pd.DataFrame(assignments)
    assignment = assignment.sort_values(
        ["match_ppm", "match_rt_sec", "truth_row"]
    ).drop_duplicates(["spectrum_key", "truth_ik14"])

    with h5py.File(args.reference_hdf5, "r") as handle:
        ref_ik = np.asarray([decode(value)[:14].upper() for value in handle["INCHIKEY"][:]])
        ref_mz = np.asarray(handle["precursor_mz"][:], dtype=float)
        ref_adduct = np.asarray([decode(value) for value in handle["adduct"][:]])
    candidate_rows: list[dict] = []
    query_rows: list[dict] = []
    for source in assignment.itertuples(index=False):
        mask = (ref_adduct == source.adduct) & (
            np.abs(ref_mz - source.feature_mz) <= source.feature_mz * args.candidate_ppm * 1e-6
        )
        rows = np.flatnonzero(mask)
        identities = set(ref_ik[rows])
        if source.truth_ik14 not in identities or len(identities) < 2:
            continue
        if args.query_prefix is not None:
            prefix = str(args.query_prefix)
        elif args.scope == "development":
            prefix = "M3H"
        elif args.scope == "internal_rplc":
            prefix = "M3R"
        else:
            raise RuntimeError("external cache requires an explicit unique --query-prefix")
        query_id = f"{prefix}:{int(source.truth_row):04d}:{source.spectrum_id}"
        query_rows.append({**source._asdict(), "query_id": query_id, "candidate_identities": len(identities)})
        for ref_row in rows:
            candidate_rows.append({
                "query_id": query_id, "candidate_id": str(ref_ik[ref_row]),
                "reference_row": int(ref_row), "truth_candidate_id": source.truth_ik14,
                "truth_formula": source.truth_formula, "adduct": source.adduct,
            })
    all_external = assignment.copy().reset_index(drop=True)
    query = pd.DataFrame(query_rows)
    candidate = pd.DataFrame(candidate_rows)
    if query.empty:
        raise RuntimeError("no ambiguity-free MetDNA3 DreaMS queries")
    if len(query) < args.minimum_queries or query["truth_ik14"].nunique() < args.minimum_identities:
        raise RuntimeError(
            "insufficient ambiguity-free development coverage: "
            f"queries={len(query)}, identities={query['truth_ik14'].nunique()}, "
            f"minimum_queries={args.minimum_queries}, minimum_identities={args.minimum_identities}"
        )

    wanted = set(all_external["spectrum_key"])
    tensors: dict[str, np.ndarray] = {}
    for position, path in enumerate(paths, 1):
        local = {key for key in wanted if key.startswith(path.name + "|")}
        if not local:
            continue
        for spectrum in iter_ms2_spectra(path):
            key = f"{path.name}|{spectrum['spectrum_id']}"
            if key not in local:
                continue
            raw = np.vstack([spectrum["mz"], spectrum["intensity"]])
            tensors[key] = preprocess_spectrum(
                raw, float(spectrum["precursor_mz"]), args.n_highest_peaks
            ).numpy()
        print(f"[query spectra {position}/{len(paths)}] recovered={len(tensors)}/{len(wanted)}", flush=True)
    if set(tensors) != wanted:
        raise RuntimeError(f"failed to decode {len(wanted-set(tensors))} selected query spectra")
    external_tensor = np.stack(
        [tensors[key] for key in all_external["spectrum_key"]]
    ).astype(np.float32)
    query_tensor = np.stack([tensors[key] for key in query["spectrum_key"]]).astype(np.float32)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = [
        output / "queries.csv.gz", output / "candidate_references.csv.gz",
        output / "query_tensors.npz", output / "external_spectra.csv.gz",
        output / "external_tensors.npz", output / "report.json",
    ]
    if any(path.exists() for path in files):
        raise RuntimeError(f"fail-closed: DreaMS cache output already exists: {output}")
    query.to_csv(files[0], index=False, compression="gzip")
    candidate.to_csv(files[1], index=False, compression="gzip")
    np.savez_compressed(files[2], query_tensor=query_tensor)
    all_external.to_csv(files[3], index=False, compression="gzip")
    np.savez_compressed(files[4], external_tensor=external_tensor)
    split = pd.read_csv(split_path)
    evaluable_per_rotation = {
        str(fold): int(query["truth_ik14"].isin(
            split[(split["fold"] == fold) & (split["role"] == "heldout")]["ik14"]
        ).sum()) for fold in range(10)
    }
    report = {
        "status": "bioaware_metdna3_dreams_cache_complete", "formal": True,
        "scope": args.scope,
        "queries": int(len(query)), "query_identities": int(query["truth_ik14"].nunique()),
        "query_formulas": int(query["truth_formula"].nunique()),
        "external_level1_spectra": int(len(all_external)),
        "external_level1_identities": int(all_external["truth_ik14"].nunique()),
        "truth_rows_total": int(len(truth)),
        "truth_rows_with_no_matching_ms2": int(len(unmatched_truth_rows)),
        "unmatched_truth_rows": unmatched_truth_rows,
        "spectra_compatible_with_multiple_truth_identities": int(sum(
            len(identities) > 1 for identities in compatible_identities.values()
        )),
        "truth_rows_without_any_exclusive_spectrum": int(no_exclusive_spectrum),
        "candidate_reference_rows": int(len(candidate)),
        "candidate_identities": int(candidate["candidate_id"].nunique()),
        "median_candidate_identities_per_query": float(query["candidate_identities"].median()),
        "power_boundary": (
            "This cache is an execution artifact only. Statistical evidence is computed by the "
            "frozen evaluator; the external 16-panel test remains locked."
        ),
        "evaluable_queries_per_rotation": evaluable_per_rotation,
        "contracts": {
            "one_external_spectrum_one_truth_identity": True,
            "unmatched_external_truth_rows_explicitly_reported": args.scope == "external",
            "candidate_protocol": f"same adduct, {args.candidate_ppm:g} ppm around frozen Level-1 feature m/z",
            "ties": "count against truth in evaluator", "P2b": "forbidden",
            "external_test_opened": False,
        },
        "provenance": {
            "truth_sha256": sha256(truth_path), "split_sha256": sha256(split_path),
            "preflight_sha256": sha256(args.preflight),
            "reference_hdf5_sha256": sha256(args.reference_hdf5),
            "queries_sha256": sha256(files[0]), "candidates_sha256": sha256(files[1]),
            "tensors_sha256": sha256(files[2]), "external_spectra_sha256": sha256(files[3]),
            "external_tensors_sha256": sha256(files[4]),
        },
        "claim_limit": "Frozen execution cache only; no embedding or BioAware performance.",
    }
    atomic_json(files[5], report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
