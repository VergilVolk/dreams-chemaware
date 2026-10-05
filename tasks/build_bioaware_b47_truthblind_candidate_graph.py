#!/usr/bin/env python
"""Freeze B47 observable query events and mass-only candidate graphs without truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import h5py
import numpy as np


EXPECTED_HDF5_SHA256 = (
    "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f"
)
ALLOWED_ADDUCTS = {"[M+H]+", "[M+Na]+"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_events(join_dir: Path) -> tuple[list[dict], dict]:
    report_path = join_dir / "report.json"
    events_path = join_dir / "query_events.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("pass_to_truthblind_candidate_graph"):
        raise RuntimeError(f"feature-join gate failed: {join_dir}")
    if report.get("provenance", {}).get("query_events_sha256") != digest(events_path):
        raise RuntimeError(f"query-event provenance mismatch: {join_dir}")
    with gzip.open(events_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != report["primary_join"]["sample_feature_events"]:
        raise RuntimeError(f"query-event count mismatch: {join_dir}")
    return rows, report


def quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "median": 0.0, "p90": 0.0, "maximum": 0}
    ordered = sorted(values)
    return {
        "minimum": int(ordered[0]), "median": float(statistics.median(ordered)),
        "p90": float(ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]),
        "maximum": int(ordered[-1]),
    }


def copy_selected_mgf_blocks(queries: list[dict], destination: Path) -> None:
    wanted: dict[str, dict[int, str]] = defaultdict(dict)
    for row in queries:
        path = str(Path(row["mgf_path"]).resolve())
        index = int(row["representative_spectrum_index"])
        if index in wanted[path]:
            raise RuntimeError(f"one MGF spectrum selected by multiple events: {path}#{index}")
        wanted[path][index] = row["query_id"]
    written = set()
    with destination.open("w", encoding="utf-8", newline="\n") as target:
        for path_string in sorted(wanted):
            path = Path(path_string)
            if not path.is_file():
                raise FileNotFoundError(path)
            current: list[str] | None = None
            spectrum_index = -1
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for raw in source:
                    line = raw.rstrip("\r\n")
                    upper = line.strip().upper()
                    if upper == "BEGIN IONS":
                        if current is not None:
                            raise RuntimeError(f"nested MGF block: {path}")
                        spectrum_index += 1
                        current = ["BEGIN IONS"]
                    elif upper == "END IONS":
                        if current is None:
                            raise RuntimeError(f"MGF END without BEGIN: {path}")
                        query_id = wanted[path_string].get(spectrum_index)
                        if query_id is not None:
                            body = [value for value in current[1:] if not value.upper().startswith("TITLE=")]
                            target.write("BEGIN IONS\n")
                            target.write(f"TITLE={query_id}\n")
                            for value in body:
                                target.write(value + "\n")
                            target.write("END IONS\n\n")
                            written.add(query_id)
                        current = None
                    elif current is not None:
                        current.append(line)
            if current is not None:
                raise RuntimeError(f"unterminated MGF block: {path}")
    expected = {row["query_id"] for row in queries}
    if written != expected:
        raise RuntimeError(
            f"selected MGF extraction mismatch missing={len(expected-written)} extra={len(written-expected)}"
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--st001122-join", type=Path, required=True)
    parser.add_argument("--st003356-join", type=Path, required=True)
    parser.add_argument("--reference-hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    reference = args.reference_hdf5.resolve()
    if digest(reference) != EXPECTED_HDF5_SHA256:
        raise RuntimeError("reference HDF5 SHA256 mismatch")

    all_events = []
    reports = {}
    excluded_sample_types = Counter()
    for join_dir in (args.st001122_join.resolve(), args.st003356_join.resolve()):
        rows, report = read_events(join_dir)
        study = report["study"]
        reports[study] = {"path": str(join_dir), "report_sha256": digest(join_dir / "report.json")}
        for row in rows:
            sample = row["sample"]
            if study == "ST003356" and "_BK_" in sample:
                excluded_sample_types["ST003356_blank"] += 1
                continue
            if study == "ST003356" and "_QC_" in sample:
                excluded_sample_types["ST003356_pooled_QC"] += 1
                continue
            all_events.append(row)
    if not all_events:
        raise RuntimeError("no real-sample observable events")

    with h5py.File(reference, "r") as handle:
        required = {"precursor_mz", "adduct", "INCHIKEY", "FORMULA"}
        if not required.issubset(handle.keys()):
            raise RuntimeError(f"reference HDF5 missing {sorted(required-set(handle.keys()))}")
        ref_mz = np.asarray(handle["precursor_mz"][:], dtype=float)
        ref_adduct = decode(handle["adduct"][:])
        ref_ik14 = np.asarray([value[:14].upper() for value in decode(handle["INCHIKEY"][:])])
        ref_formula = decode(handle["FORMULA"][:])
    valid = (
        np.isfinite(ref_mz) & (ref_mz > 0)
        & np.isin(ref_adduct.astype(str), sorted(ALLOWED_ADDUCTS))
        & (np.char.str_len(ref_ik14.astype(str)) == 14)
    )
    valid_rows = np.flatnonzero(valid)
    order = np.argsort(ref_mz[valid_rows], kind="stable")
    sorted_rows = valid_rows[order]
    sorted_mz = ref_mz[sorted_rows]

    queries = []
    candidate_references = []
    events_lt_two = 0
    for position, event in enumerate(all_events, 1):
        feature_mz = float(event["feature_mz"])
        tolerance = feature_mz * args.ppm * 1e-6
        left = int(np.searchsorted(sorted_mz, feature_mz - tolerance, side="left"))
        right = int(np.searchsorted(sorted_mz, feature_mz + tolerance, side="right"))
        rows = sorted_rows[left:right]
        identities = sorted(set(ref_ik14[rows]))
        if len(identities) < 2:
            events_lt_two += 1
            continue
        query_id = "B47|" + event["event_id"]
        queries.append({
            **event, "query_id": query_id, "candidate_identities": len(identities),
            "candidate_reference_spectra": len(rows),
            "candidate_protocol": (
                f"positive-library precursor m/z within {args.ppm:g} ppm; adduct unknown"
            ),
        })
        for row in rows:
            candidate_references.append({
                "query_id": query_id, "candidate_id": str(ref_ik14[row]),
                "reference_row": int(row), "reference_precursor_mz": float(ref_mz[row]),
                "reference_adduct": str(ref_adduct[row]),
                "candidate_formula": str(ref_formula[row]),
                "mass_error_ppm": float((ref_mz[row] - feature_mz) / feature_mz * 1e6),
            })
        if position % 20000 == 0:
            print(f"[candidate graph] {position:,}/{len(all_events):,}", flush=True)
    if not queries or not candidate_references:
        raise RuntimeError("candidate graph is empty")
    if len({row["query_id"] for row in queries}) != len(queries):
        raise RuntimeError("query IDs are not unique")

    output.mkdir(parents=True, exist_ok=False)
    query_path = output / "queries.csv.gz"
    candidate_path = output / "candidate_references.csv.gz"
    mgf_path = output / "queries.mgf"
    with gzip.open(query_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queries[0]))
        writer.writeheader(); writer.writerows(queries)
    with gzip.open(candidate_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_references[0]))
        writer.writeheader(); writer.writerows(candidate_references)
    copy_selected_mgf_blocks(queries, mgf_path)

    by_query = Counter(row["query_id"] for row in candidate_references)
    by_identity = defaultdict(set)
    for row in candidate_references:
        by_identity[row["query_id"]].add(row["candidate_id"])
    if any(len(by_identity[row["query_id"]]) != int(row["candidate_identities"]) for row in queries):
        raise RuntimeError("candidate molecule counts do not replay")
    source_counts = Counter(row["study"] for row in queries)
    gates = {
        "observable_events_ge_10000": len(all_events) >= 10000,
        "evaluable_queries_ge_1000": len(queries) >= 1000,
        "both_sources_ge_500_queries": all(source_counts[name] >= 500 for name in ("ST001122", "ST003356")),
        "every_query_has_ge_two_candidates": all(len(value) >= 2 for value in by_identity.values()),
        "only_positive_reference_adducts": all(row["reference_adduct"] in ALLOWED_ADDUCTS for row in candidate_references),
        "phenotype_columns_absent": all("disease" not in row and "condition" not in row for row in queries),
        "truth_columns_absent": all(not any("truth" in key.casefold() for key in row) for row in queries),
        "P2b_absent": True,
    }
    report = {
        "status": "bioaware_b47_truthblind_candidate_graph_frozen",
        "formal": True, "observable_real_sample_events": len(all_events),
        "events_with_lt_two_candidate_identities": events_lt_two,
        "queries": len(queries), "sources": dict(sorted(source_counts.items())),
        "distinct_sample_feature_ids": len(
            {(row["study"], row["sample"], row["feature_id"]) for row in queries}
        ),
        "candidate_reference_rows": len(candidate_references),
        "candidate_identities_global": len({row["candidate_id"] for row in candidate_references}),
        "candidate_identities_per_query": quantiles([len(value) for value in by_identity.values()]),
        "candidate_reference_spectra_per_query": quantiles(list(by_query.values())),
        "excluded_observable_controls": dict(sorted(excluded_sample_types.items())),
        "reference": {
            "rows": len(ref_mz), "eligible_positive_rows": len(valid_rows),
            "adducts": dict(Counter(ref_adduct[valid].astype(str))),
        },
        "protocol": {
            "query_unit": "real-sample MS1 feature with one truth-blind representative MS2",
            "representative": "highest peak TIC, then peak count, then first index",
            "candidate_generation": (
                f"reference precursor m/z within {args.ppm:g} ppm, positive adduct "
                "library, grouped by IK14"
            ),
            "query_adduct": "unknown; [M+H]+ and [M+Na]+ library spectra compete in one deployable mass window",
            "ties": "will count against the positive after sealed truth is opened",
        },
        "gates": gates, "pass_to_query_embedding_and_seed_construction": all(gates.values()),
        "contracts": {
            "truth_opened": False, "algorithm_outputs_opened": False,
            "phenotype_used": False, "P2b_used": False,
        },
        "provenance": {
            "reference_hdf5_sha256": digest(reference), "joins": reports,
            "queries_sha256": digest(query_path),
            "candidate_references_sha256": digest(candidate_path),
            "queries_mgf_sha256": digest(mgf_path),
        },
        "claim_limit": (
            "This freezes a truth-blind query/candidate denominator. It does not establish "
            "truth coverage, DreaMS errors, reaction-reachable errors, BioAware gain, or SOTA."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
