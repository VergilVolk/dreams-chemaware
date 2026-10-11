"""Upgrade a complete Enveda audit by adding newly registered exclusions.

The expensive structure parsing and spectrum hashing are reused byte-for-byte
from the earlier score-blind manifest.  Only exclusion sources absent from that
manifest are loaded, and their identity/formula/hash membership is ORed into
the existing per-record flags.  No model or evaluation truth is accessed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_enveda180_scoreblind_manifest import (
    load_csv,
    load_exclusion_registry,
    load_hdf5,
    load_mgf,
    iter_mgf,
    secondary_spectrum_hash,
    sha256_file,
)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--source-mgf", type=Path)
    parser.add_argument("--expected-source-bytes", type=int)
    parser.add_argument("--expected-source-md5")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite upgraded audit: {args.out}")
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    base_report_path = args.base_audit / "report.json"
    base_manifest = args.base_audit / "eligible_records.csv.gz"
    base_conflicts = args.base_audit / "conflicting_spectrum_hashes.txt.gz"
    report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ENVEDA180_SCOREBLIND_MANIFEST_COMPLETE":
        raise RuntimeError("base Enveda audit is incomplete")
    if report.get("performance_scores_opened") is not False or report.get("model_loaded") is not False:
        raise RuntimeError("base Enveda audit is not score blind")
    declared_manifest_sha256 = report.get("manifest_sha256")
    actual_manifest_sha256 = sha256_file(base_manifest)
    manifest_byte_hash_status = (
        "not_recorded" if not declared_manifest_sha256 else
        "match" if actual_manifest_sha256 == declared_manifest_sha256 else "mismatch"
    )
    declared_conflicts_sha256 = report.get("conflicting_hashes_sha256")
    actual_conflicts_sha256 = sha256_file(base_conflicts)
    conflicts_byte_hash_status = (
        "not_recorded" if not declared_conflicts_sha256 else
        "match" if actual_conflicts_sha256 == declared_conflicts_sha256 else "mismatch"
    )
    with gzip.open(base_conflicts, "rt", encoding="utf-8") as handle:
        conflicts = [line.strip() for line in handle if line.strip()]
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in conflicts):
        raise RuntimeError("base conflict ledger contains malformed hashes")
    if conflicts != sorted(set(conflicts)):
        raise RuntimeError("base conflict ledger is not sorted and unique")
    expected_conflicts = int(report.get("cross_identity_conflicting_hashes", -1))
    if len(conflicts) != expected_conflicts:
        raise RuntimeError(
            f"base conflict-ledger semantic count mismatch: {len(conflicts)} != {expected_conflicts}"
        )

    base_columns = pd.read_csv(base_manifest, nrows=0).columns.tolist()
    reconstruct_secondary = "spectrum_hash_secondary" not in base_columns
    secondary_by_source_row: dict[int, str] = {}
    if reconstruct_secondary:
        if args.source_mgf is None or not args.source_mgf.is_file():
            raise RuntimeError("legacy manifest lacks secondary hashes; --source-mgf is required")
        declared_source_sha256 = report.get("source_mgf_sha256")
        if declared_source_sha256:
            if sha256_file(args.source_mgf) != declared_source_sha256:
                raise RuntimeError("source MGF differs from legacy Enveda audit")
            source_binding = "legacy_sha256_match"
        else:
            if args.expected_source_bytes is None or not args.expected_source_md5:
                raise RuntimeError(
                    "legacy audit did not record source SHA-256; expected bytes and MD5 are required"
                )
            actual_bytes = args.source_mgf.stat().st_size
            actual_md5 = md5_file(args.source_mgf)
            if actual_bytes != args.expected_source_bytes or actual_md5 != args.expected_source_md5:
                raise RuntimeError(
                    "source MGF fails the independent bytes+MD5 binding: "
                    f"bytes={actual_bytes}, md5={actual_md5}"
                )
            source_binding = "independent_bytes_and_md5_match"
        needed_rows = set(
            pd.read_csv(base_manifest, usecols=["source_row"])["source_row"].astype(int)
        )
        for source_row, (_, peaks) in enumerate(iter_mgf(args.source_mgf)):
            if source_row in needed_rows:
                secondary_by_source_row[source_row] = secondary_spectrum_hash(peaks)
            if source_row and source_row % 200_000 == 0:
                print(
                    f"secondary-hash source rows={source_row:,} "
                    f"recovered={len(secondary_by_source_row):,}", flush=True,
                )
        if len(secondary_by_source_row) != len(needed_rows):
            raise RuntimeError(
                "failed to reconstruct every legacy secondary spectrum hash: "
                f"{len(secondary_by_source_row)} != {len(needed_rows)}"
            )

    old_loaded_paths = {
        Path(str(row.get("path", ""))).as_posix()
        for row in report.get("exclusion_sources", [])
        if row.get("status") == "loaded"
    }
    identities: set[str] = set()
    formulas: set[str] = set()
    hashes = {"primary_sha256": set(), "secondary_blake2b": set()}
    sources: list[dict] = []
    added_names: list[str] = []
    loaders = {"hdf5": load_hdf5, "csv": load_csv, "mgf": load_mgf}
    for source in load_exclusion_registry(args.exclusion_registry):
        name = str(source["name"])
        kind = str(source.get("kind", ""))
        path = Path(str(source.get("path", "")))
        if kind not in loaders:
            raise RuntimeError(f"unsupported exclusion source kind for {name}: {kind}")
        if not path.is_file():
            raise RuntimeError(f"required exclusion source missing: {name} -> {path}")
        current = loaders[kind](path, identities, formulas, hashes)
        if path.as_posix() not in old_loaded_paths:
            added_names.append(name)
        current.update({"name": name, "kind": kind, "required": True})
        sources.append(current)

    args.out.mkdir(parents=True)
    output_manifest = args.out / "eligible_records.csv.gz"
    totals = {
        "rows": 0, "identity": 0, "formula": 0, "spectrum": 0,
        "old_identity": 0, "old_formula": 0, "old_spectrum": 0,
    }
    first = True
    last_source_row = -1
    for chunk in pd.read_csv(base_manifest, chunksize=args.chunk_size, low_memory=False):
        required = {
            "ik14", "formula", "spectrum_hash", "source_row",
        }
        if not required.issubset(chunk.columns):
            raise RuntimeError(f"base manifest lacks columns: {sorted(required-set(chunk.columns))}")
        expected_row = np.arange(totals["rows"], totals["rows"] + len(chunk), dtype=np.int64)
        if not np.array_equal(chunk["row"].to_numpy(np.int64), expected_row):
            raise RuntimeError("base manifest row registry is not contiguous")
        source_rows = chunk["source_row"].to_numpy(np.int64)
        if len(source_rows) and (source_rows[0] <= last_source_row or np.any(np.diff(source_rows) <= 0)):
            raise RuntimeError("base manifest source rows are not strictly increasing")
        if len(source_rows):
            last_source_row = int(source_rows[-1])
        if not chunk["ik14"].fillna("").astype(str).str.fullmatch(r"[A-Z]{14}").all():
            raise RuntimeError("base manifest contains malformed ik14 values")
        if not chunk["spectrum_hash"].fillna("").astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise RuntimeError("base manifest contains malformed primary spectrum hashes")
        if reconstruct_secondary:
            chunk["spectrum_hash_secondary"] = chunk["source_row"].astype(int).map(secondary_by_source_row)
        if not chunk["spectrum_hash_secondary"].fillna("").astype(str).str.fullmatch(r"[0-9a-f]{32}").all():
            raise RuntimeError("base manifest contains malformed secondary spectrum hashes")
        for column, key in (
            ("consumed_identity_overlap", "old_identity"),
            ("consumed_formula_overlap", "old_formula"),
            ("consumed_spectrum_overlap", "old_spectrum"),
        ):
            if column in chunk:
                totals[key] += int(as_bool(chunk[column]).sum())
        identity = chunk["ik14"].fillna("").astype(str).isin(identities)
        formula = chunk["formula"].fillna("").astype(str).isin(formulas)
        spectrum = (
            chunk["spectrum_hash"].fillna("").astype(str).isin(hashes["primary_sha256"])
            | chunk["spectrum_hash_secondary"].fillna("").astype(str).isin(hashes["secondary_blake2b"])
        )
        chunk["consumed_identity_overlap"] = identity
        chunk["consumed_formula_overlap"] = formula
        chunk["consumed_spectrum_overlap"] = spectrum
        totals["rows"] += len(chunk)
        totals["identity"] += int(identity.sum())
        totals["formula"] += int(formula.sum())
        totals["spectrum"] += int(spectrum.sum())
        chunk.to_csv(
            output_manifest, mode="wt" if first else "at", index=False,
            header=first, compression="gzip",
        )
        first = False
        print(f"upgraded rows={totals['rows']:,}", flush=True)

    old_counts = report.get("counts", {})
    expected_rows = int(old_counts.get("eligible_metadata_rows", -1))
    if totals["rows"] != expected_rows:
        raise RuntimeError(
            f"base manifest semantic row-count mismatch: {totals['rows']} != {expected_rows}"
        )
    base_columns_set = set(base_columns)
    for key, total_key in (
        ("consumed_identity_overlap", "old_identity"),
        ("consumed_formula_overlap", "old_formula"),
        ("consumed_spectrum_overlap", "old_spectrum"),
    ):
        if key in base_columns_set and key in old_counts and int(old_counts[key]) != totals[total_key]:
            raise RuntimeError(
                f"base manifest semantic {key} mismatch: "
                f"{totals[total_key]} != {old_counts[key]}"
            )

    output_conflicts = args.out / "conflicting_spectrum_hashes.txt.gz"
    shutil.copyfile(base_conflicts, output_conflicts)
    counts = dict(report.get("counts", {}))
    counts.update({
        "eligible_metadata_rows": totals["rows"],
        "consumed_identity_overlap": totals["identity"],
        "consumed_formula_overlap": totals["formula"],
        "consumed_spectrum_overlap": totals["spectrum"],
    })
    report.update({
        "status": "ENVEDA180_SCOREBLIND_MANIFEST_COMPLETE",
        "schema": "enveda180_scoreblind_manifest_v2_incremental",
        "performance_scores_opened": False,
        "model_loaded": False,
        "exclusion_policy_complete": True,
        "exclusion_registry": str(args.exclusion_registry),
        "exclusion_registry_sha256": sha256_file(args.exclusion_registry),
        "exclusion_sources": sources,
        "incremental_upgrade": {
            "base_audit": str(args.base_audit),
            "base_report_sha256": sha256_file(base_report_path),
            "base_manifest_declared_sha256": declared_manifest_sha256,
            "base_manifest_actual_sha256": actual_manifest_sha256,
            "base_manifest_byte_hash_status": manifest_byte_hash_status,
            "base_manifest_semantic_contract_pass": True,
            "base_conflicts_declared_sha256": declared_conflicts_sha256,
            "base_conflicts_actual_sha256": actual_conflicts_sha256,
            "base_conflicts_byte_hash_status": conflicts_byte_hash_status,
            "base_conflicts_semantic_contract_pass": True,
            "source_mgf_binding": source_binding if reconstruct_secondary else "not_needed",
            "secondary_spectrum_hashes": (
                "reconstructed_from_source_mgf_without_structure_parsing"
                if reconstruct_secondary else "reused_from_base_manifest"
            ),
            "added_sources": added_names,
            "structure_and_primary_spectrum_hashes_reused": True,
            "all_current_exclusion_sources_reloaded": True,
        },
        "counts": counts,
        "manifest": str(output_manifest),
        "manifest_sha256": sha256_file(output_manifest),
        "conflicting_hashes": str(output_conflicts),
        "conflicting_hashes_sha256": sha256_file(output_conflicts),
    })
    report["consumed_identity_count"] = len(identities)
    report["consumed_formula_count"] = len(formulas)
    report["consumed_spectrum_hash_count"] = {
        key: len(value) for key, value in hashes.items()
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "rows": totals["rows"],
        "added_sources": added_names, "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
