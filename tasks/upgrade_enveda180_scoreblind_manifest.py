"""Upgrade a complete Enveda audit by adding newly registered exclusions.

The expensive structure parsing and spectrum hashing are reused byte-for-byte
from the earlier score-blind manifest.  Only exclusion sources absent from that
manifest are loaded, and their identity/formula/hash membership is ORed into
the existing per-record flags.  No model or evaluation truth is accessed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_enveda180_scoreblind_manifest import (
    load_csv,
    load_exclusion_registry,
    sha256_file,
)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-audit", type=Path, required=True)
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
    declared_manifest_sha256 = str(report.get("manifest_sha256", ""))
    actual_manifest_sha256 = sha256_file(base_manifest)
    manifest_byte_hash_match = actual_manifest_sha256 == declared_manifest_sha256
    if sha256_file(base_conflicts) != report.get("conflicting_hashes_sha256"):
        raise RuntimeError("base Enveda conflict ledger hash mismatch")

    old_by_path = {
        Path(str(row.get("path", ""))).as_posix(): row
        for row in report.get("exclusion_sources", [])
    }
    identities: set[str] = set()
    formulas: set[str] = set()
    hashes = {"primary_sha256": set(), "secondary_blake2b": set()}
    sources: list[dict] = []
    added_names: list[str] = []
    for source in load_exclusion_registry(args.exclusion_registry):
        name = str(source["name"])
        kind = str(source.get("kind", ""))
        path = Path(str(source.get("path", "")))
        old = old_by_path.get(path.as_posix())
        if old is not None and old.get("status") == "loaded":
            current = dict(old)
        else:
            if kind != "csv":
                raise RuntimeError(
                    f"base audit lacks non-CSV source {name}; full rebuild required"
                )
            if not path.is_file():
                raise RuntimeError(f"required added exclusion source missing: {name} -> {path}")
            current = load_csv(path, identities, formulas, hashes)
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
            "ik14", "formula", "spectrum_hash", "spectrum_hash_secondary",
            "consumed_identity_overlap", "consumed_formula_overlap", "consumed_spectrum_overlap",
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
        if not chunk["spectrum_hash_secondary"].fillna("").astype(str).str.fullmatch(r"[0-9a-f]{32}").all():
            raise RuntimeError("base manifest contains malformed secondary spectrum hashes")
        old_identity = as_bool(chunk["consumed_identity_overlap"])
        old_formula = as_bool(chunk["consumed_formula_overlap"])
        old_spectrum = as_bool(chunk["consumed_spectrum_overlap"])
        totals["old_identity"] += int(old_identity.sum())
        totals["old_formula"] += int(old_formula.sum())
        totals["old_spectrum"] += int(old_spectrum.sum())
        identity = old_identity | chunk["ik14"].fillna("").astype(str).isin(identities)
        formula = old_formula | chunk["formula"].fillna("").astype(str).isin(formulas)
        spectrum = (
            old_spectrum
            | chunk["spectrum_hash"].fillna("").astype(str).isin(hashes["primary_sha256"])
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
    for key, total_key in (
        ("consumed_identity_overlap", "old_identity"),
        ("consumed_formula_overlap", "old_formula"),
        ("consumed_spectrum_overlap", "old_spectrum"),
    ):
        if key in old_counts and int(old_counts[key]) != totals[total_key]:
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
            "base_manifest_byte_hash_match": manifest_byte_hash_match,
            "base_manifest_semantic_contract_pass": True,
            "added_sources": added_names,
            "structure_and_spectrum_hashes_reused": True,
        },
        "counts": counts,
        "manifest": str(output_manifest),
        "manifest_sha256": sha256_file(output_manifest),
        "conflicting_hashes": str(output_conflicts),
        "conflicting_hashes_sha256": sha256_file(output_conflicts),
    })
    # Exact row-level exclusions are recomputed; union cardinalities are not
    # guessed from overlapping source registries.
    report.pop("consumed_identity_count", None)
    report.pop("consumed_formula_count", None)
    report.pop("consumed_spectrum_hash_count", None)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "rows": totals["rows"],
        "added_sources": added_names, "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
