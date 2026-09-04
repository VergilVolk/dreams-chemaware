#!/usr/bin/env python
"""Preflight the frozen MetDNA3 HILIC development MS2 files.

This stage answers only whether real targeted-MS2 spectra can be paired with
the already-opened NIST-urine HILIC Level-1 development identities under the
published 15 ppm / 25 second windows.  It does not encode spectra, rank
candidates, use the reaction graph, or touch RP/external-test outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from metdna3_mzml import iter_spectrum_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def quantiles(values: list[float]) -> dict[str, float | None]:
    x = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(x):
        return {key: None for key in ("min", "median", "p90", "max")}
    return {
        "min": float(x.min()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(x.max()),
    }


def audit_file(path: Path) -> tuple[dict, list[dict]]:
    ms2: list[dict] = []
    n_ms1 = 0
    for row in iter_spectrum_metadata(path):
        if row["ms_level"] == 1:
            n_ms1 += 1
        elif row["ms_level"] == 2:
            ms2.append(row)
    valid = [
        row
        for row in ms2
        if math.isfinite(row["precursor_mz"])
        and math.isfinite(row["rt_sec"])
        and row["n_peaks"] >= 2
    ]
    for row in valid:
        row["source_file"] = path.name
        row["spectrum_key"] = f"{path.name}|{row['spectrum_id']}"
    report = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "n_ms1": n_ms1,
        "n_ms2": len(ms2),
        "n_usable_ms2": len(valid),
        "missing_precursor": sum(not math.isfinite(row["precursor_mz"]) for row in ms2),
        "missing_rt": sum(not math.isfinite(row["rt_sec"]) for row in ms2),
        "empty_ms2": sum(row["n_peaks"] < 2 for row in ms2),
        "precursor_mz": quantiles([row["precursor_mz"] for row in valid]),
        "rt_sec": quantiles([row["rt_sec"] for row in valid]),
        "peak_count": quantiles([float(row["n_peaks"]) for row in valid]),
    }
    return report, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1"),
    )
    parser.add_argument(
        "--mzml-dir",
        type=Path,
        default=Path("data/external/metdna3_2025/mzml/development"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_development_ms2_preflight.json"),
    )
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--manifest-report", type=Path, default=None)
    parser.add_argument("--expected-files", type=int, default=16)
    parser.add_argument("--minimum-matched-rows", type=int, default=100)
    parser.add_argument("--minimum-matched-identities", type=int, default=100)
    parser.add_argument("--minimum-exclusive-identities", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true", help="local smoke only")
    args = parser.parse_args()

    truth_path = args.development_dir / args.truth_name
    manifest_path = args.development_dir / "download_manifest.json"
    report_path = args.manifest_report or (args.development_dir / "report.json")
    for path in (truth_path, manifest_path, report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    development_report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.scope == "development":
        if development_report.get("internal_validation_opened") is not False:
            raise RuntimeError("development contract does not prove RP remained unopened")
        if development_report.get("external_test_opened") is not False:
            raise RuntimeError("development contract does not prove external test remained unopened")
    elif args.scope == "internal_rplc":
        if development_report.get("status") != "bioaware_metdna3_internal_rplc_manifest_complete":
            raise RuntimeError("invalid internal RPLC manifest report")
        if development_report.get("external_16_panel_outcomes_opened") is not False:
            raise RuntimeError("internal contract does not prove external outcomes remained unopened")
        contracts = development_report.get("contracts", {})
        if contracts.get("router_refit_on_rplc") is not False or contracts.get("threshold_tuning_on_rplc") is not False:
            raise RuntimeError("internal RPLC contract permits model refit or threshold tuning")
    else:
        if development_report.get("status") != "bioaware_metdna3_external_manifest_frozen":
            raise RuntimeError("invalid external 16-panel manifest report")
        contracts = development_report.get("contracts", {})
        if contracts.get("router_refit") is not False or contracts.get("threshold_tuning") is not False:
            raise RuntimeError("external contract permits model refit or threshold tuning")
        if contracts.get("one_shot_external_evaluation") is not True:
            raise RuntimeError("external manifest is not one-shot locked")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    expected = {row["local_name"]: int(row["bytes"]) for row in manifest}
    paths = sorted(args.mzml_dir.glob("*.mzML"))
    if args.allow_partial:
        if not paths:
            raise FileNotFoundError(f"no smoke mzML files in {args.mzml_dir}")
        unexpected = sorted(path.name for path in paths if path.name not in expected)
        if unexpected:
            raise RuntimeError(f"unexpected development files: {unexpected}")
    else:
        present = {path.name for path in paths}
        if present != set(expected):
            raise RuntimeError(
                f"formal preflight requires exactly {args.expected_files} frozen files; "
                f"missing={sorted(set(expected)-present)}, extra={sorted(present-set(expected))}"
            )
    for path in paths:
        if path.stat().st_size != expected[path.name]:
            raise RuntimeError(f"byte-size mismatch: {path}")

    truth = pd.read_csv(truth_path)
    file_reports: list[dict] = []
    spectra_by_polarity: dict[str, list[dict]] = {"positive": [], "negative": []}
    for position, path in enumerate(paths, 1):
        polarity = "positive" if "_pos_" in path.name else "negative"
        report, spectra = audit_file(path)
        report["polarity"] = polarity
        file_reports.append(report)
        spectra_by_polarity[polarity].extend(spectra)
        print(
            f"[preflight {position}/{len(paths)}] {path.name}: "
            f"MS2={report['n_ms2']:,} usable={report['n_usable_ms2']:,}",
            flush=True,
        )

    match_counts: list[int] = []
    matched_identities: set[str] = set()
    matched_rows = 0
    best_ppm: list[float] = []
    best_rt: list[float] = []
    best_assignments: list[dict] = []
    for row in truth.itertuples(index=False):
        matches: list[tuple[float, float]] = []
        mz = float(row.mz)
        rt = float(row.rt)
        for spectrum in spectra_by_polarity[str(row.polarity)]:
            ppm = abs(float(spectrum["precursor_mz"]) - mz) / mz * 1e6
            rt_delta = abs(float(spectrum["rt_sec"]) - rt)
            if ppm <= args.ppm and rt_delta <= args.rt_sec:
                matches.append((ppm, rt_delta))
        match_counts.append(len(matches))
        if matches:
            matched_rows += 1
            matched_identities.add(str(row.ik14))
            best = min(matches, key=lambda item: (item[0] / args.ppm) ** 2 + (item[1] / args.rt_sec) ** 2)
            best_ppm.append(best[0])
            best_rt.append(best[1])
            # Recover the exact spectrum with the same deterministic metric.
            best_spectrum = min(
                (
                    spectrum
                    for spectrum in spectra_by_polarity[str(row.polarity)]
                    if abs(float(spectrum["precursor_mz"]) - mz) / mz * 1e6 <= args.ppm
                    and abs(float(spectrum["rt_sec"]) - rt) <= args.rt_sec
                ),
                key=lambda spectrum: (
                    (abs(float(spectrum["precursor_mz"]) - mz) / mz * 1e6 / args.ppm) ** 2
                    + (abs(float(spectrum["rt_sec"]) - rt) / args.rt_sec) ** 2,
                    spectrum["spectrum_key"],
                ),
            )
            best_assignments.append(
                {
                    "spectrum_key": best_spectrum["spectrum_key"],
                    "ik14": str(row.ik14),
                }
            )

    assignment = pd.DataFrame(best_assignments)
    identities_per_spectrum = (
        assignment.groupby("spectrum_key")["ik14"].nunique()
        if len(assignment) else pd.Series(dtype=int)
    )
    ambiguous_spectra = set(identities_per_spectrum[identities_per_spectrum > 1].index)
    exclusive = assignment[~assignment["spectrum_key"].isin(ambiguous_spectra)]

    formal = not args.allow_partial
    gates = {
        "all_frozen_files_present": len(paths) == args.expected_files,
        "every_file_has_usable_ms2": all(row["n_usable_ms2"] > 0 for row in file_reports),
        "level1_rows_matched_minimum": matched_rows >= args.minimum_matched_rows,
        "level1_identities_matched_minimum": len(matched_identities) >= args.minimum_matched_identities,
        "exclusive_best_match_identities_minimum": int(exclusive["ik14"].nunique()) >= args.minimum_exclusive_identities,
    }
    payload = {
        "status": f"bioaware_metdna3_{args.scope}_ms2_preflight_complete",
        "formal": formal,
        "scope": args.scope,
        "published_matching_window": {"precursor_ppm": args.ppm, "rt_seconds": args.rt_sec},
        "files": file_reports,
        "combined": {
            "n_files": len(paths),
            "n_ms2": sum(row["n_ms2"] for row in file_reports),
            "n_usable_ms2": sum(row["n_usable_ms2"] for row in file_reports),
            "level1_rows": int(len(truth)),
            "matched_level1_rows": matched_rows,
            "matched_level1_identities": len(matched_identities),
            "matches_per_level1_row": quantiles([float(value) for value in match_counts]),
            "best_match_ppm": quantiles(best_ppm),
            "best_match_rt_seconds": quantiles(best_rt),
            "best_spectra_shared_by_multiple_truth_identities": len(ambiguous_spectra),
            "exclusive_best_match_rows": int(len(exclusive)),
            "exclusive_best_match_identities": int(exclusive["ik14"].nunique()),
        },
        "gates": gates,
        "minimums": {
            "matched_rows": args.minimum_matched_rows,
            "matched_identities": args.minimum_matched_identities,
            "exclusive_best_match_identities": args.minimum_exclusive_identities,
        },
        "pass_to_dreams_ranking": bool(formal and all(gates.values())),
        "pass_to_dreams_development_ranking": bool(
            args.scope == "development" and formal and all(gates.values())
        ),
        "provenance": {
            "truth_sha256": sha256(truth_path),
            "download_manifest_sha256": sha256(manifest_path),
            "development_report_sha256": sha256(report_path),
        },
        "claim_limit": (
            "Acquisition and Level-1 pairing preflight only; no DreaMS encoding, "
            "candidate ranking, BioAware gain, or external-test claim."
        ),
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite different preflight: {args.output}")
        print(f"[reverified] {args.output}", flush=True)
    else:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if formal and not payload["pass_to_dreams_ranking"]:
        raise RuntimeError("MetDNA3 MS2 development preflight gates failed")


if __name__ == "__main__":
    main()
