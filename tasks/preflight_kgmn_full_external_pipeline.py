#!/usr/bin/env python3
"""Validate every manually uploaded dependency before the full KGMN run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SOURCE_COMMIT = "5685ab219269c2f35cd5087655b0470b2da4d93c"
SHA256_FILES = {
    "data/models/MassSpecGym_MurckoHist_split.hdf5": "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
    "data/e1/official_embedding_slim.pt": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    "dreams/models/pretrained/ssl_model_server.pt": "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
    "data/validation/kgmn_dreams_edge_calibration_manifest_20260831/report.json": "dd2b2914d6c3a50187fd1976e17ed8e8e873a2cd6a8c6e7829c1d2f6b5ff1968",
    "data/validation/kgmn_dreams_edge_calibration_manifest_20260831/paired_reaction_decoy_triples.csv.gz": "11d41bfc3c6404fd1d9abfc0a1ce1473f19128c356b195c50eda93a86048c7d2",
}
MD5_FILES = {
    "data/reference/kgmn_zenodo_7089991/Supplementary data1.xlsx": "8eadc3821d6e6973cc81cb3596ef414b",
    "data/reference/kgmn_zenodo_7089991/Supplementary data2.xlsx": "9a047288772908c6bb0d34573bb3b2f8",
    "data/reference/kgmn_zenodo_7089991/Supplementary data3.xlsx": "3e936cbbb22863371213ff8825c9f006",
}
PYTHON_MODULES = ("numpy", "pandas", "scipy", "sklearn", "torch", "h5py", "openpyxl")


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/reference/OEP003284_raw")
    parser.add_argument(
        "--raw-contract", type=Path,
        default=ROOT / "tasks/contracts/kgmn_oep003284_node_files_20260831.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite full-pipeline preflight: {args.output}")

    problems: list[str] = []
    files: dict[str, object] = {}
    for relative, expected in SHA256_FILES.items():
        path = ROOT / relative
        observed = digest(path, "sha256") if path.is_file() else None
        files[relative] = {"algorithm": "sha256", "expected": expected, "observed": observed, "pass": observed == expected}
        if observed != expected:
            problems.append(f"missing or SHA256-mismatched dependency: {relative}")
    for relative, expected in MD5_FILES.items():
        path = ROOT / relative
        observed = digest(path, "md5") if path.is_file() else None
        files[relative] = {"algorithm": "md5", "expected": expected, "observed": observed, "pass": observed == expected}
        if observed != expected:
            problems.append(f"missing or MD5-mismatched dependency: {relative}")

    source_dir = ROOT / "third_party/MetDNA2"
    try:
        commit = git_output("-C", str(source_dir), "rev-parse", "HEAD")
        dirty = git_output("-C", str(source_dir), "status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "", "unavailable"
    source_pass = commit == EXPECTED_SOURCE_COMMIT and dirty == ""
    if not source_pass:
        problems.append("third_party/MetDNA2 is missing, dirty, or at the wrong commit")
    required_extdata = (
        "peak_table_200STD_neg_200805.csv", "spectra_200STD_neg_200805.msp",
        "peak_table_annotated_200STD_neg_200805.csv", "annotation_initial.csv", "GenForm",
    )
    extdata_missing = [name for name in required_extdata if not (source_dir / "inst/extdata" / name).is_file()]
    if extdata_missing:
        problems.append(f"MetDNA2 bundled 200STD dependencies missing: {extdata_missing}")

    raw_rows: list[dict[str, object]] = []
    if not args.raw_contract.is_file():
        problems.append(f"missing frozen NODE raw contract: {args.raw_contract}")
    else:
        with args.raw_contract.open(encoding="utf-8", newline="") as handle:
            expected_raw = list(csv.DictReader(handle))
        if len(expected_raw) != 24:
            problems.append("frozen NODE contract does not contain exactly 24 files")
        for row in expected_raw:
            path = args.raw_root / row["name"]
            expected_bytes = int(row["bytes"])
            size_ok = path.is_file() and path.stat().st_size == expected_bytes
            observed_md5 = digest(path, "md5") if size_ok else None
            passed = size_ok and observed_md5 == row["md5"]
            raw_rows.append({
                "name": row["name"], "expected_bytes": expected_bytes,
                "expected_md5": row["md5"], "observed_md5": observed_md5, "pass": passed,
            })
            if not passed:
                problems.append(f"missing, truncated or MD5-mismatched raw file: {row['name']}")
        if args.raw_root.is_dir():
            observed_names = {path.name for path in args.raw_root.glob("*.mzXML")}
            expected_names = {row["name"] for row in expected_raw}
            extras = sorted(observed_names - expected_names)
            if extras:
                problems.append(f"unexpected mzXML files in raw root: {extras}")

    modules = {name: importlib.util.find_spec(name) is not None for name in PYTHON_MODULES}
    missing_modules = [name for name, present in modules.items() if not present]
    if missing_modules:
        problems.append(f"missing Python runtime modules: {missing_modules}")
    executables = {name: shutil.which(name) for name in ("Rscript", "git", "sha256sum")}
    missing_executables = [name for name, path in executables.items() if path is None]
    if missing_executables:
        problems.append(f"missing runtime executables: {missing_executables}")

    ready = not problems
    report = {
        "status": "kgmn_full_external_pipeline_preflight_ready" if ready else "kgmn_full_external_pipeline_preflight_failed",
        "formal": True, "ready": ready, "problems": problems,
        "manual_dependencies": files,
        "metdna2": {
            "expected_commit": EXPECTED_SOURCE_COMMIT, "observed_commit": commit,
            "clean": dirty == "", "required_extdata_missing": extdata_missing, "pass": source_pass and not extdata_missing,
        },
        "oep003284_raw": {
            "root": str(args.raw_root), "files_expected": 24,
            "files_passed": sum(bool(row["pass"]) for row in raw_rows),
            "bytes_expected": sum(int(row["expected_bytes"]) for row in raw_rows),
            "files": raw_rows,
        },
        "runtime": {"python_modules": modules, "executables": executables},
        "contracts": {
            "all_dependencies_checked_before_any_training_or_evaluation": True,
            "credentials_stored": False, "P2b_used": False, "phenotype_used": False,
        },
        "claim_limit": "Execution-readiness audit only; no annotation result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.require_ready and not ready:
        raise RuntimeError("full external pipeline dependency preflight failed")


if __name__ == "__main__":
    main()
