#!/usr/bin/env python3
"""Inventory OEP003284 and fail closed unless both KGMN polarities are runnable.

This stage is deliberately outcome-free.  It checks only author workflow inputs:
an MS1 feature table, sample metadata, and a homogeneous MetDNA2-supported MS2 set for
each polarity.  Supplementary annotation tables are not accepted as inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SUPPORTED_MS2_SUFFIXES = {".mgf", ".msp", ".mzxml", ".cef"}
DREAMS_EDGE_MS2_SUFFIXES = {".mgf", ".msp"}
RAW_ONLY_SUFFIXES = {".raw", ".wiff", ".scan", ".mzml"}
FORBIDDEN_MS1_COLUMNS = {
    "annotation",
    "candidate",
    "compound",
    "compound_id",
    "confidence_level",
    "id_kegg",
    "inchikey",
    "inchikey1",
    "truth",
    "validation_standard",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def find_polarity_dir(root: Path, polarity: str) -> Path | None:
    aliases = {
        "positive": {"positive", "pos", "pos_hilic", "hilic_pos"},
        "negative": {"negative", "neg", "neg_hilic", "hilic_neg"},
    }[polarity]
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.lower() in aliases]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous {polarity} directories: {[str(path) for path in matches]}")
    return matches[0] if matches else None


def select_named_file(directory: Path, names: set[str]) -> Path | None:
    matches = [path for path in directory.iterdir() if path.is_file() and path.name.lower() in names]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous files in {directory}: {[path.name for path in matches]}")
    return matches[0] if matches else None


def inspect_polarity(root: Path, polarity: str, allow_author_mapped_mzxml: bool = False) -> dict:
    directory = find_polarity_dir(root, polarity)
    if directory is None:
        return {"status": "missing_polarity_directory", "polarity": polarity}

    ms1 = select_named_file(directory, {"data.csv", "peak_table.csv"})
    sample_info = select_named_file(directory, {"sample.info.csv", "sample_info.csv"})
    ms2 = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MS2_SUFFIXES
    )
    raw_only = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in RAW_ONLY_SUFFIXES
    )

    problems: list[str] = []
    if ms1 is None:
        problems.append("missing MS1 data.csv/peak_table.csv")
    if sample_info is None:
        problems.append("missing sample.info.csv")
    if len(ms2) < 1:
        problems.append("expected at least one supported MS2 file")
    ms2_suffixes = sorted({path.suffix.lower() for path in ms2})
    if len(ms2_suffixes) > 1:
        problems.append(f"mixed MS2 formats are forbidden: {ms2_suffixes}")

    ms1_header: list[str] = []
    sample_header: list[str] = []
    if ms1 is not None:
        ms1_header = read_csv_header(ms1)
        if [value.strip().lower() for value in ms1_header[:3]] != ["name", "mz", "rt"]:
            problems.append("MS1 table must start with name,mz,rt")
        forbidden = sorted({value.strip().lower() for value in ms1_header} & FORBIDDEN_MS1_COLUMNS)
        if forbidden:
            problems.append(f"MS1 table contains forbidden annotation columns: {forbidden}")
    if sample_info is not None:
        sample_header = read_csv_header(sample_info)
        if [value.strip().lower() for value in sample_header[:2]] != ["sample.name", "group"]:
            problems.append("sample metadata must start with sample.name,group")
    if ms1_header and sample_info is not None:
        with sample_info.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        sample_column = next((name for name in sample_header if name.strip().lower() == "sample.name"), None)
        sample_names = [str(row.get(sample_column, "")).strip() for row in rows] if sample_column else []
        if not sample_names or any(not value for value in sample_names) or len(set(sample_names)) != len(sample_names):
            problems.append("sample metadata has empty or duplicate sample names")
        elif set(sample_names) != set(ms1_header[3:]):
            problems.append("MS1 abundance columns do not exactly match sample metadata")

    status = "ready" if not problems else ("raw_only" if raw_only and not ms2 else "incomplete")
    direct_dreams_edge = (
        status == "ready"
        and len(ms2_suffixes) == 1
        and ms2_suffixes[0] in DREAMS_EDGE_MS2_SUFFIXES
    )
    mapped_mzxml_edge = (
        status == "ready"
        and allow_author_mapped_mzxml
        and ms2_suffixes == [".mzxml"]
    )
    dreams_edge_ready = direct_dreams_edge or mapped_mzxml_edge
    if direct_dreams_edge:
        dreams_edge_mode = "direct_identifier_preserving_msp_or_mgf"
    elif mapped_mzxml_edge:
        dreams_edge_mode = "metdna2_author_mapped_mzxml_cache"
    else:
        dreams_edge_mode = None
    files = []
    for role, path in (("ms1", ms1), ("sample_info", sample_info)):
        if path is not None:
            files.append({"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
    for path in ms2:
        files.append({"role": "ms2", "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
    return {
        "status": status,
        "polarity": polarity,
        "directory": str(directory),
        "files": files,
        "raw_only_files": [str(path) for path in raw_only],
        "problems": problems,
        "ms2_file_count": len(ms2),
        "ms2_suffixes": ms2_suffixes,
        "dreams_edge_ready": dreams_edge_ready,
        "dreams_edge_mode": dreams_edge_mode,
        "dreams_edge_problem": (
            None
            if dreams_edge_ready
            else (
                "DreaMS edge arms require identifier-preserving MSP/MGF, or the explicit "
                "MetDNA2 author-mapped mzXML cache bridge"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/reference/OEP003284"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/kgmn_oep003284_input_preflight_20260831.json"),
    )
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--require-dreams-edge", action="store_true")
    parser.add_argument(
        "--allow-author-mapped-mzxml",
        action="store_true",
        help=(
            "Permit homogeneous mzXML sets only when the execution workflow exports the "
            "feature-mapped MetDNA2 initial-seed MS2 cache before DreaMS encoding."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_root.is_dir():
        report = {
            "status": "kgmn_oep003284_inputs_missing",
            "formal": True,
            "input_root": str(args.input_root),
            "ready": False,
            "dreams_edge_ready": False,
            "panels": {},
            "claim_limit": "Input inventory only; no annotation outcome or performance claim.",
        }
    else:
        panels = {
            polarity: inspect_polarity(
                args.input_root,
                polarity,
                allow_author_mapped_mzxml=args.allow_author_mapped_mzxml,
            )
            for polarity in ("positive", "negative")
        }
        ready = all(panel["status"] == "ready" for panel in panels.values())
        dreams_edge_ready = ready and all(panel["dreams_edge_ready"] for panel in panels.values())
        report = {
            "status": "kgmn_oep003284_inputs_ready" if ready else "kgmn_oep003284_inputs_incomplete",
            "formal": True,
            "input_root": str(args.input_root.resolve()),
            "ready": ready,
            "dreams_edge_ready": dreams_edge_ready,
            "panels": panels,
            "accepted_ms2_formats": sorted(SUPPORTED_MS2_SUFFIXES),
            "dreams_edge_ms2_formats": sorted(DREAMS_EDGE_MS2_SUFFIXES),
            "author_mapped_mzxml_bridge_enabled": args.allow_author_mapped_mzxml,
            "forbidden_input_semantics": "truth, candidate identities, validation labels, and final author annotations",
            "claim_limit": "Input inventory only; no annotation outcome or performance claim.",
        }

    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing preflight report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.require_ready and not report["ready"]:
        raise RuntimeError("OEP003284 is not ready for KGMN hidden-seed replay; see preflight report")
    if args.require_dreams_edge and not report["dreams_edge_ready"]:
        raise RuntimeError(
            "OEP003284 is not ready for DreaMS edge arms; use identifier-preserving MSP/MGF "
            "or explicitly enable the MetDNA2 author-mapped mzXML cache bridge"
        )


if __name__ == "__main__":
    main()
