"""Audit whether local MSnLib MS2 exports can support a future external benchmark.

This is a metadata/readiness audit only.  It never runs a model, assigns a
train/test split, or authorizes training.  One JSON line is one exported,
usually merged, library spectrum; it is not one raw scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
SUPPORTED_ADDUCTS = {"[M+H]+", "[M+Na]+", "[M-H]-"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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
    if ".mzML" in feature:
        return feature.split(".mzML", 1)[0]
    return feature


def acquisition_date(row: dict[str, Any], raw_id: str) -> str:
    fields: Iterable[Any] = (
        raw_id,
        row.get("raw_file_name"),
        row.get("source_scan_usi"),
        row.get("usi"),
        row.get("feature_id"),
        row.get("featurelist_feature_id"),
    )
    for value in fields:
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            match = DATE_RE.search("" if item is None else str(item))
            if match:
                return match.group(1)
    return ""


def source_name(path: Path) -> tuple[str, str, str]:
    match = re.match(r"(\d{8})_(.+)_(pos|neg)_ms2\.json$", path.name)
    if not match:
        raise ValueError(f"unexpected MSnLib filename: {path.name}")
    release, library, polarity = match.groups()
    return release, library, polarity


def load_massspecgym_ik14(path: Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        raw = {decode(value) for value in handle["INCHIKEY"][:]}
    values: set[str] = set()
    for value in raw:
        if len(value) == 14:
            values.add(value)
        elif valid_full_ik(value):
            values.add(ik14(value))
        else:
            raise RuntimeError(f"unexpected MassSpecGym INCHIKEY value: {value!r}")
    return values


def load_mona_ik14(path: Path) -> set[str]:
    values: set[str] = set()
    with h5py.File(path, "r") as handle:
        for raw in handle["data"]:
            row = json.loads(decode(raw))
            value = str(row[1])
            if len(value) == 14:
                values.add(value)
    return values


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
        "--massspecgym",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--mona-pos", type=Path, default=ROOT / "data/models/mona_pos_full.hdf5",
    )
    parser.add_argument(
        "--mona-neg", type=Path, default=ROOT / "data/models/mona_neg_full.hdf5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_msnlib_external_readiness_v2/report.json",
    )
    parser.add_argument(
        "--post-gems-after",
        default="20221130",
        help="Conservative acquisition-date boundary, exclusive (YYYYMMDD).",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    msg_ik14 = load_massspecgym_ik14(args.massspecgym)
    mona_ik14 = load_mona_ik14(args.mona_pos) | load_mona_ik14(args.mona_neg)

    per_file: list[dict[str, Any]] = []
    date_counts = Counter()
    adduct_counts = Counter()
    merge_counts = Counter()
    valid_full: set[str] = set()
    valid_ik14: set[str] = set()
    formulas: set[str] = set()
    readiness: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"records": 0, "raw_files": set(), "precursor_mz": [], "formulas": set()}
    )
    input_hashes: dict[str, str] = {}
    invalid_json_examples: list[dict[str, Any]] = []

    files = sorted(args.msnlib_dir.glob("*_ms2.json"))
    if not files:
        raise FileNotFoundError(f"no *_ms2.json files under {args.msnlib_dir}")
    for path in files:
        release, library, filename_polarity = source_name(path)
        counters = Counter()
        file_full: set[str] = set()
        file_ik14: set[str] = set()
        file_dates = Counter()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                counters["nonempty_lines"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    counters["invalid_json_lines"] += 1
                    if len(invalid_json_examples) < 20:
                        invalid_json_examples.append(
                            {
                                "file": path.name,
                                "line": line_number,
                                "error": str(error),
                                "prefix": line[:120],
                            }
                        )
                    continue
                counters["records"] += 1
                full = str(row.get("inchikey") or "").strip()
                connectivity = ik14(full)
                formula = str(row.get("formula") or "").strip()
                adduct = str(row.get("adduct") or "").strip()
                raw_id = raw_file_id(row)
                date = acquisition_date(row, raw_id)
                peaks = row.get("peaks") or []
                num_peaks = int(row.get("num_peaks") or len(peaks))
                quality = str(row.get("quality_chimeric") or "")
                precursor_mz = row.get("precursor_mz")
                merge_type = str(row.get("merge_type") or "missing")

                counters[f"ms_level::{row.get('ms_level', 'missing')}"] += 1
                counters[f"quality_chimeric::{quality or 'missing'}"] += 1
                counters["valid_full_inchikey"] += int(bool(connectivity))
                counters["known_acquisition_date"] += int(bool(date))
                counters["post_gems_acquisition"] += int(bool(date and date > args.post_gems_after))
                counters["supported_adduct"] += int(adduct in SUPPORTED_ADDUCTS)
                counters["at_least_5_peaks"] += int(num_peaks >= 5)
                counters["passed_chimeric"] += int(quality == "PASSED")
                if date:
                    file_dates[date] += 1
                    date_counts[date] += 1
                adduct_counts[adduct or "missing"] += 1
                merge_counts[merge_type] += 1
                if connectivity:
                    valid_full.add(full)
                    valid_ik14.add(connectivity)
                    file_full.add(full)
                    file_ik14.add(connectivity)
                if formula:
                    formulas.add(formula)

                ready = (
                    bool(connectivity)
                    and bool(date and date > args.post_gems_after)
                    and adduct in SUPPORTED_ADDUCTS
                    and num_peaks >= 5
                    and quality == "PASSED"
                    and connectivity not in msg_ik14
                    and connectivity not in mona_ik14
                    and precursor_mz is not None
                )
                if ready:
                    item = readiness[(connectivity, adduct)]
                    item["records"] += 1
                    if raw_id:
                        item["raw_files"].add(raw_id)
                    item["precursor_mz"].append(float(precursor_mz))
                    if formula:
                        item["formulas"].add(formula)
        per_file.append(
            {
                "file": path.name,
                "release": release,
                "library": library,
                "filename_polarity": filename_polarity,
                "bytes": path.stat().st_size,
                "nonempty_lines": counters["nonempty_lines"],
                "invalid_json_lines": counters["invalid_json_lines"],
                "records": counters["records"],
                "unique_full_inchikey": len(file_full),
                "unique_ik14": len(file_ik14),
                "known_acquisition_date_records": counters["known_acquisition_date"],
                "post_gems_acquisition_records": counters["post_gems_acquisition"],
                "date_counts": dict(sorted(file_dates.items())),
                "valid_full_inchikey_records": counters["valid_full_inchikey"],
                "supported_adduct_records": counters["supported_adduct"],
                "at_least_5_peaks_records": counters["at_least_5_peaks"],
                "passed_chimeric_records": counters["passed_chimeric"],
                "ms_level_counts": {
                    key.split("::", 1)[1]: value
                    for key, value in sorted(counters.items())
                    if key.startswith("ms_level::")
                },
                "quality_chimeric_counts": {
                    key.split("::", 1)[1]: value
                    for key, value in sorted(counters.items())
                    if key.startswith("quality_chimeric::")
                },
            }
        )
        input_hashes[path.name] = sha256(path)

    total_records = sum(row["records"] for row in per_file)
    total_lines = sum(row["nonempty_lines"] for row in per_file)
    total_invalid = sum(row["invalid_json_lines"] for row in per_file)
    post_records = sum(row["post_gems_acquisition_records"] for row in per_file)
    repeated = {
        key: value
        for key, value in readiness.items()
        if value["records"] >= 2 and len(value["raw_files"]) >= 2
    }
    repeated_identities = {key[0] for key in repeated}
    repeated_formulas = {formula for value in repeated.values() for formula in value["formulas"]}
    repeated_record_counts = [int(value["records"]) for value in repeated.values()]
    repeated_raw_counts = [len(value["raw_files"]) for value in repeated.values()]

    report = {
        "status": "chemaware_msnlib_external_readiness_audited",
        "formal_training_authorized": False,
        "model_scores_read": False,
        "unit_of_count": (
            "One JSON line is one exported MS2/pseudo-MS2 library spectrum, often merged "
            "from multiple source scans; it is not one raw scan."
        ),
        "scope": {
            "files": len(files),
            "bytes": sum(row["bytes"] for row in per_file),
            "nonempty_lines": total_lines,
            "valid_json_spectrum_records": total_records,
            "invalid_json_lines": total_invalid,
            "invalid_json_examples": invalid_json_examples,
            "unique_full_inchikey": len(valid_full),
            "unique_ik14": len(valid_ik14),
            "unique_formula": len(formulas),
            "known_acquisition_date_records": sum(
                row["known_acquisition_date_records"] for row in per_file
            ),
            "post_gems_acquisition_records": post_records,
            "post_gems_boundary_exclusive": args.post_gems_after,
        },
        "identity_overlap": {
            "massspecgym_identity_field_granularity": "IK14 connectivity block",
            "massspecgym_ik14": len(valid_ik14 & msg_ik14),
            "mona_ik14": len(valid_ik14 & mona_ik14),
            "ik14_disjoint_from_massspecgym_and_mona": len(valid_ik14 - msg_ik14 - mona_ik14),
        },
        "future_external_pairing_eligible": {
            "definition": (
                "post-boundary acquisition; valid full InChIKey; supported adduct; >=5 peaks; "
                "quality_chimeric PASSED; IK14 absent from MassSpecGym and local MoNA; and at "
                "least two exported spectra from distinct raw-file identifiers for the same "
                "IK14/adduct. No final split has been assigned."
            ),
            "identity_adduct_groups": len(repeated),
            "unique_ik14": len(repeated_identities),
            "unique_formula": len(repeated_formulas),
            "records_per_identity_adduct": describe(repeated_record_counts),
            "raw_files_per_identity_adduct": describe(repeated_raw_counts),
            "adduct_groups": dict(sorted(Counter(key[1] for key in repeated).items())),
        },
        "adduct_record_counts": dict(sorted(adduct_counts.items())),
        "merge_type_record_counts": dict(sorted(merge_counts.items())),
        "acquisition_date_record_counts": dict(sorted(date_counts.items())),
        "per_file": per_file,
        "interpretation": {
            "independence_label": "future preregistered external stress-test candidate",
            "not_yet_claimable": [
                "fully independent benchmark",
                "external generalization performance",
                "chemical-awareness performance",
            ],
            "reasons": [
                "MSnLib is accession MSV000094528, which the official DreaMS paper already used for a property-evaluation analysis.",
                "Temporal and identity disjointness do not by themselves prove spectrum-hash or molecular-scaffold disjointness.",
                "A query/reference split and mass-window candidate manifest have not yet been frozen.",
            ],
        },
        "next_required_gates": [
            "freeze query/reference by distinct raw acquisition before any model scoring",
            "exclude rounded-spectrum-hash overlap across query/reference and internal corpora",
            "freeze exact mass/adduct candidate generation and tie handling",
            "report identity-equal and formula-clustered uncertainty",
            "separate same-formula, near-structure, and easy-negative strata",
        ],
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "massspecgym_sha256": sha256(args.massspecgym),
            "mona_pos_sha256": sha256(args.mona_pos),
            "mona_neg_sha256": sha256(args.mona_neg),
            "input_sha256": input_hashes,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
