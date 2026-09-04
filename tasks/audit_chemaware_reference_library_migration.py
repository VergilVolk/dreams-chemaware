"""Compare the legacy unified_v2 gallery with the corrected unified_v3 gallery.

Only MGF headers are scanned.  Peak arrays are intentionally ignored because
this audit asks which representative identity/adduct keys survived, which
source won each key, and whether MassSpecGym condition metadata was preserved.
It does not evaluate retrieval quality and does not construct training pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = re.compile(
    rb"(?m)^(INCHIKEY|ADDUCT|SOURCE|INSTRUMENT_TYPE|COLLISION_ENERGY|FOLD|"
    rb"SIMULATION_CHALLENGE)=([^\r\n]*)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "data/reference/unified_v2")
    parser.add_argument("--corrected-dir", type=Path, default=ROOT / "data/reference/unified_v3")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_reference_library_migration_v2_to_v3/report.json",
    )
    return parser.parse_args()


def scan_file(path: Path, polarity: str) -> dict[tuple[str, str, str], dict[str, str]]:
    records: dict[tuple[str, str, str], dict[str, str]] = {}
    current: dict[str, str] = {}

    def commit() -> None:
        if not current:
            return
        missing = {"INCHIKEY", "ADDUCT", "SOURCE"} - set(current)
        if missing:
            raise RuntimeError(f"incomplete header record in {path}: {sorted(missing)}")
        key = (current["INCHIKEY"], polarity, current["ADDUCT"])
        if key in records:
            raise RuntimeError(f"duplicate representative key in {path}: {key}")
        records[key] = dict(current)

    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for match in HEADER.finditer(mapped):
                field = match.group(1).decode("ascii")
                value = match.group(2).decode("utf-8", "replace")
                if field == "INCHIKEY":
                    commit()
                    current = {field: value}
                else:
                    current[field] = value
    commit()
    return records


def scan_library(directory: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    records: dict[tuple[str, str, str], dict[str, str]] = {}
    for polarity, filename in (("pos", "unified_pos.mgf"), ("neg", "unified_neg.mgf")):
        part = scan_file(directory / filename, polarity)
        overlap = set(records) & set(part)
        if overlap:
            raise RuntimeError(f"cross-file duplicate keys: {len(overlap)}")
        records.update(part)
    return records


def finite_collision_energy(value: str) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed and parsed not in (float("inf"), float("-inf"))


def summarize(records: dict[tuple[str, str, str], dict[str, str]]) -> dict:
    identities = {key[0] for key in records}
    ik14 = {value[:14] for value in identities}
    source = Counter(record["SOURCE"] for record in records.values())
    msg = [record for record in records.values() if record["SOURCE"] == "massspecgym"]
    return {
        "representative_keys": len(records),
        "unique_full_inchikey": len(identities),
        "unique_ik14": len(ik14),
        "source_counts": dict(sorted(source.items())),
        "massspecgym_representatives": len(msg),
        "massspecgym_simulation_challenge_membership": dict(sorted(Counter(
            record.get("SIMULATION_CHALLENGE", "missing") or "missing" for record in msg
        ).items())),
        "massspecgym_with_instrument_type": sum(bool(record.get("INSTRUMENT_TYPE")) for record in msg),
        "massspecgym_with_collision_energy_field": sum("COLLISION_ENERGY" in record for record in msg),
        "massspecgym_with_finite_collision_energy": sum(
            finite_collision_energy(record.get("COLLISION_ENERGY", "")) for record in msg
        ),
        "massspecgym_with_fold": sum(bool(record.get("FOLD")) for record in msg),
    }


def main() -> None:
    args = parse_args()
    required = [
        args.legacy_dir / "build_report.json",
        args.legacy_dir / "unified_pos.mgf",
        args.legacy_dir / "unified_neg.mgf",
        args.corrected_dir / "build_report.json",
        args.corrected_dir / "unified_pos.mgf",
        args.corrected_dir / "unified_neg.mgf",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    legacy = scan_library(args.legacy_dir)
    corrected = scan_library(args.corrected_dir)
    legacy_keys = set(legacy)
    corrected_keys = set(corrected)
    common = legacy_keys & corrected_keys
    transitions = Counter(
        f"{legacy[key]['SOURCE']}->{corrected[key]['SOURCE']}" for key in common
    )
    legacy_full = {key[0] for key in legacy}
    corrected_full = {key[0] for key in corrected}
    legacy_ik14 = {value[:14] for value in legacy_full}
    corrected_ik14 = {value[:14] for value in corrected_full}
    legacy_report = json.loads((args.legacy_dir / "build_report.json").read_text(encoding="utf-8"))
    corrected_report = json.loads((args.corrected_dir / "build_report.json").read_text(encoding="utf-8"))
    output = {
        "status": "chemaware_reference_library_migration_audited",
        "training_was_run": False,
        "legacy_v2": summarize(legacy),
        "corrected_v3": summarize(corrected),
        "delta_v3_minus_v2": {
            "representative_keys": len(corrected) - len(legacy),
            "unique_full_inchikey": len(corrected_full) - len(legacy_full),
            "unique_ik14": len(corrected_ik14) - len(legacy_ik14),
            "keys_added": len(corrected_keys - legacy_keys),
            "keys_removed": len(legacy_keys - corrected_keys),
            "full_inchikey_added": len(corrected_full - legacy_full),
            "full_inchikey_removed": len(legacy_full - corrected_full),
            "ik14_added": len(corrected_ik14 - legacy_ik14),
            "ik14_removed": len(legacy_ik14 - corrected_ik14),
            "common_key_source_transitions": dict(sorted(transitions.items())),
        },
        "semantic_audit": {
            "legacy_n_sim_dropped": legacy_report.get("per_source", {}).get("massspecgym", {}).get("n_sim_dropped"),
            "corrected_schema_semantics": corrected_report.get("schema_semantics", {}),
            "simulation_challenge_is_provenance": False,
            "legacy_v2_admissible_as_current_gallery": False,
            "corrected_v3_role": "representative retrieval gallery only; not a replicate training bank",
        },
        "provenance": {
            "legacy_build_report_sha256": sha256(args.legacy_dir / "build_report.json"),
            "corrected_build_report_sha256": sha256(args.corrected_dir / "build_report.json"),
            "legacy_pos_sha256": sha256(args.legacy_dir / "unified_pos.mgf"),
            "legacy_neg_sha256": sha256(args.legacy_dir / "unified_neg.mgf"),
            "corrected_pos_sha256": sha256(args.corrected_dir / "unified_pos.mgf"),
            "corrected_neg_sha256": sha256(args.corrected_dir / "unified_neg.mgf"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
