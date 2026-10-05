#!/usr/bin/env python
"""Metadata-only inventory of the candidate B47 external benchmark archives.

The archive payloads are never extracted or opened.  This stage verifies the
deposited MD5 values, rejects unsafe ZIP paths, inventories filenames and sizes,
and assigns only provisional filename-based namespaces for manual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "head_to_head": {
        "record": "https://zenodo.org/records/21574649",
        "filename": "Supplementary Data 2. Head-to-head comparison between metabolite annotation algorithms.zip",
        "md5": "97510e6a680e7d42d5ca3ca0a360196d",
        "reported_size_mb": 392.9,
    },
    "external_validation": {
        "record": "https://zenodo.org/records/21574649",
        "filename": "Supplementary Data 5. External validation of MSMICA results.zip",
        "md5": "6adabc70d429e75fac75555bab3cca5e",
        "reported_size_mb": 95.4,
    },
}


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def meaningful_member_name(name: str) -> str:
    """Remove packaging-only path components before filename classification."""
    parts = list(PurePosixPath(name.replace("\\", "/")).parts)
    if parts and parts[0].casefold() == "__macosx":
        parts = parts[1:]
    if parts and parts[0].casefold().startswith("supplementary data "):
        parts = parts[1:]
    return "/".join(parts)


def provisional_namespace(name: str) -> str:
    meaningful = meaningful_member_name(name)
    lower = meaningful.casefold()
    basename = PurePosixPath(meaningful).name.casefold()
    if basename.endswith((".r", ".rmd", ".py", ".ipynb", ".sh")) or "script" in lower:
        return "code_like"
    if any(
        token in lower
        for token in (
            "ground truth",
            "ground_truth",
            "confirmed_metabolite",
            "/validation/",
            "eva_output",
        )
    ) or any(
        token in basename
        for token in ("truth", "validated", "validation summary", "outcome")
    ):
        return "sealed_truth_or_evaluation_like"
    if any(
        token in lower
        for token in (
            "metdna2",
            "metdna3",
            "msmica",
            "xmsannotator",
            "annotation result",
            "output",
        )
    ):
        return "algorithm_output_like"
    if any(
        token in lower
        for token in (
            "peak table",
            "peak_table",
            "sample info",
            "sample_info",
            "abundance",
            "mzml",
            "mgf",
            "msp",
            "ms2",
            "retention",
        )
    ):
        return "observable_input_like"
    return "unclassified"


def safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(normalized) and not path.is_absolute() and ".." not in path.parts


def inspect_archive(path: Path) -> dict:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            members = []
            seen = set()
            duplicates = []
            unsafe = []
            for info in infos:
                name = info.filename.replace("\\", "/")
                folded = name.casefold()
                if folded in seen:
                    duplicates.append(name)
                seen.add(folded)
                if not safe_member(name):
                    unsafe.append(name)
                members.append(
                    {
                        "name": name,
                        "bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                        "is_directory": bool(info.is_dir()),
                        "provisional_namespace": provisional_namespace(name),
                    }
                )
    except BadZipFile as error:
        raise RuntimeError(f"invalid ZIP archive: {path}") from error

    counts: dict[str, int] = {}
    for item in members:
        key = item["provisional_namespace"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "members": len(members),
        "uncompressed_bytes": int(sum(item["bytes"] for item in members)),
        "unsafe_member_paths": unsafe,
        "casefold_duplicate_paths": duplicates,
        "provisional_namespace_counts": dict(sorted(counts.items())),
        "member_inventory": members,
    }


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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-to-head", type=Path, required=True)
    parser.add_argument("--external-validation", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/bioaware_b47_external_bundle_preflight_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")

    supplied = {
        "head_to_head": args.head_to_head.resolve(),
        "external_validation": args.external_validation.resolve(),
    }
    archives = {}
    for key, path in supplied.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_md5 = digest(path, "md5")
        observed_sha256 = digest(path, "sha256")
        inventory = inspect_archive(path)
        archives[key] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "expected": EXPECTED[key],
            "observed_md5": observed_md5,
            "observed_sha256": observed_sha256,
            "md5_matches_deposit": observed_md5 == EXPECTED[key]["md5"],
            **inventory,
        }

    gates = {
        "both_archives_present": len(archives) == 2,
        "all_deposited_md5_match": all(item["md5_matches_deposit"] for item in archives.values()),
        "all_archives_nonempty": all(item["members"] > 0 for item in archives.values()),
        "no_unsafe_member_paths": all(
            not item["unsafe_member_paths"] for item in archives.values()
        ),
        "no_casefold_duplicate_paths": all(
            not item["casefold_duplicate_paths"] for item in archives.values()
        ),
    }
    report = {
        "status": "bioaware_b47_external_bundle_metadata_preflight_complete",
        "formal": True,
        "archive_payloads_opened": False,
        "archive_payloads_extracted": False,
        "truth_values_read": False,
        "archives": archives,
        "gates": gates,
        "pass_to_manual_namespace_mapping": all(gates.values()),
        "next_stage": (
            "Manually map members into observable_input, context_construction, "
            "algorithm_output and sealed_truth. Filename heuristics are not evidence."
        ),
        "forbidden_next_action": (
            "Do not train, tune or compute an annotation metric until the namespace map "
            "and truth-separation gate pass."
        ),
        "claim_limit": (
            "This stage verifies archive identity and path safety only. Provisional filename "
            "namespaces do not establish that a prospective BioAware benchmark is constructible."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    console = {
        "status": report["status"],
        "formal": report["formal"],
        "truth_values_read": report["truth_values_read"],
        "archives": {
            key: {
                "bytes": item["bytes"],
                "members": item["members"],
                "observed_md5": item["observed_md5"],
                "observed_sha256": item["observed_sha256"],
                "md5_matches_deposit": item["md5_matches_deposit"],
                "unsafe_member_paths": len(item["unsafe_member_paths"]),
                "casefold_duplicate_paths": len(item["casefold_duplicate_paths"]),
                "provisional_namespace_counts": item["provisional_namespace_counts"],
            }
            for key, item in archives.items()
        },
        "gates": gates,
        "pass_to_manual_namespace_mapping": report["pass_to_manual_namespace_mapping"],
        "report": str(output / "report.json"),
    }
    print(json.dumps(console, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
