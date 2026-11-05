#!/usr/bin/env python3
"""Build the deterministic compact evidence bundle for frozen CALMS v1.0.

The bundle deliberately contains the executable frozen release, the two
2026-11-05 result records, their exact analysis code and compact generated
outputs.  Large per-query ledgers and upstream benchmark inputs remain in the
repository/data archive and are not duplicated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


BUNDLE_NAME = "CALMS_v1.0_evidence_bundle_20261105"
BUNDLE_ROOT = PurePosixPath(BUNDLE_NAME)
FIXED_ZIP_TIME = (2026, 11, 5, 0, 0, 0)

RESULT_DOCS = (
    "docs/CALMS_GNPS_PAIRED_PUBLIC_METHOD_RESULT_20261105.md",
    "docs/CALMS_MECHANISM_AND_SELECTIVE_RANKING_RESULT_20261105.md",
)

ANALYSIS_CODE = (
    "tasks/summarize_calms_paired_superiority.py",
    "tasks/analyze_calms_correction_landscape.py",
    "tasks/evaluate_calms_closed_set_selective_ranking.py",
    "tasks/test_analyze_calms_correction_landscape.py",
    "tasks/test_evaluate_calms_closed_set_selective_ranking.py",
    "tasks/render_calms_mechanism_selective_figure.py",
    "tasks/build_calms_v1_evidence_bundle.py",
)

COMPACT_RESULTS = (
    "data/validation/calms_gnps_paired_superiority_v3_20261105/report.json",
    "data/validation/calms_gnps_paired_superiority_v3_20261105/calms_paired_superiority.tsv",
    "data/validation/calms_gnps_paired_superiority_v3_20261105/calms_paired_superiority.png",
    "data/validation/calms_gnps_paired_superiority_v3_20261105/calms_paired_superiority.pdf",
    "data/validation/calms_correction_landscape_20261105/report.json",
    "data/validation/calms_correction_landscape_20261105/strata_summary.tsv",
    "data/validation/calms_v132_closed_set_selective_ranking_20261105/report.json",
    "data/validation/calms_v132_closed_set_selective_ranking_20261105/global_metrics.csv",
    "data/validation/calms_v132_closed_set_selective_ranking_20261105/paired_functional_deltas.csv",
    "data/validation/calms_v132_closed_set_selective_ranking_20261105/selective_operating_points.csv",
    "deliverables/calms_v1_mechanism_selective_20261105/calms_mechanism_selective.png",
    "deliverables/calms_v1_mechanism_selective_20261105/calms_mechanism_selective.pdf",
    "deliverables/calms_v1_mechanism_selective_20261105/calms_mechanism_selective.svg",
    "deliverables/calms_v1_mechanism_selective_20261105/figure_caption.txt",
    "deliverables/calms_v1_mechanism_selective_20261105/panel_a_exchange_values.tsv",
)

EXCLUDED_LARGE_LEDGERS = (
    "data/validation/calms_correction_landscape_20261105/query_ledger_formula_disjoint_official_dreams.csv.gz",
    "data/validation/calms_correction_landscape_20261105/query_ledger_formula_disjoint_weighted_spectral_entropy.csv.gz",
    "data/validation/calms_correction_landscape_20261105/query_ledger_identity_disjoint_official_dreams.csv.gz",
    "data/validation/calms_correction_landscape_20261105/query_ledger_identity_disjoint_weighted_spectral_entropy.csv.gz",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deliverables") / f"{BUNDLE_NAME}.zip",
    )
    parser.add_argument(
        "--checksum-output",
        type=Path,
        default=None,
        help="Defaults to <output>.sha256.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_frozen_release_files(repository: Path) -> tuple[str, ...]:
    release = repository / "deploy/calms_v1"
    checksum_path = release / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    relative_files: list[str] = []
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative = raw_line.split(maxsplit=1)
        relative = relative.strip().replace("\\", "/")
        source = release / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = sha256_file(source)
        if observed != digest:
            raise RuntimeError(
                f"frozen release checksum mismatch: {source} "
                f"expected={digest} observed={observed}"
            )
        relative_files.append(f"deploy/calms_v1/{relative}")
    if len(relative_files) != len(set(relative_files)):
        raise RuntimeError("duplicate path in frozen CALMS checksum ledger")
    relative_files.append("deploy/calms_v1/checksums.sha256")
    return tuple(relative_files)


def bundle_readme() -> bytes:
    text = f"""# CALMS v1.0 compact evidence bundle

This deterministic archive contains the executable frozen CALMS v1.0 release,
the two 2026-11-05 consumed-GNPS result records, their exact summary and
diagnostic scripts, synthetic contract tests, and compact reports/tables/plots.

Scientific boundary: GNPS was consumed during development and residual-scale
selection. These results are development evidence, not unopened external
confirmation or a field-wide SOTA claim.

The four per-query correction ledgers are intentionally excluded to keep this
archive compact. Re-running the full diagnostics therefore additionally
requires the original repository benchmark inputs and query ledgers. The
frozen inference package itself is executable from this archive.

From the archive root, validate frozen inference with:

    python deploy/calms_v1/test_calms_frozen_release.py

The analysis tests require numpy, pandas, scikit-learn and pytest and can be
run with `tasks` on `PYTHONPATH`.

All payload hashes are in `checksums.sha256`; `bundle_manifest.json` records
the exact inclusion and exclusion policy. Archive entry order, timestamps,
permissions and compression settings are fixed by
`tasks/build_calms_v1_evidence_bundle.py`.
"""
    return text.encode("utf-8")


def manifest_payload(source_entries: dict[str, bytes]) -> bytes:
    records = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for path, payload in sorted(source_entries.items())
    ]
    manifest = {
        "schema": "calms_v1_compact_evidence_bundle_v1",
        "bundle_name": BUNDLE_NAME,
        "release": "CALMS v1.0 (Full23 v1.3.2)",
        "created_for_date": "2026-11-05",
        "deterministic_zip_timestamp": "2026-11-05T00:00:00",
        "scientific_boundary": (
            "Consumed GNPS development evidence; not unopened external "
            "confirmation or a field-wide SOTA claim."
        ),
        "contents": records,
        "excluded_large_ledgers": list(EXCLUDED_LARGE_LEDGERS),
        "exclusion_reason": (
            "Per-query ledgers are reproducibility inputs but are omitted from "
            "this compact evidence bundle; compact derived reports are included."
        ),
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def checksum_payload(entries: dict[str, bytes]) -> bytes:
    lines = [f"{sha256_bytes(payload)}  {path}" for path, payload in sorted(entries.items())]
    return ("\n".join(lines) + "\n").encode("utf-8")


def zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def write_deterministic_zip(path: Path, entries: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for relative, payload in sorted(entries.items()):
                archive.writestr(
                    zip_info(str(BUNDLE_ROOT / relative)),
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_archive(path: Path, expected: dict[str, bytes]) -> None:
    expected_names = [str(BUNDLE_ROOT / name) for name in sorted(expected)]
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        observed_names = [info.filename for info in infos]
        if observed_names != expected_names:
            raise RuntimeError("archive entry order or membership is not deterministic")
        if len(observed_names) != len(set(observed_names)):
            raise RuntimeError("archive contains duplicate paths")
        for info, relative in zip(infos, sorted(expected)):
            if info.date_time != FIXED_ZIP_TIME:
                raise RuntimeError(f"non-fixed timestamp: {info.filename}")
            if info.is_dir():
                raise RuntimeError(f"unexpected directory entry: {info.filename}")
            payload = archive.read(info.filename)
            if payload != expected[relative]:
                raise RuntimeError(f"archive payload differs: {info.filename}")
            lower = info.filename.lower()
            if "__pycache__" in lower or lower.endswith(".pyc"):
                raise RuntimeError(f"runtime cache leaked into bundle: {info.filename}")
            if "query_ledger_" in lower:
                raise RuntimeError(f"large query ledger leaked into bundle: {info.filename}")


def main() -> None:
    args = arguments()
    repository = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else repository / args.output
    checksum_output = args.checksum_output
    if checksum_output is None:
        checksum_output = output.with_name(output.name + ".sha256")
    elif not checksum_output.is_absolute():
        checksum_output = repository / checksum_output

    frozen_release = parse_frozen_release_files(repository)
    relative_sources = (*frozen_release, *RESULT_DOCS, *ANALYSIS_CODE, *COMPACT_RESULTS)
    if len(relative_sources) != len(set(relative_sources)):
        raise RuntimeError("bundle source list contains duplicate paths")

    source_entries: dict[str, bytes] = {}
    for relative in relative_sources:
        source = repository / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_entries[relative] = source.read_bytes()

    entries = dict(source_entries)
    entries["README.md"] = bundle_readme()
    entries["bundle_manifest.json"] = manifest_payload(source_entries)
    entries["checksums.sha256"] = checksum_payload(entries)

    write_deterministic_zip(output, entries)
    validate_archive(output, entries)
    archive_sha = sha256_file(output)
    checksum_output.parent.mkdir(parents=True, exist_ok=True)
    checksum_output.write_text(
        f"{archive_sha}  {output.name}\n",
        encoding="ascii",
        newline="\n",
    )

    print(json.dumps({
        "status": "CALMS_V1_COMPACT_EVIDENCE_BUNDLE_COMPLETE",
        "output": str(output.relative_to(repository)),
        "checksum_output": str(checksum_output.relative_to(repository)),
        "sha256": archive_sha,
        "entries": len(entries),
        "archive_bytes": output.stat().st_size,
        "excluded_large_query_ledgers": len(EXCLUDED_LARGE_LEDGERS),
    }, indent=2))


if __name__ == "__main__":
    main()
