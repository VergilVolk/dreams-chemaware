"""Bind a completed incremental Enveda audit to the verified source MGF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_enveda180_scoreblind_manifest import sha256_file


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-mgf", type=Path, required=True)
    parser.add_argument("--expected-source-bytes", type=int, required=True)
    parser.add_argument("--expected-source-md5", required=True)
    args = parser.parse_args()

    report_path = args.audit / "report.json"
    manifest = args.audit / "eligible_records.csv.gz"
    conflicts = args.audit / "conflicting_spectrum_hashes.txt.gz"
    if not all(path.is_file() for path in (report_path, manifest, conflicts, args.source_mgf)):
        raise RuntimeError("completed audit, ledgers, or source MGF is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ENVEDA180_SCOREBLIND_MANIFEST_COMPLETE":
        raise RuntimeError("Enveda audit is not complete")
    if report.get("performance_scores_opened") is not False or report.get("model_loaded") is not False:
        raise RuntimeError("Enveda audit is not score blind")
    if report.get("exclusion_policy_complete") is not True:
        raise RuntimeError("Enveda exclusion policy is incomplete")
    if sha256_file(manifest) != report.get("manifest_sha256"):
        raise RuntimeError("upgraded Enveda manifest hash mismatch")
    if sha256_file(conflicts) != report.get("conflicting_hashes_sha256"):
        raise RuntimeError("upgraded Enveda conflict-ledger hash mismatch")
    actual_bytes = args.source_mgf.stat().st_size
    actual_md5 = md5_file(args.source_mgf)
    if actual_bytes != args.expected_source_bytes or actual_md5 != args.expected_source_md5:
        raise RuntimeError(
            f"source MGF bytes/MD5 mismatch: bytes={actual_bytes}, md5={actual_md5}"
        )
    source_sha256 = sha256_file(args.source_mgf)
    report.update({
        "source_mgf": str(args.source_mgf),
        "source_mgf_bytes": actual_bytes,
        "source_mgf_md5": actual_md5,
        "source_mgf_sha256": source_sha256,
    })
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({
        "status": "ENVEDA180_UPGRADED_AUDIT_SOURCE_BOUND",
        "audit": str(args.audit),
        "source_mgf_sha256": source_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
