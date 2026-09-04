#!/usr/bin/env python3
"""Require no-op overlay equivalence for all identity-bearing KGMN outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freeze_kgmn_metdna2_dreams_arm import compare_csv_multisets


RELATIVE_OUTPUTS = {
    "credential": Path("03_annotation_credential/annontation_credential_long.csv"),
    "wide_table": Path("00_annotation_table/table1_identification.csv"),
    "candidate_table": Path("00_annotation_table/table3_identification_pair.csv"),
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-run", type=Path, required=True)
    parser.add_argument("--noop-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite no-op audit: {args.output}")
    comparisons: dict[str, dict[str, object]] = {}
    for label, relative in RELATIVE_OUTPUTS.items():
        author = args.author_run / relative
        noop = args.noop_run / relative
        if not author.is_file() or not noop.is_file():
            raise FileNotFoundError(f"missing {label} output: author={author}, noop={noop}")
        equal, detail = compare_csv_multisets(author, noop)
        comparisons[label] = {
            "multiset_equal": bool(equal), "detail": detail,
            "author_sha256": sha256(author), "noop_sha256": sha256(noop),
        }
    passed = all(value["multiset_equal"] for value in comparisons.values())
    report = {
        "status": "kgmn_hidden_seed_noop_audit_complete", "formal": True,
        "comparisons": comparisons, "pass": passed,
        "contract": (
            "No-op overlay must reproduce the complete CSV row multiset for credential, "
            "wide identity and candidate identity outputs before any experimental arm."
        ),
        "claim_limit": "Technical identity gate only; no algorithm-performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError("no-op overlay changed one or more KGMN identity-bearing outputs")


if __name__ == "__main__":
    main()
