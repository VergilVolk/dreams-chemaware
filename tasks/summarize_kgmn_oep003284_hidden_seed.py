#!/usr/bin/env python3
"""Freeze the final decision for the external KGMN hidden-seed experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_report(path: Path, status: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != status:
        raise RuntimeError(f"status mismatch for {path}: {report.get('status')}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite final hidden-seed decision: {args.output}")

    paths = {
        "noop_positive": args.root / "noop_audit_positive.json",
        "noop_negative": args.root / "noop_audit_negative.json",
        "official_dreams": args.root / "evaluation_official_dreams/report.json",
        "author_official_intersection": args.root / "evaluation_author_official_intersection/report.json",
    }
    noop_positive = read_report(paths["noop_positive"], "kgmn_hidden_seed_noop_audit_complete")
    noop_negative = read_report(paths["noop_negative"], "kgmn_hidden_seed_noop_audit_complete")
    official = read_report(paths["official_dreams"], "kgmn_hidden_seed_recovery_evaluation_complete")
    intersection = read_report(
        paths["author_official_intersection"], "kgmn_hidden_seed_recovery_evaluation_complete"
    )
    if noop_positive.get("pass") is not True or noop_negative.get("pass") is not True:
        raise RuntimeError("external no-op identity gate failed")
    if official.get("candidate_arm") != "official_dreams":
        raise RuntimeError("official DreaMS arm label mismatch")
    if intersection.get("candidate_arm") != "author_official_intersection":
        raise RuntimeError("pre-registered primary arm label mismatch")
    official_provenance = official.get("provenance", {})
    intersection_provenance = intersection.get("provenance", {})
    for field in ("contract_report_sha256", "hidden_seed_splits_sha256", "author_predictions_sha256"):
        if official_provenance.get(field) != intersection_provenance.get(field):
            raise RuntimeError(f"evaluation arms do not share the frozen denominator: {field}")

    primary_pass = intersection.get("pass") is True
    report = {
        "status": (
            "kgmn_oep003284_hidden_seed_external_passed"
            if primary_pass else "kgmn_oep003284_hidden_seed_external_failed"
        ),
        "formal": True,
        "primary_arm": "author_official_intersection",
        "primary": intersection,
        "secondary_arm": "official_dreams",
        "secondary": official,
        "no_op_identity_gates": {
            "positive": noop_positive,
            "negative": noop_negative,
        },
        "decision": {
            "pass_to_network_teacher_embedding_distillation": primary_pass,
            "failed_primary_is_not_rescued_by_secondary": True,
            "thresholds_tuned_on_oep003284": False,
        },
        "interpretation": (
            "The pre-registered conservative DreaMS-author edge intersection improved closed-world "
            "hidden-seed recovery under the full author KGMN pipeline."
            if primary_pass else
            "The pre-registered conservative edge intersection did not pass the external hidden-seed gates."
        ),
        "provenance": {name: sha256(path) for name, path in paths.items()},
        "claim_limit": (
            "This is an external closed-world KGMN edge-reliability test. Even a pass is not evidence of "
            "open-world SOTA, MSI Level 1 identity, phenotype mechanism, or improved shared embedding."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
