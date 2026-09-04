#!/usr/bin/env python3
"""Preregistered decision over the matched-protocol MetDNA2 edge arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIMARY_ARM = "author_official_intersection"
MECHANISM_ARM = "official_dreams"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(path: Path, expected: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "kgmn_metdna2_200std_dreams_arm_frozen":
        raise RuntimeError(f"invalid frozen arm report: {path}")
    if report.get("arm") != expected:
        raise RuntimeError(f"arm report mismatch: expected {expected}, found {report.get('arm')}")
    return report


def summarize(noop: dict[str, object], official: dict[str, object], primary: dict[str, object]) -> dict[str, object]:
    comparisons = noop.get("noop_author_table_reproduction")
    if not isinstance(comparisons, dict) or not comparisons:
        raise RuntimeError("no-op report lacks author table reproduction evidence")
    no_op_tables_equal = all(bool(item.get("equal")) for item in comparisons.values())
    no_op_exact = bool(
        no_op_tables_equal
        and noop.get("corrected") == 0
        and noop.get("introduced") == 0
        and abs(float(noop.get("delta_recall1", float("nan")))) <= 1e-15
    )
    if not no_op_exact:
        raise RuntimeError("no-op overlay did not exactly reproduce the author workflow")

    author_reference = noop.get("author")
    for name, report in ((MECHANISM_ARM, official), (PRIMARY_ARM, primary)):
        if report.get("author") != author_reference:
            raise RuntimeError(f"author metrics differ in {name}")
        if report.get("noop_author_table_reproduction") is not None:
            raise RuntimeError(f"experimental arm unexpectedly contains no-op evidence: {name}")
        if report.get("external_provenance", {}).get("author_baseline_sha256") != noop.get(
            "external_provenance", {}
        ).get("author_baseline_sha256"):
            raise RuntimeError(f"author baseline provenance differs in {name}")

    primary_corrected = int(primary["corrected"])
    primary_introduced = int(primary["introduced"])
    primary_delta = float(primary["delta_recall1"])
    primary_top5_delta = float(primary["candidate"]["recall5"] - primary["author"]["recall5"])
    primary_coverage_delta = float(primary["candidate"]["coverage"] - primary["author"]["coverage"])
    gates = {
        "noop_exact_author_reproduction": no_op_exact,
        "primary_recall1_positive": primary_delta > 0,
        "primary_corrected_gt_introduced": primary_corrected > primary_introduced,
        "primary_recall5_nonnegative": primary_top5_delta >= 0,
        "primary_coverage_nonnegative": primary_coverage_delta >= 0,
    }
    technical_demo_pass = all(gates.values())
    return {
        "status": "kgmn_metdna2_dreams_arm_decision_complete",
        "formal": True,
        "preregistered_primary_arm": PRIMARY_ARM,
        "mechanism_only_arm": MECHANISM_ARM,
        "author": author_reference,
        "primary": {
            "delta_recall1": primary_delta,
            "delta_recall5": primary_top5_delta,
            "delta_coverage": primary_coverage_delta,
            "corrected": primary_corrected,
            "introduced": primary_introduced,
            "mcnemar_exact_p": float(primary["mcnemar_exact_p"]),
        },
        "mechanism": {
            "delta_recall1": float(official["delta_recall1"]),
            "corrected": int(official["corrected"]),
            "introduced": int(official["introduced"]),
            "mcnemar_exact_p": float(official["mcnemar_exact_p"]),
        },
        "gates": gates,
        "technical_demo_pass": technical_demo_pass,
        "eligible_for_external_hidden_seed_validation": technical_demo_pass,
        "decision": (
            "Freeze the primary arm and proceed to an external, study-isolated hidden-seed recovery protocol."
            if technical_demo_pass
            else "Stop the edge-replacement branch; do not choose the mechanism arm post hoc."
        ),
        "claim_limit": (
            "Passing establishes only a positive matched author-demo result and permission for external validation. "
            "It does not establish independent metabolite-annotation improvement, shared-embedding improvement, "
            "biological discovery, or SOTA."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noop", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--intersection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite decision artifact: {args.output}")
    noop = load_arm(args.noop, "noop_author")
    official = load_arm(args.official, MECHANISM_ARM)
    primary = load_arm(args.intersection, PRIMARY_ARM)
    report = summarize(noop, official, primary)
    report["provenance"] = {
        "noop_sha256": sha256(args.noop),
        "official_sha256": sha256(args.official),
        "intersection_sha256": sha256(args.intersection),
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
