#!/usr/bin/env python
"""Freeze the two-axis A1 evaluation manifest for reverse-context modelling.

The unit recorded here is a *positive spectral observation* plus its repository
opportunity strata.  A missing match remains unlabelled.  The manifest freezes
project and chemical-identity holdouts before any B0/B1/M1 model is fitted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from preflight_reverse_context_joint_model import (
    BLANK_TYPES,
    DatasetteClient,
    _atomic_csv_gz,
    _atomic_json,
    _sha256_file,
)


MISSING = {None, "", "missing value", "nan"}


def _stable_test(key: str, fraction: float, seed: int, namespace: str) -> bool:
    payload = f"{namespace}|{seed}|{key}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64 < fraction


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _enrich_anchors(
    client: DatasetteClient, anchors: list[dict[str, Any]], batch_size: int = 64
) -> list[dict[str, Any]]:
    ids = [int(row["spectrum_id_int"]) for row in anchors]
    details: dict[int, dict[str, Any]] = {}
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        id_sql = ",".join(str(value) for value in batch)
        sql = f"""
select spectrum_id_int,
       Smiles as smiles,
       INCHI as inchi,
       ExactMass as exact_mass,
       Precursor_MZ as precursor_mz,
       Adduct as adduct,
       Ion_Mode as ion_mode,
       classyfire_superclass,
       classyfire_class,
       classyfire_subclass,
       fp_morgan,
       fp_pattern
from library_table
where spectrum_id_int in ({id_sql})
""".strip()
        for row in client.query(sql):
            details[int(row["spectrum_id_int"])] = row
    missing = sorted(set(ids) - set(details))
    if missing:
        raise RuntimeError(f"missing anchor details for {len(missing)} spectra")
    enriched = []
    for anchor in anchors:
        spectrum_id = int(anchor["spectrum_id_int"])
        combined = dict(anchor)
        combined.update(details[spectrum_id])
        enriched.append(combined)
    return enriched


def _summarize_quadrant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    biological = [row for row in rows if row["observation_type"] == "biological"]
    return {
        "rows": len(rows),
        "biological_rows": len(biological),
        "biological_matched_files": sum(int(row["matched_files"]) for row in biological),
        "projects": len({row["dataset"] for row in biological}),
        "identities": len({row["ik14"] for row in biological}),
        "sample_types": len({row["sample_type"] for row in biological}),
    }


def build_manifest(
    report: dict[str, Any],
    occurrences: list[dict[str, str]],
    opportunities: list[dict[str, str]],
    project_test_fraction: float,
    identity_test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    anchors = report["anchor_metrics"]
    anchor_by_id = {int(row["spectrum_id_int"]): row for row in anchors}
    if len(anchor_by_id) != len(anchors):
        raise RuntimeError("duplicate anchor spectrum IDs")

    projects = sorted(
        {row["dataset"] for row in opportunities if row.get("dataset") not in MISSING}
    )
    project_split = {
        project: (
            "test"
            if _stable_test(project, project_test_fraction, seed, "project")
            else "development"
        )
        for project in projects
    }
    identity_split = {
        str(anchor["ik14"]): (
            "test"
            if _stable_test(str(anchor["ik14"]), identity_test_fraction, seed, "identity")
            else "development"
        )
        for anchor in anchors
    }

    ledger: list[dict[str, Any]] = []
    for occurrence in occurrences:
        spectrum_id = int(occurrence["spectrum_id_int"])
        if spectrum_id not in anchor_by_id:
            raise RuntimeError(f"occurrence references unknown anchor: {spectrum_id}")
        dataset = occurrence["dataset"]
        if dataset not in project_split:
            raise RuntimeError(f"occurrence project absent from opportunity ledger: {dataset}")
        anchor = anchor_by_id[spectrum_id]
        sample_type = occurrence.get("sample_type")
        if sample_type in BLANK_TYPES:
            observation_type = "blank_or_qc"
        elif sample_type in MISSING:
            observation_type = "metadata_missing"
        else:
            observation_type = "biological"
        p_split = project_split[dataset]
        i_split = identity_split[str(anchor["ik14"])]
        if p_split == "development" and i_split == "development":
            quadrant = "development"
        elif p_split == "test" and i_split == "development":
            quadrant = "project_holdout"
        elif p_split == "development" and i_split == "test":
            quadrant = "identity_holdout"
        else:
            quadrant = "joint_holdout"
        ledger.append(
            {
                **occurrence,
                "ik14": anchor["ik14"],
                "classyfire_superclass": anchor["classyfire_superclass"],
                "project_split": p_split,
                "identity_split": i_split,
                "evaluation_quadrant": quadrant,
                "observation_type": observation_type,
            }
        )

    by_quadrant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_quadrant[row["evaluation_quadrant"]].append(row)
    quadrant_names = (
        "development",
        "project_holdout",
        "identity_holdout",
        "joint_holdout",
    )
    quadrants = {name: _summarize_quadrant(by_quadrant[name]) for name in quadrant_names}
    biological = [row for row in ledger if row["observation_type"] == "biological"]
    bio_identity_counts = Counter(row["identity_split"] for row in biological)
    bio_project_counts = Counter(row["project_split"] for row in biological)
    sample_types = Counter()
    for row in biological:
        sample_types[row["sample_type"]] += int(row["matched_files"])

    gates = {
        "a0b_passed": bool(report.get("pass_to_a1")),
        "anchors_ge_200": len(anchors) >= 200,
        "biological_hit_identities_ge_80": len({row["ik14"] for row in biological}) >= 80,
        "biological_hit_projects_ge_500": len({row["dataset"] for row in biological}) >= 500,
        "biological_sample_types_ge_8": len(sample_types) >= 8,
        "development_and_test_have_biological_rows": all(
            bio_project_counts.get(split, 0) > 0 for split in ("development", "test")
        ),
        "development_and_test_have_hit_identities": all(
            bio_identity_counts.get(split, 0) > 0 for split in ("development", "test")
        ),
        "joint_holdout_biological_rows_ge_50": quadrants["joint_holdout"]["biological_rows"] >= 50,
        "blank_qc_retained": any(row["observation_type"] == "blank_or_qc" for row in ledger),
    }
    manifest = {
        "status": "reverse_context_joint_a1_manifest_frozen",
        "formal": True,
        "seed": seed,
        "splits": {
            "project_test_fraction": project_test_fraction,
            "identity_test_fraction": identity_test_fraction,
            "projects": Counter(project_split.values()),
            "identities": Counter(identity_split.values()),
        },
        "positive_observations": {
            "rows": len(ledger),
            "biological_rows": len(biological),
            "biological_matched_files": sum(int(row["matched_files"]) for row in biological),
            "sample_type_matched_files": dict(sample_types.most_common()),
        },
        "quadrants": quadrants,
        "gates": gates,
        "pass_to_model_baselines": all(gates.values()),
        "model_contract": {
            "B0": "repository exposure and global observation-rate baseline",
            "B1": "B0 plus analytical-condition detectability; no chemical-context interaction",
            "M1": "B1 plus chemical-by-biological-context interaction",
            "primary_evaluation": "project_holdout and joint_holdout, never random spectrum split",
            "zero_semantics": "unobserved match under repository opportunity, not biological absence",
            "health_status": "audit-only in v1 because missingness is dominant",
        },
        "claim_limit": (
            "This freezes data axes and positive observations. It is not a trained model, "
            "does not establish biological absence, and does not establish novelty."
        ),
    }
    split_rows = [
        {"axis": "project", "key": key, "split": split}
        for key, split in sorted(project_split.items())
    ] + [
        {"axis": "identity", "key": key, "split": split}
        for key, split in sorted(identity_split.items())
    ]
    return ledger, manifest, split_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a0-dir", type=Path, default=Path("data/validation/reverse_context_joint_a0b")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/validation/reverse_context_joint_a1")
    )
    parser.add_argument("--project-test-fraction", type=float, default=0.20)
    parser.add_argument("--identity-test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.project_test_fraction < 0.5:
        raise ValueError("project-test-fraction must be in (0, 0.5)")
    if not 0.0 < args.identity_test_fraction < 0.5:
        raise ValueError("identity-test-fraction must be in (0, 0.5)")
    report_path = args.output_dir / "manifest.json"
    if report_path.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite frozen A1 manifest: {report_path}")

    a0_report_path = args.a0_dir / "report.json"
    occurrence_path = args.a0_dir / "occurrences.csv.gz"
    opportunity_path = args.a0_dir / "opportunities.csv.gz"
    for path in (a0_report_path, occurrence_path, opportunity_path):
        if not path.exists():
            raise FileNotFoundError(path)
    report = json.loads(a0_report_path.read_text(encoding="utf-8"))
    if not report.get("pass_to_a1"):
        raise RuntimeError("A0 did not pass; refusing to freeze A1")
    client = DatasetteClient(args.output_dir / "api_cache", timeout=args.timeout)
    report["anchor_metrics"] = _enrich_anchors(client, report["anchor_metrics"])
    occurrences = _read_csv_gz(occurrence_path)
    opportunities = _read_csv_gz(opportunity_path)
    ledger, manifest, split_rows = build_manifest(
        report,
        occurrences,
        opportunities,
        args.project_test_fraction,
        args.identity_test_fraction,
        args.seed,
    )
    manifest["provenance"] = {
        "a0_report_sha256": _sha256_file(a0_report_path),
        "occurrences_sha256": _sha256_file(occurrence_path),
        "opportunities_sha256": _sha256_file(opportunity_path),
        "script_sha256": _sha256_file(Path(__file__)),
    }
    ledger_fields = [
        "spectrum_id_int", "ik14", "classyfire_superclass", "dataset",
        "sample_type", "body_part", "health_status", "polarity", "instrument",
        "matched_files", "matched_spectra", "mean_cosine", "max_cosine",
        "mean_matching_peaks", "project_split", "identity_split",
        "evaluation_quadrant", "observation_type",
    ]
    anchor_fields = sorted({key for row in report["anchor_metrics"] for key in row})
    _atomic_csv_gz(args.output_dir / "positive_observations.csv.gz", ledger, ledger_fields)
    _atomic_csv_gz(args.output_dir / "anchors.csv.gz", report["anchor_metrics"], anchor_fields)
    _atomic_csv_gz(args.output_dir / "split_assignments.csv.gz", split_rows, ["axis", "key", "split"])
    _atomic_text(
        args.output_dir / "README.txt",
        "Frozen A1 two-axis manifest. Missing matches are unlabelled, not negatives.\n",
    )
    _atomic_json(report_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), flush=True)


if __name__ == "__main__":
    main()
