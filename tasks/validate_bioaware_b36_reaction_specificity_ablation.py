#!/usr/bin/env python
"""Independent structural and arithmetic validation of a B36 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REAL_ARMS = {
    "spectral_only",
    "spectral_plus_catalog",
    "spectral_plus_reaction",
    "spectral_plus_catalog_plus_reaction",
}
FORBIDDEN_FEATURE_TOKENS = (
    "truth", "candidate_id", "query_id", "formula", "source", "is_positive",
    "baseline_correct", "final_correct", "phenotype", "p2b",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    transitions_path = args.output_dir / "nested_domain_loso_arm_transitions.csv.gz"
    selection_path = args.output_dir / "inner_gate_selection_ledger.csv.gz"
    transform_path = args.output_dir / "null_transform_audit.csv.gz"
    for path in (report_path, transitions_path, selection_path, transform_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = pd.read_csv(transitions_path)
    selection = pd.read_csv(selection_path)
    transforms = pd.read_csv(transform_path)
    if report.get("status") != "bioaware_b36_reaction_specificity_ablation_complete":
        raise RuntimeError("wrong B36 status")
    if not REAL_ARMS.issubset(set(rows["arm"].astype(str))):
        raise RuntimeError("B36 real evidence arms are incomplete")
    arms = sorted(rows["arm"].astype(str).unique())
    repeats = int(report["null_controls"]["repeats_per_method"])
    if len(arms) != 4 + 2 * repeats:
        raise RuntimeError("B36 arm count does not match the frozen null design")
    saved_arms = {
        **report["real_arms"],
        **report["null_controls"]["arm_results"],
    }
    if set(saved_arms) != set(arms):
        raise RuntimeError("B36 saved arm reports do not match transition arms")
    for arm in arms:
        local = rows.loc[rows["arm"].eq(arm)]
        if len(local) != 860 or local["query_id"].nunique() != 860:
            raise RuntimeError(f"{arm}: invalid outer OOF coverage")
        if local.duplicated("query_id").any():
            raise RuntimeError(f"{arm}: duplicate query")
        corrected = int((~local["baseline_correct"].astype(bool) & local["final_correct"].astype(bool)).sum())
        introduced = int((local["baseline_correct"].astype(bool) & ~local["final_correct"].astype(bool)).sum())
        delta = float((local["final_correct"].astype(int) - local["baseline_correct"].astype(int)).mean())
        saved = saved_arms[arm]
        if corrected != int(saved["corrected"]) or introduced != int(saved["introduced"]):
            raise RuntimeError(f"{arm}: corrected/introduced mismatch")
        if not np.isclose(delta, float(saved["delta_recall1"]), atol=1e-15, rtol=0):
            raise RuntimeError(f"{arm}: delta mismatch")
    baseline = rows.pivot(index="query_id", columns="arm", values="baseline_correct")
    if not baseline.nunique(axis=1).eq(1).all():
        raise RuntimeError("baseline changed across B36 arms")
    if not selection.groupby(["outer_domain", "arm"], sort=False).size().eq(16).all():
        raise RuntimeError("every B36 arm/fold must report no-op plus 15 gates")
    if not (transforms["maximum_group_feature_sum_error"].astype(float) <= 1e-9).all():
        raise RuntimeError("a reaction null failed multiset preservation")
    if not (transforms["assigned_from_other_row"].astype(int) > 0).any():
        raise RuntimeError("reaction null controls did not move any rows")
    for method in ("within_query", "catalog_stratum"):
        local = transforms.loc[transforms["method"].eq(method)]
        if local.empty:
            raise RuntimeError(f"missing B36 null transform: {method}")
        changed_fraction = (
            local["value_changed_rows"].astype(float).sum()
            / local["rows"].astype(float).sum()
        )
        if changed_fraction <= 0.20:
            raise RuntimeError(
                f"{method}: too few reaction vectors changed ({changed_fraction:.3f})"
            )
    feature_contract = report["feature_contract"]
    feature_names = (
        feature_contract["spectral"]
        + feature_contract["catalog"]
        + feature_contract["reaction"]
    )
    for feature in feature_names:
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS):
            raise RuntimeError(f"forbidden B36 feature: {feature}")
    provenance = report["provenance"]
    expected_hashes = {
        "transitions_sha256": sha256(transitions_path),
        "selection_ledger_sha256": sha256(selection_path),
        "null_transform_audit_sha256": sha256(transform_path),
    }
    for key, value in expected_hashes.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"B36 provenance mismatch: {key}")
    print(
        "[validate_bioaware_b36_reaction_specificity_ablation] PASS",
        "SCIENTIFIC_PASS" if report.get("scientific_pass") else "SCIENTIFIC_FAIL_VALID_ARTIFACT",
        report["gates"],
    )


if __name__ == "__main__":
    main()
