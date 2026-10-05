#!/usr/bin/env python
"""Validate B47 truth-blind score/seed artefacts without opening truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    report_path = directory / "report.json"
    outputs = {
        "candidate_scores": directory / "candidate_scores.csv.gz",
        "query_summaries": directory / "query_summaries.csv.gz",
        "feature_consensus": directory / "feature_consensus.csv.gz",
        "seeds_primary": directory / "seeds_primary.csv.gz",
        "seeds_strict": directory / "seeds_strict.csv.gz",
    }
    for path in (report_path, *outputs.values()):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b47_truthblind_seed_construction_complete":
        raise RuntimeError("unexpected B47 seed status")
    if not report.get("formal"):
        raise RuntimeError("B47 seed output is not formal")
    contract = report.get("contracts", {})
    required_false = ("truth_opened", "phenotype_used", "performance_computed", "P2b_used")
    if any(contract.get(key) is not False for key in required_false):
        raise RuntimeError("B47 truth-blind seed contract changed")
    if contract.get("original_seed_set_frozen_before_truth") is not True:
        raise RuntimeError("B47 original seed set was not frozen")
    for key, path in outputs.items():
        if report.get("provenance", {}).get(f"{key}_sha256") != sha256_file(path):
            raise RuntimeError(f"B47 seed output hash mismatch: {key}")

    scores = pd.read_csv(outputs["candidate_scores"])
    query = pd.read_csv(outputs["query_summaries"])
    consensus = pd.read_csv(outputs["feature_consensus"])
    primary = pd.read_csv(outputs["seeds_primary"])
    strict = pd.read_csv(outputs["seeds_strict"])
    if len(scores) != int(report["candidate_identities"]):
        raise RuntimeError("B47 candidate-score count mismatch")
    if scores["query_id"].nunique() != int(report["queries"]) or len(query) != int(report["queries"]):
        raise RuntimeError("B47 scored query count mismatch")
    if query["query_id"].duplicated().any() or (scores.groupby("query_id").size() < 2).any():
        raise RuntimeError("B47 query uniqueness/candidate multiplicity failure")
    if not np.isfinite(scores["spectral_score"]).all():
        raise RuntimeError("B47 scores are non-finite")
    if any("truth" in column.casefold() for column in scores.columns):
        raise RuntimeError("truth-like column found in B47 candidate scores")
    if len(consensus) != int(report["features"]):
        raise RuntimeError("B47 feature-consensus count mismatch")

    policy_frames = (("primary", primary), ("strict", strict))
    for name, frame in policy_frames:
        expected = report["primary_seeds" if name == "primary" else "strict_sensitivity_seeds"]
        if len(frame) != int(expected["rows"]):
            raise RuntimeError(f"B47 {name} seed count mismatch")
        if frame.duplicated(["study", "sample", "seed_compound_id"]).any():
            raise RuntimeError(f"B47 {name} seed support was not sample/identity collapsed")
        policy = report["policies"][name]
        if len(frame) and (
            (frame["seed_score"] < float(policy["minimum_score"])).any()
            or (frame["seed_margin"] < float(policy["minimum_margin"])).any()
            or (frame["feature_support_samples"] < int(policy["minimum_modal_samples"])).any()
            or (frame["feature_consensus_fraction"] < float(policy["minimum_modal_fraction"])).any()
            or (frame["reaction_degree"] <= 0).any()
            or (
                frame["reaction_degree"]
                > int(report["parameters"]["maximum_seed_degree"])
            ).any()
        ):
            raise RuntimeError(f"B47 {name} seed violates its frozen policy")
    print(
        "[validate_bioaware_b47_truthblind_seeds] PASS "
        f"queries={len(query):,} primary_seeds={len(primary):,} "
        f"event_graph_gate={bool(report.get('pass_to_truthblind_event_graph'))}"
    )


if __name__ == "__main__":
    main()
