#!/usr/bin/env python
"""Independent structural validator for the BioAware B39-M0 ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = {
    "truth_candidate_id", "truth_formula", "is_positive", "baseline_candidate_id",
    "baseline_correct", "corrected", "introduced", "final_correct", "delta",
    "spectral_score", "baseline_gap",
}
EXPECTED_SOURCES = {
    "BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma",
    "ST001154_same_formula_10ppm", "KGMN200STD_hidden_seed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "report": args.input_dir / "report.json",
        "events": args.input_dir / "atomic_reaction_events.csv.gz",
        "contexts": args.input_dir / "candidate_context_semantics.csv.gz",
        "seed_contexts": args.input_dir / "visible_seed_contexts.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    events = pd.read_csv(paths["events"], dtype={"reaction_id": str}, low_memory=False)
    contexts = pd.read_csv(paths["contexts"])
    seed_contexts = pd.read_csv(paths["seed_contexts"])
    if report.get("status") != "bioaware_b39_m0_atomic_event_ledger_complete":
        raise RuntimeError("unexpected B39-M0 status")
    if report.get("model_fitted") or report.get("embedding_values_read") or report.get("ranking_outcomes_read"):
        raise RuntimeError("B39-M0 accessed a forbidden model/outcome channel")
    leaked = FORBIDDEN & (set(events.columns) | set(contexts.columns) | set(seed_contexts.columns))
    if leaked:
        raise RuntimeError(f"B39-M0 outcome leakage: {sorted(leaked)}")
    if not len(events) or not len(contexts):
        raise RuntimeError("B39-M0 ledger is empty")
    if set(events["source"].astype(str)) != EXPECTED_SOURCES:
        raise RuntimeError("B39-M0 event ledger does not cover all six sources")
    if set(contexts["source"].astype(str)) != EXPECTED_SOURCES:
        raise RuntimeError("B39-M0 context ledger does not cover all six sources")
    if events.duplicated(
        ["query_id", "candidate_id", "seed_stratum", "seed_identity", "edge_source", "edge_key", "reaction_id"]
    ).any():
        raise RuntimeError("B39-M0 contains duplicate atomic events")
    if contexts.duplicated(["query_id", "candidate_id", "seed_stratum"]).any():
        raise RuntimeError("B39-M0 contains duplicate candidate contexts")
    if seed_contexts.duplicated(["query_id", "seed_stratum"]).any():
        raise RuntimeError("B39-M0 contains duplicate visible-seed contexts")
    if set(seed_contexts["source"].astype(str)) != EXPECTED_SOURCES:
        raise RuntimeError("B39-M0 visible-seed ledger does not cover all six sources")
    listed_counts = seed_contexts["visible_seed_identity_list"].fillna("").astype(str).map(
        lambda value: len([item for item in value.split(";") if item])
    )
    if not listed_counts.eq(seed_contexts["visible_seed_count"].astype(int)).all():
        raise RuntimeError("B39-M0 visible-seed list/count mismatch")
    if not events["event_supported_candidate_count"].ge(1).all():
        raise RuntimeError("invalid candidate-specificity denominator")
    expected_specific = events["event_supported_candidate_count"].eq(1)
    if not expected_specific.eq(events["event_candidate_specific"].astype(bool)).all():
        raise RuntimeError("candidate-specificity flag is inconsistent")
    if events["event_specific_spectral_evidence_available"].astype(bool).any():
        raise RuntimeError("legacy spectral proxy was promoted to event evidence")
    if events["event_specific_coabundance_evidence_available"].astype(bool).any():
        raise RuntimeError("legacy co-abundance proxy was promoted to event evidence")
    if contexts["legacy_experimental_evidence_is_event_specific"].astype(bool).any():
        raise RuntimeError("legacy candidate-context evidence was mislabeled as event-specific")
    st = report.get("st_leave_one_seed_out_audit", {})
    if st.get("manifest_queries") != st.get("evaluation_identity_already_in_sample_seed_set"):
        raise RuntimeError("ST leave-one-seed-out audit no longer reproduces")
    if report.get("prospective_unknown_sources") != 0:
        raise RuntimeError("existing six-domain development set was mislabeled prospective")
    provenance = report.get("provenance", {})
    if provenance.get("atomic_events") != sha256(paths["events"]):
        raise RuntimeError("atomic event provenance mismatch")
    if provenance.get("candidate_contexts") != sha256(paths["contexts"]):
        raise RuntimeError("candidate-context provenance mismatch")
    if provenance.get("visible_seed_contexts") != sha256(paths["seed_contexts"]):
        raise RuntimeError("visible-seed context provenance mismatch")
    gates = report.get("gates", {})
    if bool(report.get("pass_to_b39_m1")) != bool(gates and all(gates.values())):
        raise RuntimeError("B39-M0 decision does not equal gate conjunction")
    print(
        "[validate_bioaware_b39_atomic_event_ledger] PASS",
        {
            "events": len(events),
            "contexts": len(contexts),
            "queries": contexts["query_id"].nunique(),
            "candidate_specific": int(events["event_candidate_specific"].sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
