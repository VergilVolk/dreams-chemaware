#!/usr/bin/env python
"""Independent validator for BioAware B39-M1 internal edge evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = {
    "truth_candidate_id", "truth_formula", "is_positive", "baseline_correct",
    "corrected", "introduced", "final_correct", "delta", "spectral_score",
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
        "edges": args.input_dir / "candidate_seed_edge_evidence.csv.gz",
        "events": args.input_dir / "internal_atomic_events_with_edge_evidence.csv.gz",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    edges = pd.read_csv(paths["edges"], low_memory=False)
    events = pd.read_csv(paths["events"], dtype={"reaction_id": str}, low_memory=False)
    if report.get("status") != "bioaware_b39_m1_internal_edge_evidence_complete":
        raise RuntimeError("unexpected B39-M1 status")
    if report.get("model_fitted") or report.get("embedding_values_read") or report.get("ranking_outcomes_read"):
        raise RuntimeError("B39-M1 accessed a forbidden model/outcome channel")
    leaked = FORBIDDEN & (set(edges.columns) | set(events.columns))
    if leaked:
        raise RuntimeError(f"B39-M1 outcome leakage: {sorted(leaked)}")
    if edges.duplicated(["query_id", "candidate_id", "fold", "seed_identity"]).any():
        raise RuntimeError("duplicate candidate-seed edge evidence")
    if events.duplicated(
        ["query_id", "candidate_id", "seed_stratum", "seed_identity", "edge_source", "edge_key", "reaction_id"]
    ).any():
        raise RuntimeError("duplicate joined atomic events")
    if not set(events["context_class"].astype(str)) == {"S"}:
        raise RuntimeError("B39-M1 internal ledger mixed context classes")
    if set(edges["evidence_resolution"].dropna().astype(str)) != {
        "candidate_seed_edge_not_reaction_id_specific"
    }:
        raise RuntimeError("B39-M1 evidence resolution was mislabeled")
    if not events["event_specific_spectral_evidence_available"].astype(bool).equals(
        events["spectral_available"].astype(bool)
    ):
        raise RuntimeError("spectral availability flag mismatch")
    if not events["event_specific_coabundance_evidence_available"].astype(bool).equals(
        events["coabundance_available"].astype(bool)
    ):
        raise RuntimeError("co-abundance availability flag mismatch")
    provenance = report.get("provenance", {})
    if provenance.get("edge_evidence") != sha256(paths["edges"]):
        raise RuntimeError("edge evidence provenance mismatch")
    if provenance.get("joined_events") != sha256(paths["events"]):
        raise RuntimeError("joined event provenance mismatch")
    if report.get("formal"):
        for item in report.get("historical_cache_reproduction", {}).values():
            if float(item.get("maximum_absolute_error", 1.0)) > 1e-10:
                raise RuntimeError("formal B3/B9 historical replay drift")
    print(
        "[validate_bioaware_b39_m1_internal_edge_evidence] PASS",
        {
            "edges": len(edges),
            "events": len(events),
            "spectral": int(events["spectral_available"].sum()),
            "coabundance": int(events["coabundance_available"].sum()),
            "both": int(events["both_experimental_layers_available"].sum()),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
