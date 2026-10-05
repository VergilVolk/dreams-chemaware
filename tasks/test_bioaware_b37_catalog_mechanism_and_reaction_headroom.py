#!/usr/bin/env python
"""Dependency-free unit checks for B37 logic."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b37_catalog_mechanism_and_reaction_headroom import (  # noqa: E402
    CATALOG_ARMS,
    compare_b36_catalog_replay,
)


def main() -> None:
    required = {
        "spectral_only", "catalog_topology", "catalog_observability", "catalog_full"
    }
    assert required <= set(CATALOG_ARMS)
    assert CATALOG_ARMS["catalog_full"] == [
        "spectral_score", "network_member", "known_log_degree",
        "known_mass_candidate_fraction", "log_reference_spectra",
    ]
    # Make a tiny replay deliberately fail at the frozen-size guard.  This
    # ensures tests do not silently weaken the exact 860-query contract.
    row = {
        "query_id": "q", "source": "s", "truth_candidate_id": "t",
        "truth_formula": "f", "baseline_candidate_id": "b",
        "proposed_candidate_id": "p", "final_candidate_id": "p",
        "baseline_correct": False, "proposal_unique": True,
        "intervene": True, "final_correct": True, "corrected": True,
        "introduced": False, "delta": 1, "gate_name": "g",
        "proposal_probability": 0.9, "gate_margin": 0.05,
        "gate_probability": 0.7,
    }
    archived = pd.DataFrame([{**row, "arm": "spectral_plus_catalog"}])
    observed = pd.DataFrame([{**row, "arm": "catalog_full"}])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archive.csv.gz"
        archived.to_csv(path, index=False, compression="gzip")
        try:
            compare_b36_catalog_replay(observed, path)
        except RuntimeError as error:
            assert "exactly 860" in str(error)
        else:
            raise AssertionError("frozen replay-size guard did not trigger")
    print("[test_bioaware_b37_catalog_mechanism_and_reaction_headroom] PASS", flush=True)


if __name__ == "__main__":
    main()
