#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b47_b1_artifact_registry import (  # noqa: E402
    artifact_row, forbidden_headers, registry_decisions,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        clean = root / "clean.csv.gz"
        dirty = root / "dirty.csv.gz"
        pd.DataFrame({"query_id": ["q"], "candidate_id": ["c"]}).to_csv(
            clean, index=False, compression="gzip"
        )
        pd.DataFrame({"query_id": ["q"], "truth_identity": ["c"]}).to_csv(
            dirty, index=False, compression="gzip"
        )
        row = artifact_row("test", "clean", clean)
        assert row["present"] and row["bytes"] > 0 and len(row["sha256"]) == 64
        assert forbidden_headers(clean) == []
        assert forbidden_headers(dirty) == ["truth_identity"]
    decisions = registry_decisions(
        graph_valid=True, embedding_valid=True, seed_artifact_valid=True,
        seed_policy_pass=False, u0_valid=True, headers_valid=True,
    )
    assert decisions["pass_b1_provenance"] is True
    assert decisions["pass_to_b2_exact_event"] is False
    print("[test_bioaware_b47_b1_artifact_registry] PASS")


if __name__ == "__main__":
    main()
