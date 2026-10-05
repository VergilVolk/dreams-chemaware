#!/usr/bin/env python
"""Unit checks for B42 Rhea graph construction."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b42_independent_catalog_topology import rhea_signature  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "participants.csv.gz"
        frame = pd.DataFrame({
            "compound_id": ["A", "B", "W", "B", "C", "D", "E"],
            "reaction_id": ["r1", "r1", "r1", "r2", "r2", "r3", "r3"],
            "is_currency": [False, False, True, False, False, False, False],
        })
        frame.to_csv(path, index=False, compression="gzip")
        members, degree, edges, report = rhea_signature(path, maximum_participants=8)
        assert members == {"A", "B", "C", "D", "E"}
        assert edges == {("A", "B"), ("B", "C"), ("D", "E")}
        assert np.isclose(degree["B"], np.log1p(2))
        assert report["reactions_used"] == 3
        assert report["unique_undirected_nonself_edges"] == 3
    print("[test_bioaware_b42_independent_catalog_topology] PASS", flush=True)


if __name__ == "__main__":
    main()
