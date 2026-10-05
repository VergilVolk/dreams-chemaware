#!/usr/bin/env python
"""Unit checks for B41 topology and formula-preserving null construction."""
from __future__ import annotations

import gzip
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b41_cross_catalog_topology import (  # noqa: E402
    CROSS_FEATURES,
    add_catalog_features,
    topology_signature,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        kegg = root / "kegg.csv.gz"
        emrn = root / "emrn.csv.gz"
        edges = pd.DataFrame({
            "ik14_a": ["A", "B", "A", "C", "C"],
            "ik14_b": ["B", "A", "C", "C", "D"],
        })
        edges.to_csv(kegg, index=False, compression="gzip")
        pd.concat([edges, pd.DataFrame({"ik14_a": ["D"], "ik14_b": ["E"]})]).to_csv(
            emrn, index=False, compression="gzip"
        )
        members, degree, report = topology_signature(kegg)
        assert members == {"A", "B", "C", "D"}
        assert np.isclose(degree["A"], np.log1p(2))
        assert report["unique_undirected_nonself_edges"] == 3

        frame = pd.DataFrame({
            "query_id": ["q1", "q1", "q2", "q2"],
            "candidate_id": ["A", "B", "C", "D"],
            "truth_formula": ["F1", "F1", "F2", "F2"],
            "source": ["S"] * 4,
            "is_positive": [True, False, True, False],
            "network_member": [1.0, 0.0, 1.0, 0.0],
            "known_log_degree": [1.0, 0.0, 2.0, 0.0],
        })
        output, audit = add_catalog_features(frame, kegg, emrn, 3, 7)
        assert len(output) == 4
        assert output["kegg_member"].eq(1.0).all()
        assert len(audit["formula_preserving_nulls"]) == 3
        for repeat in range(3):
            before = output.groupby("truth_formula")[CROSS_FEATURES].sum().to_numpy(float)
            after = output.groupby("truth_formula")[[
                f"null{repeat}_{column}" for column in CROSS_FEATURES
            ]].sum().to_numpy(float)
            assert np.allclose(before, after)
    print("[test_bioaware_b41_cross_catalog_topology] PASS", flush=True)


if __name__ == "__main__":
    main()
