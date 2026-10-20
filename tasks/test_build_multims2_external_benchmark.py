#!/usr/bin/env python
"""Executable contracts for the score-blind MultiMS2 panel builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_multims2_external_benchmark import build_panel


def fixture() -> pd.DataFrame:
    rows = []
    identities = (
        ("BBBBBBBBBBBBBB", "C6H12O6", 101.00000, False, False),
        ("AAAAAAAAAAAAAA", "C5H10O5", 101.00050, False, False),
        ("CCCCCCCCCCCCCC", "C7H14O7", 150.0, True, True),
    )
    source_row = 0
    for ik14, formula, mz, identity_overlap, formula_overlap in identities:
        for method, energy in (("CID", 40.0), ("CID", 60.0), ("EAD", 16.0)):
            rows.append({
                "source_row": source_row,
                "ik14": ik14,
                "formula": formula,
                "source_collection": "NEXUS" if ik14 != "AAAAAAAAAAAAAA" else "SELLECK",
                "smiles": "CCO",
                "precursor_mz": mz,
                "fragmentation_method": method,
                "collision_energy": energy,
                "n_peaks": 20,
                "consumed_identity_overlap": identity_overlap,
                "consumed_formula_overlap": formula_overlap,
                "consumed_spectrum_overlap": False,
            })
            source_row += 1
    return pd.DataFrame(rows)


def main() -> None:
    observable, truth, report = build_panel(fixture(), "CID", 60.0, "identity", 10.0)
    assert report["competitive_queries"] == 2
    assert "query_ik14" not in observable
    assert "candidate_label" not in observable
    assert "near_query" not in observable
    assert set(truth["query_source_collection"]) == {"NEXUS", "SELLECK"}
    assert set(observable["candidate_source_collection"]) == {"NEXUS", "SELLECK"}
    assert np.array_equal(observable["query_token"], truth["query_token"])
    assert len(observable["query_ptr"]) == 3
    positive_positions = []
    for left, right in zip(truth["query_ptr"][:-1], truth["query_ptr"][1:]):
        assert int(truth["candidate_label"][left:right].sum()) == 1
        positive_positions.append(int(np.argmax(truth["candidate_label"][left:right])))
    assert positive_positions == [1, 0]  # candidate order does not reveal truth
    _, _, formula_report = build_panel(fixture(), "CID", 60.0, "formula", 10.0)
    assert formula_report["competitive_queries"] == 2
    _, _, ead_report = build_panel(fixture(), "EAD", 16.0, "identity", 10.0)
    assert ead_report["competitive_queries"] == 2
    print("PASS: MultiMS2 external score-blind benchmark contracts")


if __name__ == "__main__":
    main()
