#!/usr/bin/env python
"""Dependency-light unit checks for the full-16 BioAware M0 audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from audit_bioaware_full16_action_support_m0 import (
        CandidateIndex,
        loso_report,
        strict_seed_pool,
        topology_outcome,
        undirected_adjacency,
    )
except ModuleNotFoundError:
    from tasks.audit_bioaware_full16_action_support_m0 import (
        CandidateIndex,
        loso_report,
        strict_seed_pool,
        topology_outcome,
        undirected_adjacency,
    )


def test_candidate_index() -> None:
    index = CandidateIndex(
        precursor_mz=np.asarray([100.0000, 100.0005, 100.0008, 100.01]),
        adduct=np.asarray(["[M+H]+", "[M+H]+", "[M+H]+", "[M-H]-"], dtype=object),
        ik14=np.asarray(["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"]),
        formula=np.asarray(["C1", "C1", "C2", "C3"], dtype=object),
        reference_source=np.asarray(["pos", "pos", "pos", "neg"], dtype=object),
    )
    result = index.query(100.0, "[M+H]+", 10.0)
    assert [item["candidate_ik14"] for item in result] == ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
    assert result[0]["candidate_spectra"] == 2
    assert result[1]["candidate_spectra"] == 1
    assert result[0]["candidate_reference_sources"] == "pos"
    assert index.query(100.0, "[M-H]-", 10.0) == []
    negative = index.query(100.01, "[M-H]-", 10.0)
    assert len(negative) == 1
    assert negative[0]["candidate_ik14"] == "CCCCCCCCCCCCCC"
    assert negative[0]["candidate_reference_sources"] == "neg"


def test_topology() -> None:
    edges = pd.DataFrame({"a": ["A" * 14, "B" * 14], "b": ["B" * 14, "C" * 14]})
    graph = undirected_adjacency(edges, "a", "b")
    assert graph["B" * 14] == {"A" * 14, "C" * 14}
    identities = {"s": {"A" * 14, "B" * 14, "C" * 14, "D" * 14}}
    formulas = {
        "A" * 14: {"F1"}, "B" * 14: {"F2"},
        "C" * 14: {"F1"}, "D" * 14: {"F4"},
    }
    seeds = strict_seed_pool("s", "A" * 14, "F1", identities, formulas)
    assert seeds == {"B" * 14, "D" * 14}
    assert topology_outcome(2, [1, 0]) == "truth_advantaged"
    assert topology_outcome(1, [1, 0]) == "tied"
    assert topology_outcome(0, [1, 0]) == "wrong_advantaged"


def test_loso_purge() -> None:
    ambiguous = pd.DataFrame(
        {
            "source": ["BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"],
            "panel_id": ["p0", "p1", "p2", "p3"],
            "truth_ik14": ["A" * 14, "B" * 14, "C" * 14, "D" * 14],
            "truth_formula": ["F0", "F1", "F0", "F3"],
            "union_any_candidate_supported": [True, True, False, True],
        }
    )
    # A held-source nonambiguous feature adds F1 to the purge universe.  That
    # must remove Mouse_brain even though the feature is absent from `ambiguous`.
    full = pd.concat(
        [
            ambiguous,
            pd.DataFrame(
                {
                    "source": ["BV2cell"],
                    "panel_id": ["p0"],
                    "truth_ik14": ["E" * 14],
                    "truth_formula": ["F1"],
                    "union_any_candidate_supported": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    report = loso_report(ambiguous, full)
    held = report.loc[report.held_source == "BV2cell"].iloc[0]
    # Mouse_liver shares F0 and Mouse_brain shares nonambiguous held formula F1.
    assert held.train_rows_after_identity_formula_purge == 1
    assert held.train_identities_after_purge == 1
    assert held.held_full_level1_identities_used_for_purge == 2


def main() -> None:
    test_candidate_index()
    test_topology()
    test_loso_purge()
    print("[test_bioaware_full16_action_support_m0] PASS")


if __name__ == "__main__":
    main()
