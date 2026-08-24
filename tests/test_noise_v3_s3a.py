from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tasks.analyze_noise_v3_s3a_extended_matrix import add_rule_evidence, top_counts
from tasks.audit_noise_v3_s2_sequential import registered_actions


def test_registered_s3a_actions_are_exact_and_unique() -> None:
    args = SimpleNamespace(
        action_specs=[
            "candidate_gradient:0.50", "role_confounder:1.00",
            "role_shared:0.25", "role_unmatched:1.00",
        ],
        selectors=[], attenuations=[],
    )
    assert registered_actions(args) == [
        ("candidate_gradient", 0.5), ("role_confounder", 1.0),
        ("role_shared", 0.25), ("role_unmatched", 1.0),
    ]


def test_rule_evidence_is_attached_without_becoming_a_label() -> None:
    frame = pd.DataFrame({
        "query_row": [10],
        "baseline_winner_pair_row": [11],
        "winner_pair_row": [12],
    })
    lookup = {
        10: np.asarray([1, 1, 0, 0], dtype=bool),
        11: np.asarray([1, 0, 1, 0], dtype=bool),
        12: np.asarray([1, 1, 0, 1], dtype=bool),
    }
    categories = np.asarray(["CF", "NL", "ISO", "HR"])
    libraries = np.asarray(["core", "core", "massbank", "massbank"])
    result = add_rule_evidence(frame, lookup, categories, libraries)
    assert result.loc[0, "baseline_rule_jaccard"] == 1 / 3
    assert result.loc[0, "target_rule_jaccard"] == 2 / 3
    assert result.loc[0, "target_rule_jaccard_core"] == 1.0
    assert result.loc[0, "baseline_rule_jaccard_massbank"] == 0.0
    assert "label" not in result.columns


def test_top_counts_retains_joint_error_destination() -> None:
    frame = pd.DataFrame({
        "query_formula": ["A", "A", "B"],
        "winner_formula": ["X", "X", "Y"],
    })
    records = top_counts(frame, ["query_formula", "winner_formula"])
    assert records[0] == {
        "query_formula": "A", "winner_formula": "X", "queries": 2,
    }
