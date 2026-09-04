from __future__ import annotations

import pytest

from tasks.summarize_kgmn_metdna2_dreams_arms import summarize


def arm(name: str, *, delta: float, corrected: int, introduced: int, recall5: float = 0.8) -> dict:
    return {
        "status": "kgmn_metdna2_200std_dreams_arm_frozen",
        "arm": name,
        "author": {"recall1": 0.7, "recall5": 0.8, "coverage": 0.9},
        "candidate": {"recall1": 0.7 + delta, "recall5": recall5, "coverage": 0.9},
        "delta_recall1": delta,
        "corrected": corrected,
        "introduced": introduced,
        "mcnemar_exact_p": 0.1,
        "noop_author_table_reproduction": None,
        "external_provenance": {"author_baseline_sha256": "same"},
    }


def test_preregistered_intersection_can_pass_to_external_validation() -> None:
    noop = arm("noop_author", delta=0.0, corrected=0, introduced=0)
    noop["noop_author_table_reproduction"] = {
        "credential": {"equal": True},
        "final_identification": {"equal": True},
        "final_pairs": {"equal": True},
    }
    official = arm("official_dreams", delta=0.02, corrected=4, introduced=2)
    primary = arm("author_official_intersection", delta=0.01, corrected=3, introduced=1)
    result = summarize(noop, official, primary)
    assert result["technical_demo_pass"] is True
    assert result["preregistered_primary_arm"] == "author_official_intersection"


def test_mechanism_arm_cannot_rescue_failed_preregistered_primary() -> None:
    noop = arm("noop_author", delta=0.0, corrected=0, introduced=0)
    noop["noop_author_table_reproduction"] = {"all": {"equal": True}}
    official = arm("official_dreams", delta=0.03, corrected=5, introduced=1)
    primary = arm("author_official_intersection", delta=-0.01, corrected=1, introduced=2)
    result = summarize(noop, official, primary)
    assert result["technical_demo_pass"] is False
    assert result["eligible_for_external_hidden_seed_validation"] is False


def test_noop_mismatch_fails_closed() -> None:
    noop = arm("noop_author", delta=0.0, corrected=0, introduced=0)
    noop["noop_author_table_reproduction"] = {"all": {"equal": False}}
    with pytest.raises(RuntimeError, match="no-op"):
        summarize(
            noop,
            arm("official_dreams", delta=0.01, corrected=2, introduced=1),
            arm("author_official_intersection", delta=0.01, corrected=2, introduced=1),
        )
