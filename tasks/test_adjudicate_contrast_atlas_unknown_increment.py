from __future__ import annotations

import pandas as pd
import pytest

from adjudicate_contrast_atlas_unknown_increment import adjudicate, validate_outcomes


WORLDS = ("all", "known_only", "measurement_matched_known", "unresolved_shuffled_ms2")


def synthetic_outcomes() -> pd.DataFrame:
    rows = []
    for query in range(24):
        truth = "match" if query % 2 == 0 else "no_match"
        for world in WORLDS:
            if world == "all":
                answered = truth == "match"
                correct = True
                rank = 1 if truth == "match" else None
            else:
                answered = True
                correct = truth == "match" and query % 4 == 0
                rank = 1 if correct else (3 if truth == "match" else None)
            rows.append({
                "query_id": f"q{query}", "cluster_id": f"c{query}",
                "source_holdout": f"s{query % 2}", "instrument_holdout": f"i{query % 2}",
                "world": world, "truth_type": truth, "answered": answered,
                "correct": correct, "rank": rank,
            })
    return pd.DataFrame(rows)


def test_unknown_increment_passes_strong_paired_synthetic() -> None:
    report = adjudicate(synthetic_outcomes(), draws=499, seed=17)
    assert report["query_denominator"] == 24
    assert report["cluster_denominator"] == 24
    assert report["pass_unknown_increment"]
    assert report["overall_metrics"]["all"]["balanced_endpoint"] == pytest.approx(1.0)


def test_outcome_contract_rejects_world_and_no_match_drift() -> None:
    frame = synthetic_outcomes()
    broken = frame.loc[~((frame["query_id"] == "q0") & (frame["world"] == "known_only"))]
    with pytest.raises(RuntimeError, match="every query"):
        validate_outcomes(broken)
    broken = frame.copy()
    mask = (broken["query_id"] == "q1") & (broken["world"] == "all")
    broken.loc[mask, "correct"] = False
    with pytest.raises(RuntimeError, match="no-match correctness"):
        validate_outcomes(broken)
