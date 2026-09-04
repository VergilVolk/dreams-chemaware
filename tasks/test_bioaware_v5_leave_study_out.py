#!/usr/bin/env python
"""Deterministic smoke test for nested leave-study-out BioAware V5."""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

from develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES,
    RAW_FAMILIES,
    add_family_features,
    apply_gate,
    score_queries,
)
from develop_bioaware_v5_leave_study_out import (
    inner_study_oof,
    query_metrics,
)


def main() -> None:
    rows: list[dict] = []
    studies = ("A", "B", "C")
    for study_index, study in enumerate(studies):
        for query_index in range(60):
            truth = f"T{study_index}_{query_index}"
            for candidate_index in range(3):
                candidate = truth if candidate_index == 0 else f"N{candidate_index}_{query_index}"
                row = {
                    "query_id": f"{study}:{query_index}",
                    "candidate_id": candidate,
                    "truth_candidate_id": truth,
                    "truth_formula": f"C{query_index + 2}H{query_index + 4}",
                    "study_id": study,
                    "spectral_score": .70 - .04 * candidate_index,
                }
                for family_columns in RAW_FAMILIES.values():
                    for column in family_columns:
                        row[column] = float(candidate_index == 0) + .01 * study_index
                rows.append(row)
    frame = add_family_features(pd.DataFrame(rows))
    args = Namespace(temperature=.1, l2=.05, maximum_family_weight=1.0)
    prediction = inner_study_oof(frame, args)
    assert len(prediction) == 180
    assert not prediction.query_id.duplicated().any()
    prediction["intervene"] = False
    prediction["final_correct"] = prediction["baseline_correct"]
    prediction["corrected"] = False
    prediction["introduced"] = False
    report = query_metrics(prediction)
    assert report["queries"] == 180
    assert np.isclose(report["delta_recall1"], 0.0)

    # A displayed truth inside a spectral tie is still wrong.  Abstention must
    # inherit that strict baseline rather than create a free correction.
    tied = pd.DataFrame([
        {"query_id": "tie", "candidate_id": "A_truth", "truth_candidate_id": "A_truth",
         "truth_formula": "C2H4", "spectral_score": .5},
        {"query_id": "tie", "candidate_id": "B_wrong", "truth_candidate_id": "A_truth",
         "truth_formula": "C2H4", "spectral_score": .5},
    ])
    for feature in FAMILY_FEATURES:
        tied[feature] = 0.0
    strict = apply_gate(
        score_queries(tied, np.zeros(len(FAMILY_FEATURES))),
        (0.0, float("inf"), len(RAW_FAMILIES) + 1),
    ).iloc[0]
    assert not bool(strict.baseline_correct)
    assert not bool(strict.intervene)
    assert not bool(strict.final_correct)
    assert not bool(strict.corrected)
    print("[test_bioaware_v5_leave_study_out] PASS")


if __name__ == "__main__":
    main()
