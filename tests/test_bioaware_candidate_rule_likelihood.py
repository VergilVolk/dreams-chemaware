from __future__ import annotations

import numpy as np

from tasks.audit_bioaware_candidate_rule_likelihood import weighted_jaccard, weighted_overlap


def test_weighted_rule_scores_prefer_matching_candidate() -> None:
    query = np.asarray([1.0, 1.0, 0.0])
    matching = np.asarray([0.9, 0.8, 0.0])
    wrong = np.asarray([0.1, 0.0, 1.0])
    weight = np.asarray([1.0, 2.0, 1.0])
    assert weighted_overlap(query, matching, weight) > weighted_overlap(query, wrong, weight)
    assert weighted_jaccard(query, matching, weight) > weighted_jaccard(query, wrong, weight)
