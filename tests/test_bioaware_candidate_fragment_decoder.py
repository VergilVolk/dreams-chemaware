from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.audit_bioaware_candidate_fragment_decoder import (
    candidate_score,
    decoder_prediction,
    fit_kernel_decoder,
    top1,
)


def test_kernel_decoder_prefers_matching_target() -> None:
    embeddings = np.eye(3, dtype=float)
    targets = np.eye(3, dtype=float)
    train_x, dual, mean = fit_kernel_decoder(embeddings, targets, alpha=0.01)
    prediction = decoder_prediction(embeddings[:1], train_x, dual, mean)[0]
    matching = candidate_score(prediction, targets[0], mean)
    other = candidate_score(prediction, targets[1], mean)
    assert matching > other


def test_top1_counts_tie_against_truth() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "truth_candidate_id": ["A", "A"],
        "score": [0.5, 0.5],
    })
    predicted, correct = top1(frame, "score")
    assert predicted == "A"
    assert not correct
