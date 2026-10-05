"""CPU contracts for clean-boundary chemical-rule transfer."""

from __future__ import annotations

import numpy as np
import torch

from chemaware_rule_clean_boundary_core import (
    choose_opposed_boundaries,
    formula_balanced_weights,
    formula_cluster_interval,
    paired_margin_each,
    retrieval_metrics,
    weighted_mean,
)


def main() -> None:
    scores = np.asarray([0.7, 0.62, 0.60, 0.20])
    labels = np.asarray([1, 0, 0, 0])
    predicate = np.asarray([True, False, True, True])
    assert choose_opposed_boundaries(scores, labels, predicate) == (0, 1, 2)
    assert choose_opposed_boundaries(scores, labels, np.asarray([True, False, False, False])) is None
    assert choose_opposed_boundaries(scores, labels, np.asarray([False, False, True, True])) is None

    weights = formula_balanced_weights(np.asarray(["A", "A", "B"]))
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[:2].sum(), weights[2])

    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    positive = torch.tensor([[0.8, 0.0], [0.9, 0.0]])
    negative = torch.tensor([[0.7, 0.0], [0.1, 0.0]])
    each = paired_margin_each(query, positive, negative, 0.05, 0.05)
    assert each[1] < each[0]
    assert torch.isfinite(weighted_mean(each, torch.ones(2)))

    metrics = retrieval_metrics(np.asarray([1, 2, 4]), np.asarray([4, 4, 4]))
    assert metrics["recall1"] == 1 / 3
    assert metrics["recall5"] == 1.0
    assert np.isclose(metrics["micro_auc"], 5 / 9)
    interval = formula_cluster_interval(
        np.asarray([1.0, 1.0, -1.0]), np.asarray(["A", "A", "B"]), 7, 1000
    )
    assert interval["formula_clusters"] == 2
    print("PASS: ChemAware rule-selected clean-boundary core contracts")


if __name__ == "__main__":
    main()
