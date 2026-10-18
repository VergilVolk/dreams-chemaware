#!/usr/bin/env python
"""Contracts for the fixed risk-aligned additive final experiment."""
from __future__ import annotations

import numpy as np
import torch

from conditional_null_energy_core import BoundedResidualEnergy, conditional_null_loss
from train_wse_risk_aligned_additive_final import chemistry_gate, ranker_gate, variant_features


def main() -> None:
    model = BoundedResidualEnergy(1, hidden=2, residual_bound=0.2)
    base = torch.tensor([0.0, 0.0, 0.0, 3.0])
    features = torch.zeros((4, 1))
    nulls = torch.zeros((1, 4, 1))
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.int8)
    ptr = np.asarray([0, 2, 4], dtype=np.int64)
    unweighted = conditional_null_loss(
        model=model, base_score=base, actual_features=features, null_features=nulls,
        labels=labels, query_ptr=ptr, null_margin=0, null_weight=0, trust_weight=0,
    )
    weighted = conditional_null_loss(
        model=model, base_score=base, actual_features=features, null_features=nulls,
        labels=labels, query_ptr=ptr, null_margin=0, null_weight=0, trust_weight=0,
        query_weights=torch.tensor([2.0, 1.0]),
    )
    assert weighted.listwise < unweighted.listwise

    panel = {
        "features": np.ones((3, 2), dtype=np.float32),
        "null_features": np.stack((
            np.zeros((3, 2), dtype=np.float32),
            np.full((3, 2), 2.0, dtype=np.float32),
            np.full((3, 2), 3.0, dtype=np.float32),
        )),
    }
    actual, null = variant_features(panel, "candidate_mismatch")
    assert np.array_equal(actual, panel["null_features"][2])
    assert np.array_equal(null[0], panel["null_features"][0])
    assert np.array_equal(null[1], panel["null_features"][1])
    assert np.array_equal(null[2], panel["features"])

    candidate = {"mrr": 0.91, "micro_candidate_auroc": 0.92, "micro_candidate_auprc": 0.81}
    baseline = {"mrr": 0.90, "micro_candidate_auroc": 0.92, "micro_candidate_auprc": 0.81}
    comparison = {
        "delta_recall@1": 0.01,
        "formula_cluster_bootstrap_95ci": [0.002, 0.02],
        "risk_net_corrected_minus_2introduced": 3,
        "near": {"delta_recall@1": 0.001, "risk_net_corrected_minus_2introduced": 1},
    }
    assert all(ranker_gate(candidate, baseline, comparison).values())
    assert all(chemistry_gate(comparison).values())
    missing_near = dict(comparison)
    missing_near.pop("near")
    assert not all(ranker_gate(candidate, baseline, missing_near).values())
    auc_failure = dict(candidate)
    auc_failure["micro_candidate_auroc"] = baseline["micro_candidate_auroc"] - 0.002
    assert not all(ranker_gate(auc_failure, baseline, comparison).values())
    risk_failure = dict(comparison)
    risk_failure["risk_net_corrected_minus_2introduced"] = 0
    assert not all(chemistry_gate(risk_failure).values())
    print("PASS: WSE risk-aligned additive development contracts")


if __name__ == "__main__":
    main()
