"""Focused regression tests for ICEBERG structure-differential actions."""
from __future__ import annotations

import numpy as np
import torch

from chemaware_iceberg_peak_action_core import (
    apply_peak_action, differential_evidence, hard_negative_indices, peak_bins,
)


def main() -> None:
    ptr = np.asarray([0, 3, 5])
    distance = np.asarray([0.2, 0.4, 0.3, 0.5, 0.1])
    assert hard_negative_indices(ptr, distance).tolist() == [2, 4]

    true = np.zeros(15_000, dtype=np.float32)
    neg = np.zeros_like(true)
    bins = peak_bins(np.asarray([10.0, 20.0]))
    true[bins[0]] = 1.0
    neg[bins[1]] = 1.0
    evidence = differential_evidence(true, neg, np.asarray([10.0, 20.0, 0.0]))
    assert evidence[0] > 0 and evidence[1] < 0

    clean = torch.tensor([[50.0, 1.1], [10.0, 0.5], [20.0, 1.0], [0.0, 0.0]])
    action = apply_peak_action(clean, evidence, "signed_exp", 1.0)
    assert torch.equal(action[:, 0], clean[:, 0])
    assert action[1, 1] / action[2, 1] > clean[1, 1] / clean[2, 1]
    assert float(action[1:, 1].max()) == 1.0

    conflict = apply_peak_action(clean, evidence, "conflict_attenuate", 0.5, top_k=1)
    assert conflict[2, 1] / conflict[1, 1] < clean[2, 1] / clean[1, 1]
    print("PASS: ICEBERG peak action core")


if __name__ == "__main__":
    main()
