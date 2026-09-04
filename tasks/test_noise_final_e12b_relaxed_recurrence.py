"""Static/semantic tests for E12-B."""
from __future__ import annotations

import numpy as np
import torch

from audit_noise_final_e12b_relaxed_recurrence_matrix import POLICIES, RECIPES, relaxed_variant


def main() -> None:
    if len(POLICIES) != 5 or len(RECIPES) != 5:
        raise AssertionError("E12-B must contain a fixed 5x5 design")
    clean = torch.zeros((6, 2), dtype=torch.float32)
    clean[0] = torch.tensor([500.0, 1.0])
    clean[1] = torch.tensor([100.0, 1.0])
    missing = np.asarray([[150.0, 0.8, 0.5], [175.0, 0.4, 1.0]], dtype=np.float32)
    prevalence = np.zeros(6, dtype=np.float32)
    standard = relaxed_variant(clean, missing, prevalence, 5, 0.5, False)
    weighted = relaxed_variant(clean, missing, prevalence, 5, 0.5, True)
    if torch.equal(standard, clean) or torch.equal(weighted, clean):
        raise AssertionError("E12-B recurrence did not alter the spectrum")
    if torch.equal(standard, weighted):
        raise AssertionError("E12-B support weighting has no semantic effect")
    print("[test_noise_final_e12b_relaxed_recurrence] PASS", flush=True)


if __name__ == "__main__":
    main()
