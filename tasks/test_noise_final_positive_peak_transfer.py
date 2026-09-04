"""Deterministic transform tests for recurrent positive peak transfer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_positive_peak_transfer import apply_transfer, recurrent_missing_peaks  # noqa: E402
from audit_noise_final_positive_guided_matrix import reference_profile  # noqa: E402


def main() -> None:
    query = torch.tensor([[500.0, 1.1], [100.0, 1.0], [200.0, 0.5], [0.0, 0.0], [0.0, 0.0]])
    ref1 = torch.tensor([[500.0, 1.1], [100.01, 1.0], [300.00, 0.4], [0.0, 0.0], [0.0, 0.0]])
    ref2 = torch.tensor([[500.0, 1.1], [100.00, 0.8], [300.01, 0.6], [0.0, 0.0], [0.0, 0.0]])
    missing = recurrent_missing_peaks(query, [ref1, ref2], 0.02, 0.67, 5)
    assert missing.shape == (1, 3)
    assert np.isclose(missing[0, 0], 300.005, atol=0.01)
    prevalence, _ = reference_profile(query, [ref1, ref2], 0.02)
    for family in ("recurrent_peak_graft", "balanced_peak_exchange", "recurrent_union_mix"):
        variant, count = apply_transfer(query, missing, prevalence, family, 0.25)
        assert count == 1
        assert variant[0, 1].item() == query[0, 1].item()
        assert torch.isclose(variant[1:, 1].max(), torch.tensor(1.0))
        assert torch.any(torch.isclose(variant[1:, 0], torch.tensor(300.005), atol=0.01))
    print("[test_noise_final_positive_peak_transfer] PASS")


if __name__ == "__main__":
    main()
