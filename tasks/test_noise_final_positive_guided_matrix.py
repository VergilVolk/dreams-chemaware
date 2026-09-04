"""Fast deterministic tests for the positive-guided peak transforms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_positive_guided_matrix import apply_action, reference_profile  # noqa: E402


def main() -> None:
    query = torch.tensor([[500.0, 1.1], [100.0, 1.0], [200.0, 0.5], [300.0, 0.25], [0.0, 0.0]])
    ref1 = torch.tensor([[500.0, 1.1], [100.01, 0.4], [200.01, 1.0], [0.0, 0.0], [0.0, 0.0]])
    ref2 = torch.tensor([[500.0, 1.1], [100.00, 0.6], [250.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    prevalence, target = reference_profile(query, [ref1, ref2], 0.02)
    assert np.allclose(prevalence[1:4], [1.0, 0.5, 0.0])
    assert np.isclose(target[1], 0.5) and np.isclose(target[2], 1.0)
    for family in ("matched_intensity_transport", "prevalence_attenuation", "consensus_projection"):
        unchanged = apply_action(query, prevalence, target, family, 1e-6)
        assert unchanged.shape == query.shape
        transformed = apply_action(query, prevalence, target, family, 1.0)
        assert transformed[0, 1].item() == query[0, 1].item()
        assert torch.all(transformed[1:, 1] >= 0)
        assert torch.isclose(transformed[1:, 1].max(), torch.tensor(1.0))
    projected = apply_action(query, prevalence, target, "consensus_projection", 1.0)
    assert projected[3, 1].item() == 0.0
    print("[test_noise_final_positive_guided_matrix] PASS")


if __name__ == "__main__":
    main()
