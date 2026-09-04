"""Deterministic unit tests for the A4-B1 residual student."""
from __future__ import annotations

import numpy as np
import torch

from audit_noise_v3_a4_exact_peak_scan import strict_detail
from train_noise_v3_a4b_rescue_adapter import ResidualAdapter


def main() -> None:
    torch.manual_seed(7)
    x = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    for nonlinear in (False, True):
        model = ResidualAdapter(16, 8, nonlinear)
        with torch.inference_mode():
            output = model(x)
        assert torch.allclose(output, x, atol=1e-6), "zero-init must be exact identity"

    rows = np.asarray([10, 11], dtype=np.int64)
    ptr = np.asarray([0, 1, 2], dtype=np.int64)
    locked = np.asarray([0.5, 0.50000001], dtype=np.float64)
    recomputed = np.asarray([0.50000003, 0.5], dtype=np.float64)
    assert strict_detail(locked, rows, ptr)["rank"] == 2
    anchored = locked + (recomputed - recomputed)
    assert strict_detail(anchored, rows, ptr)["rank"] == 2
    print("[test_noise_v3_a4b_rescue_adapter] PASS", flush=True)


if __name__ == "__main__":
    main()
