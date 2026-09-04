"""Unit tests for tangent gradients and compatibility summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit_noise_final_e3_gradient_compatibility import pairwise_compatibility, tangent_direction


def main() -> None:
    clean = np.asarray([1.0, 0.0, 0.0])
    target = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    direction, magnitude = tangent_direction(clean, target)
    assert np.allclose(direction, [0.0, 1.0, 0.0])
    assert np.isclose(magnitude, 1.0 / np.sqrt(2.0))

    left = pd.DataFrame({
        "query_index": np.arange(20),
        "query_formula": [f"F{x}" for x in range(20)],
        "direction": [np.asarray([0.0, 1.0, 0.0], dtype=np.float32)] * 20,
    })
    right = left.copy()
    result = pairwise_compatibility(left, right, 500, 7)
    assert result["overlap"] == 20
    assert result["mean_cosine"] > 0.999
    assert result["negative_fraction"] == 0.0
    print("[test_noise_final_e3_gradient_compatibility] PASS")


if __name__ == "__main__":
    main()
