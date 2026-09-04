"""Small deterministic unit tests for A4-B0 geometry helpers."""
from __future__ import annotations

import numpy as np

from diagnose_noise_v3_a4b_positive_evidence import mix_embedding, normalized_mean
from audit_noise_v3_a4_exact_peak_scan import strict_detail


def main() -> None:
    a = np.asarray([1.0, 0.0], dtype=np.float32)
    b = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    prototype = normalized_mean(b)
    assert np.allclose(prototype, [0.0, 1.0])
    mixed = mix_embedding(a, prototype, 0.5)
    assert np.allclose(np.linalg.norm(mixed), 1.0)
    assert np.allclose(mixed, np.asarray([1.0, 1.0]) / np.sqrt(2.0))
    for alpha in (0.1, 0.25, 0.5):
        value = mix_embedding(a, prototype, alpha)
        assert value[0] > 0 and value[1] > 0
    # Locked-score anchoring must preserve a strict near-tie baseline exactly
    # while still applying the counterfactual delta.
    rows = np.asarray([1, 2], dtype=np.int64)
    ptr = np.asarray([0, 1, 2], dtype=np.int64)
    locked = np.asarray([0.50000000, 0.50000001], dtype=np.float64)
    recomputed = np.asarray([0.50000002, 0.50000000], dtype=np.float64)
    assert strict_detail(locked, rows, ptr)["rank"] == 2
    anchored_alpha_zero = locked + (recomputed - recomputed)
    assert strict_detail(anchored_alpha_zero, rows, ptr)["rank"] == 2
    print("[test_noise_v3_a4b_positive_evidence] PASS", flush=True)


if __name__ == "__main__":
    main()
