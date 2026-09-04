"""Small deterministic tests for the B1-C0 audit helpers."""
from __future__ import annotations

import numpy as np

from audit_noise_v3_a4b_b1_error_space import displacement_alignment, jaccard


def main() -> None:
    mask = np.asarray([True, True, False])
    a = np.asarray([True, False, True])
    b = np.asarray([True, True, False])
    assert np.isclose(jaccard(a, b, mask), 0.5)
    clean = np.asarray([1.0, 0.0])
    target = np.asarray([0.0, 1.0])
    same = np.asarray([0.0, 1.0])
    opposite = np.asarray([2.0, -1.0])
    _, _, alignment = displacement_alignment(clean, target, same)
    assert np.isclose(alignment, 1.0)
    _, _, alignment = displacement_alignment(clean, target, opposite)
    assert np.isclose(alignment, -1.0)
    print("[test_noise_v3_a4b_b1_error_space] PASS", flush=True)


if __name__ == "__main__":
    main()
