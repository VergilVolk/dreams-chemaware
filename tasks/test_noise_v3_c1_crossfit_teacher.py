"""Unit tests for support-disjoint C1 teacher helpers."""
from __future__ import annotations

import numpy as np

from build_noise_v3_c1_crossfit_teacher_space import mix


def main() -> None:
    clean = np.asarray([1.0, 0.0], dtype=np.float32)
    teacher = np.asarray([0.0, 1.0], dtype=np.float32)
    value = mix(clean, teacher, 0.25)
    assert np.isclose(np.linalg.norm(value), 1.0)
    assert value[0] > value[1] > 0
    print("[test_noise_v3_c1_crossfit_teacher] PASS", flush=True)


if __name__ == "__main__":
    main()
