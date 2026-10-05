#!/usr/bin/env python
"""Numerical unit tests for A2."""
from __future__ import annotations

import base64

import numpy as np

from pilot_reverse_context_joint_a2 import _rate, _tanimoto


def main() -> None:
    matrix = np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
    query = np.array([1, 0, 1, 0], dtype=np.float32)
    similarity = _tanimoto(query, matrix)
    assert np.allclose(similarity, [1.0, 1.0 / 3.0])
    rate = _rate(np.array([2.0, 0.0]), np.array([10.0, 10.0]), np.array([0.1, 0.2]), 5.0)
    assert np.all(rate > 0) and np.all(rate < 1)
    assert base64.b64decode(base64.b64encode(b"x")) == b"x"
    print("[test_pilot_reverse_context_joint_a2] PASS")


if __name__ == "__main__":
    main()
