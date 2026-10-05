#!/usr/bin/env python
"""Unit checks for B18 contextual pair construction."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b18_contextual_pairwise_action import pair_vector


def main() -> None:
    left = np.asarray([3.0, 1.0])
    right = np.asarray([1.0, 2.0])
    context = np.asarray([0.25, 4.0])
    forward = pair_vector(left, right, context)
    reverse = pair_vector(right, left, context)
    assert np.allclose(forward[:2], -reverse[:2])
    assert np.allclose(forward[2:4], reverse[2:4])
    assert np.allclose(forward[4:6], reverse[4:6])
    assert np.allclose(forward[-2:], context)
    assert len(forward) == 8
    print("[test_bioaware_b18_contextual_pairwise_action] PASS")


if __name__ == "__main__":
    main()
