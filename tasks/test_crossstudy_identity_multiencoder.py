#!/usr/bin/env python3
"""Small numerical contracts for complete identity-candidate scoring."""
from __future__ import annotations

import numpy as np

from score_crossstudy_identity_multiencoder import matched_peaks, minmax


def main() -> None:
    assert np.allclose(minmax(np.asarray([2.0, 4.0, 3.0])), [0.0, 1.0, 0.5])
    assert np.allclose(minmax(np.asarray([7.0, 7.0])), [0.0, 0.0])
    left = np.asarray([[100.000, 1.0], [150.000, 2.0], [200.000, 3.0]])
    right = np.asarray([[99.991, 1.0], [150.019, 2.0], [175.000, 3.0]])
    assert matched_peaks(left, right, 0.02) == 2
    print("PASS: cross-study identity multi-encoder contracts", flush=True)


if __name__ == "__main__":
    main()
