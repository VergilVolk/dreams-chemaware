"""Unit checks for C2-B0 contextual token-pair summaries."""
from __future__ import annotations

import numpy as np

from build_noise_v3_c2b_token_pair_cache import TOKEN_FEATURES, token_pair_summary


def main() -> None:
    token = np.zeros((4, 3), np.float32)
    token[:3] = np.eye(3, dtype=np.float32)
    mz = np.asarray([100.0, 150.0, 200.0, 0.0], np.float32)
    intensity = np.asarray([1.0, 0.25, 0.1, 0.0], np.float32)
    valid = mz > 0
    same = token_pair_summary(token, token, mz, mz, intensity, intensity, valid, valid, 0.02)
    shifted = token_pair_summary(token, token, mz, mz + 1.0, intensity, intensity, valid, valid, 0.02)
    assert same.shape == (len(TOKEN_FEATURES),)
    assert np.isclose(same[TOKEN_FEATURES.index("token_cosine_weighted")], 1.0)
    assert np.isclose(same[TOKEN_FEATURES.index("token_match_fraction_min")], 1.0)
    assert np.allclose(shifted, 0.0)
    conflict = token.copy(); conflict[:3] *= -1
    opposite = token_pair_summary(token, conflict, mz, mz, intensity, intensity, valid, valid, 0.02)
    assert opposite[TOKEN_FEATURES.index("token_conflict_weighted")] > 1.9
    print("[test_noise_v3_c2b_token_pair_cache] PASS")


if __name__ == "__main__": main()
