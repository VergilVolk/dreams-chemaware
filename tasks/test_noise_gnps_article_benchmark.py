#!/usr/bin/env python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from noise_gnps_article_spectral_scores import (
    apply_frozen_p2b,
    cosine_greedy,
    modified_cosine,
    prepare_spectrum,
    weighted_entropy_similarity,
)


def spectrum(mz, intensity):
    return prepare_spectrum(np.asarray([mz, intensity], dtype=np.float32), 100)


def test_classical_scores() -> None:
    left = spectrum([50.0, 75.0, 100.0], [1.0, 0.5, 0.25])
    same = spectrum([50.001, 75.001, 100.001], [1.0, 0.5, 0.25])
    shifted = spectrum([51.0, 76.0, 101.0], [1.0, 0.5, 0.25])
    far = spectrum([20.0, 30.0, 40.0], [1.0, 0.5, 0.25])
    assert cosine_greedy(left, same) > 0.999
    assert weighted_entropy_similarity(left, same) > 0.999
    assert cosine_greedy(left, far) == 0.0
    assert modified_cosine(left, 151.0, shifted, 152.0) > 0.999
    assert cosine_greedy(left, shifted) == 0.0


def test_frozen_p2b_gate() -> None:
    graph = SimpleNamespace(
        n_queries=1,
        query_ptr=np.asarray([0, 2], dtype=np.int64),
        molecule_ptr=np.asarray([0, 1, 2], dtype=np.int64),
    )
    dreams = np.asarray([0.70, 0.80], dtype=np.float32)
    sqrt = np.asarray([0.60, 0.20], dtype=np.float32)
    entropy = np.asarray([0.90, 0.10], dtype=np.float32)
    neutral = np.asarray([0.95, 0.10], dtype=np.float32)
    result = apply_frozen_p2b(graph, dreams, sqrt, entropy, neutral)
    # The frozen "absolute" normalization maps DreaMS through (x + 1) / 2
    # before the registered weights are applied.
    expected = 0.10 * (dreams.astype(np.float64) + 1.0) / 2.0 \
        + 0.10 * entropy + 0.80 * neutral
    assert np.allclose(result.pair, expected)
    assert int(np.argmax(result.molecule)) == 0

    tied = apply_frozen_p2b(
        graph, dreams,
        np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5]),
    )
    assert np.array_equal(tied.pair, dreams)


def main() -> None:
    test_classical_scores()
    test_frozen_p2b_gate()
    print("[test_noise_gnps_article_benchmark] PASS tests=2")


if __name__ == "__main__":
    main()
