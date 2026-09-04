import numpy as np


def test_plain_array_score_reconstruction() -> None:
    x = np.asarray([[1.0, 3.0], [2.0, 7.0]])
    mean = np.asarray([1.5, 5.0])
    scale = np.asarray([0.5, 2.0])
    coefficient = np.asarray([0.25, -0.75])
    manual = ((x - mean) / scale) @ coefficient
    expected = np.asarray([0.5, -0.5])
    assert np.allclose(manual, expected)
