from __future__ import annotations

import numpy as np
import pytest

from partition_contrast_atlas_open_source_identity import encoder_records_to_wse_spectra


def test_encoder_peak_axis_is_transposed_for_wse() -> None:
    records = [{
        "precursor_mz": 123.4,
        "peaks": np.asarray([[50.0, 75.0, 100.0], [1.0, 0.5, 0.25]], dtype=np.float32),
    }]
    converted = encoder_records_to_wse_spectra(["q1"], records)["q1"]
    assert converted["peaks"].shape == (3, 2)
    assert np.allclose(converted["peaks"][:, 0], [50.0, 75.0, 100.0])
    assert converted["precursor_mz"] == 123.4


def test_encoder_peak_axis_fails_closed() -> None:
    with pytest.raises(ValueError, match="2 x N"):
        encoder_records_to_wse_spectra(
            ["q1"], [{"precursor_mz": 123.4, "peaks": np.zeros((3, 2))}]
        )
