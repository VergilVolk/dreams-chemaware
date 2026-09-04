from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tasks.export_kgmn_200std_dreams_embeddings import validate_msp_frame


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["M100T1", "M101T2"],
            "precursor_mz": [100.0, 101.0],
            "spectrum": [
                np.asarray([[40.0, 50.0], [1.0, 2.0]]),
                np.asarray([[41.0], [3.0]]),
            ],
        }
    )


def test_validate_msp_frame_accepts_unique_finite_spectra() -> None:
    validate_msp_frame(valid_frame())


def test_validate_msp_frame_rejects_duplicate_feature_names() -> None:
    frame = valid_frame()
    frame.loc[1, "name"] = frame.loc[0, "name"]
    with pytest.raises(RuntimeError, match="unique"):
        validate_msp_frame(frame)


def test_validate_msp_frame_rejects_bad_spectrum_shape() -> None:
    frame = valid_frame()
    frame.at[1, "spectrum"] = np.asarray([1.0, 2.0])
    with pytest.raises(RuntimeError, match="invalid 200STD spectrum"):
        validate_msp_frame(frame)
