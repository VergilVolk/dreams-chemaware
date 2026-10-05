"""Exact representable Noise actions for the unmodified DreaMS preprocessor."""
from __future__ import annotations

import numpy as np

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.spectra import MSnSpectrum


NATIVE_ACTION_SPECTRUM_ADAPTER_VERSION = "unmodified_native_representable_v4"


def action_fragment_profile(tensor: np.ndarray) -> dict[str, int | bool]:
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.shape != (101, 2) or not np.isfinite(tensor).all():
        raise RuntimeError("Noise action tensor does not have native 101-token geometry")
    if abs(float(tensor[0, 1]) - 1.1) > 1e-6:
        raise RuntimeError("Noise action precursor intensity is not the official 1.1")
    fragments = tensor[1:][tensor[1:, 0] > 0]
    positive = int(np.sum(fragments[:, 1] > 0))
    return {
        "real_fragment_tokens": int(len(fragments)),
        "positive_fragment_intensities": positive,
        "native_action_view_representable": bool(positive > 0),
    }


def make_action_spectrum(tensor: np.ndarray) -> MSnSpectrum:
    """Create a spectrum whose native preprocessing preserves model tokens.

    Partly zeroed actions retain their exact zero intensities; the native trim
    operation retains all input tokens when there are at most 100 fragments,
    and relative normalization preserves zero exactly. An action with no
    positive fragment intensity is not representable by the published native
    preprocessor because its relative-intensity normalization divides by zero.
    Such an action must contribute only its exact measured clean-boundary
    triplet; fabricating a sentinel or changing the preprocessor is forbidden.
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    profile = action_fragment_profile(tensor)
    peak_list = tensor[1:][tensor[1:, 0] > 0].copy()
    if not profile["native_action_view_representable"]:
        raise RuntimeError(
            "all-zero action is not representable by unmodified native "
            "relative-intensity preprocessing"
        )
    return MSnSpectrum(
        peak_list=peak_list.T,
        precursor_mz=float(tensor[0, 0]),
        precursor_charge=1,
        assert_is_valid=False,
    )


def native_action_model_input(
    tensor: np.ndarray, preprocessor: SpectrumPreprocessor,
) -> np.ndarray:
    if type(preprocessor) is not SpectrumPreprocessor:
        raise RuntimeError("native Noise actions require the unmodified SpectrumPreprocessor")
    spectrum = make_action_spectrum(tensor)
    output = np.ascontiguousarray(preprocessor(
        spectrum.get_peak_list(),
        prec_mz=spectrum.get_precursor_mz(),
        high_form=False,
        augment=False,
    ).astype(np.float32, copy=False))
    return output
