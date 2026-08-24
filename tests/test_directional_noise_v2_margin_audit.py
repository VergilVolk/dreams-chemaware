import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_directional_noise_v2_margin_audit",
    ROOT / "tasks/run_directional_noise_v2_margin_audit.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_positive_selection_prefers_different_condition():
    query = pd.Series({"row": 1, "condition": "Orbitrap|ce-20"})
    peers = pd.DataFrame({
        "row": [1, 2, 3],
        "condition": ["Orbitrap|ce-20", "Orbitrap|ce-20", "QTOF|ce-20"],
    })
    assert MODULE.choose_positive(query, peers, seed=7) == 3


def test_hardest_negative_aggregates_spectra_by_identity():
    precursor = np.asarray([100.0, 100.0001, 100.0002, 100.0003])
    adduct = np.asarray(["H", "H", "H", "H"], dtype=object)
    ik14 = np.asarray(["Q", "A", "A", "B"], dtype=object)
    rows = np.arange(4, dtype=np.int64)
    mass_index = MODULE.build_mass_index(rows, precursor, adduct)
    embeddings = np.asarray([[1, 0], [.8, .6], [.99, .01], [.9, .1]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = {i: i for i in range(4)}
    assert MODULE.hardest_negative(0, precursor, adduct, ik14, mass_index, embeddings, index, 10.0) == 2
