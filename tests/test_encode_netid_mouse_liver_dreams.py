from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "encode_netid_mouse_liver_dreams.py"
SPEC = importlib.util.spec_from_file_location("encode_netid_mouse_liver_dreams", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_direct_script_declares_repository_import_path() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source


def test_unpack_records_preserves_offsets_and_filters_missing() -> None:
    class Cache(dict):
        pass

    cache = Cache(
        peak_offsets=np.array([0, 0, 2, 5]),
        precursor_mz=np.array([100.0, 200.0, 300.0]),
        fragment_mz=np.array([10, 20, 30, 40, 50], dtype=np.float32),
        fragment_intensity=np.array([1, 2, 3, 4, 5], dtype=np.float32),
    )
    records, selected = MODULE.unpack_records(cache, minimum_peaks=2)
    assert selected.tolist() == [1, 2]
    assert [record["precursor_mz"] for record in records] == [200.0, 300.0]
    assert records[0]["peaks"].shape == (2, 2)
    assert records[1]["peaks"].shape == (2, 3)
