from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from evaluate_mtbls13729_reverse_probe_raw import (
    load_selected_mgf, nearest_target, parse_feature_id_set,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.mgf"
        path.write_text(
            "BEGIN IONS\nPEPMASS=100\n10 1\n20 2\nEND IONS\n"
            "BEGIN IONS\nPEPMASS=200\n30 3\n40 4\nEND IONS\n",
            encoding="utf-8",
        )
        selected = load_selected_mgf(path, {1})
        assert set(selected) == {1}
        assert selected[1].shape == (2, 2)
    feature, dppm, drt = nearest_target(
        100.0002, 50.0,
        np.asarray([99.0, 100.0, 100.0003]),
        np.asarray([50.0, 80.0, 52.0]),
        np.asarray([1, 2, 3]),
        10.0, 20.0,
    )
    assert feature == 3 and dppm < 2 and drt == 2.0
    feature, _, _ = nearest_target(
        100.0, 50.0, np.asarray([101.0]), np.asarray([50.0]), np.asarray([1]), 10.0, 20.0
    )
    assert feature is None
    assert parse_feature_id_set("859.0;860") == {859, 860}
    print("[test_mtbls13729_reverse_probe_raw] PASS")


if __name__ == "__main__":
    main()
