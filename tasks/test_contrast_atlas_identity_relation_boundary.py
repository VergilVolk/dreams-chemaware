from pathlib import Path

import numpy as np
import pandas as pd

from audit_contrast_atlas_identity_relation_boundary import attach_mass, read_precursors
from build_contrast_atlas_mass_compatible_identity_graph import directed_mass_top1


def test_read_precursor_uses_feature_then_scans_then_title(tmp_path: Path) -> None:
    path = tmp_path / "x.mgf"
    path.write_text(
        "BEGIN IONS\nTITLE=t1\nSCANS=1\nFEATURE_ID=f1\nPEPMASS=100\nEND IONS\n"
        "BEGIN IONS\nTITLE=t2\nSCANS=2\nPEPMASS=200\nEND IONS\n"
        "BEGIN IONS\nTITLE=t3\nPEPMASS=300\nEND IONS\n",
        encoding="utf-8",
    )
    assert read_precursors(path, "S") == {
        "S::f1": 100.0,
        "S::2": 200.0,
        "S::t3": 300.0,
    }


def test_mass_top1_never_selects_high_cosine_outside_window() -> None:
    rows = directed_mass_top1(
        np.asarray(["q"]), np.asarray([100.0]), np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray(["near", "far"]), np.asarray([100.0005, 110.0]),
        np.asarray([[0.8, 0.6], [1.0, 0.0]], dtype=np.float32),
        ppm=10.0, minimum_cosine=0.7,
    )
    assert len(rows) == 1
    assert rows[0][1] == "near"
    assert rows[0][2] == np.float32(0.8)


def test_attach_mass_keeps_missing_mass_explicit() -> None:
    frame = pd.DataFrame({"left": ["a", "missing"], "right": ["b", "b"]})
    result = attach_mass(frame, {"a": 100.0, "b": 100.0005}, "left", "right")
    assert bool(result.loc[0, "mass_compatible_5ppm"])
    assert pd.isna(result.loc[1, "precursor_delta_ppm"])
    assert not bool(result.loc[1, "mass_compatible_5ppm"])
