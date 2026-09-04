from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tasks.export_kgmn_feature_dreams_embeddings import (
    read_feature_spectra,
    select_feature_names,
    validate_frame,
)


def test_mgf_title_is_reconciled_to_ms1_feature_name(tmp_path: Path) -> None:
    path = tmp_path / "features.mgf"
    path.write_text(
        "BEGIN IONS\nTITLE=F001\nPEPMASS=123.45\n50 10\n80 30\nEND IONS\n"
        "BEGIN IONS\nTITLE=F002\nPEPMASS=234.56\n70 20\n90 40\nEND IONS\n",
        encoding="utf-8",
    )
    frame = read_feature_spectra(path)
    names, column = select_feature_names(frame, {"F001", "F002"})
    validate_frame(frame, names)
    assert column == "name"
    assert names.tolist() == ["F001", "F002"]


def test_msp_name_is_reconciled_to_ms1_feature_name(tmp_path: Path) -> None:
    path = tmp_path / "features.msp"
    path.write_text(
        "NAME: F001\nPRECURSORMZ: 123.45\nNum Peaks: 1\n50 10\n\n",
        encoding="utf-8",
    )
    frame = read_feature_spectra(path)
    names, column = select_feature_names(frame, {"F001"})
    validate_frame(frame, names)
    assert column == "name"
    assert names.tolist() == ["F001"]


def test_identifier_mapping_fails_closed_when_title_is_not_ms1_name() -> None:
    frame = pd.DataFrame(
        {
            "name": ["display title"],
            "feature_id": ["F001"],
            "precursor_mz": [123.45],
            "spectrum": [[[50.0], [10.0]]],
        }
    )
    names, column = select_feature_names(frame, {"F001"})
    assert column == "feature_id"
    assert names.tolist() == ["F001"]


def test_identifier_mapping_rejects_ambiguous_exact_columns() -> None:
    frame = pd.DataFrame(
        {
            "name": ["F001"],
            "feature_id": ["F001"],
            "precursor_mz": [123.45],
            "spectrum": [[[50.0], [10.0]]],
        }
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        select_feature_names(frame, {"F001"})


def test_unsupported_author_ms2_format_requires_explicit_conversion(tmp_path: Path) -> None:
    path = tmp_path / "features.cef"
    path.write_text("placeholder", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires MSP or MGF"):
        read_feature_spectra(path)
