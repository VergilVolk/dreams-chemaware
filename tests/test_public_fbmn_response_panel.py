from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


TASKS = Path(__file__).resolve().parents[1] / "tasks"
if str(TASKS) not in sys.path:
    sys.path.insert(0, str(TASKS))

from build_public_fbmn_response_panel import (  # noqa: E402
    build_design,
    build_identity_partition,
    mgf_feature_ids,
)


def test_design_collapses_files_to_frozen_biological_units() -> None:
    metadata = pd.DataFrame({
        "filename": ["a1", "a2", "b1", "b2", "blank"],
        "sample": [" A", "A ", "B", " B ", "blank"],
        "group": ["KO", "KO ", "WT", " WT", "media"],
    })
    source = {"filename_column": "filename", "biological_unit_column": "sample"}
    contrast = {
        "object_id": "ko_vs_wt",
        "levels": {
            "ko": {"column": "group", "values": ["KO"]},
            "wt": {"column": "group", "values": ["WT"]},
        },
        "expected_units_per_level": {"ko": 1, "wt": 1},
    }
    design = build_design(metadata, source, contrast)
    assert len(design) == 4
    assert design.biological_unit_id.nunique() == 2
    assert set(design.level) == {"ko", "wt"}


def test_design_rejects_biological_unit_crossing_levels() -> None:
    metadata = pd.DataFrame({
        "filename": ["a", "b"], "sample": ["same", "same"], "group": ["KO", "WT"]
    })
    source = {"filename_column": "filename", "biological_unit_column": "sample"}
    contrast = {
        "object_id": "bad",
        "levels": {
            "ko": {"column": "group", "values": ["KO"]},
            "wt": {"column": "group", "values": ["WT"]},
        },
        "expected_units_per_level": {"ko": 1, "wt": 1},
    }
    with pytest.raises(ValueError, match="cross contrast levels"):
        build_design(metadata, source, contrast)


def test_mgf_feature_id_falls_back_to_scans(tmp_path: Path) -> None:
    path = tmp_path / "x.mgf"
    path.write_text(
        "BEGIN IONS\nFEATURE_ID=f1\nSCANS=1\n100 2\nEND IONS\n"
        "BEGIN IONS\nSCANS=f2\n200 3\nEND IONS\n"
        "BEGIN IONS\nSCANS=empty\nEND IONS\n",
        encoding="utf-8",
    )
    ids, blocks, missing = mgf_feature_ids(path)
    assert ids == {"f1", "f2"}
    assert blocks == 3
    assert missing == 0


def test_strict_library_partition_does_not_promote_suspects(tmp_path: Path) -> None:
    quant = pd.DataFrame({"row ID": [1, 2, 3], "row m/z": [100, 200, 300]})
    library = pd.DataFrame({
        "#Scan#": ["1", "2", "3"],
        "Compound_Name": ["direct", "Suspect related compound", "no planar"],
        "LibraryQualityString": ["Gold", "Gold", "Bronze"],
        "InChIKey-Planar": ["ABCDEFGHIJKLMN", "ABCDEFGHIJKLMN", ""],
        "MQScore": [0.9, 0.8, 0.7],
        "SharedPeaks": [10, 9, 8],
        "Adduct": ["M+H", "M+H", "M+H"],
        "IonMode": ["Positive", "Positive", "Positive"],
    })
    path = tmp_path / "db.tsv"
    library.to_csv(path, sep="\t", index=False)
    result = build_identity_partition(quant, path)
    assert result.identity_status.tolist() == [
        "library_supported", "unresolved_entity", "unresolved_entity"
    ]
