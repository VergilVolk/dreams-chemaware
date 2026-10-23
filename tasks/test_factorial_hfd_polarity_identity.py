from pathlib import Path

import numpy as np

from partition_factorial_hfd_panel_identity import (
    infer_polarity,
    negative_library_partition,
)


def test_polarity_is_unique() -> None:
    assert infer_polarity("LC-MS_METHOD1_POS_1.txt") == "positive"
    assert infer_polarity("LC-MS_METHOD3_NEG_1.txt") == "negative"


def test_negative_partition_excludes_non_m_h_adducts(tmp_path: Path) -> None:
    library = tmp_path / "negative.mgf"
    library.write_text(
        "BEGIN IONS\nPEPMASS=100\nADDUCT=[M-H]-\nINCHIKEY=AAAAAAAAAAAAAA-UHFFFAOYSA-N\n"
        "NAME=strict\nSOURCE=test\n10 1\n20 1\n30 1\n40 1\n50 1\n60 1\nEND IONS\n\n"
        "BEGIN IONS\nPEPMASS=100\nADDUCT=[M+Cl]-\nINCHIKEY=BBBBBBBBBBBBBB-UHFFFAOYSA-N\n"
        "NAME=wrong_adduct\nSOURCE=test\n11 1\n21 1\n31 1\n41 1\n51 1\n61 1\nEND IONS\n",
        encoding="utf-8",
    )
    query = {"q1": {"precursor_mz": 100.0, "peaks": np.asarray([
        [10, 1], [20, 1], [30, 1], [40, 1], [50, 1], [60, 1],
    ], dtype=np.float32)}}
    result, counts = negative_library_partition(
        query, ["q1"], library, 10.0, 0.02, 0.70, 6,
    )
    assert counts == {"all_negative_records": 2, "strict_m_h_records": 1}
    assert bool(result.loc[0, "library_supported"])
    assert result.loc[0, "library_adduct"] == "[M-H]-"
