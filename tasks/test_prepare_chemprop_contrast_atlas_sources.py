from pathlib import Path

import numpy as np
import pandas as pd

from prepare_chemprop_contrast_atlas_sources import (
    nonempty_mgf_ids,
    standardize_effects,
)


def test_nonempty_mgf_ids_excludes_empty_spectrum(tmp_path: Path):
    path = tmp_path / "x.mgf"
    path.write_text(
        "BEGIN IONS\nSCANS=1\nPEPMASS=100\nEND IONS\n"
        "BEGIN IONS\nSCANS=2\nPEPMASS=101\n50 10\nEND IONS\n",
        encoding="utf-8",
    )
    ids, total, nonempty = nonempty_mgf_ids(path)
    assert ids == {"2"}
    assert total == 2
    assert nonempty == 1


def test_standardize_effects_uses_only_nonempty_ms2_and_floors_zero_se():
    quant = pd.DataFrame({"row ID": [1, 2, 3]})
    components = {
        "DrugA": {
            "response_signed": (
                np.asarray([1.0, 2.0, 3.0]),
                np.asarray([0.1, 0.0, 0.3]),
            )
        }
    }
    effects, report = standardize_effects(
        "S", quant, components, np.asarray([0.2, 0.4, 0.6]), {"2", "3"}
    )
    assert effects["entity_id"].tolist() == ["2", "3"]
    assert effects["contrast_id"].tolist() == ["drug_druga", "drug_druga"]
    assert effects["se"].tolist() == [1e-8, 0.3]
    assert report["zero_se_values_floored"] == 1
