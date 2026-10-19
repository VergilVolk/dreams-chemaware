from __future__ import annotations

import numpy as np
import pandas as pd

from build_msv000096214_contrast_atlas_records import build_records


def test_build_records_uses_biological_samples_and_positive_se() -> None:
    samples = [f"w{i}" for i in range(5)] + [f"m{i}" for i in range(5)]
    matrix = pd.DataFrame(
        [
            [10, 11, 9, 10, 12, 20, 19, 21, 22, 18],
            [8, 9, 10, 11, 12, 4, 5, 6, 7, 8],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        index=pd.Index([0, 1, 2], name="group_id"),
        columns=samples,
        dtype=float,
    )
    groups = pd.DataFrame({"group_id": [0, 1, 2], "mz": [100.0, 200.0, 300.0], "rt_sec": [10.0, 20.0, 30.0]})
    mapping = {sample: ("WT_Col0" if sample.startswith("w") else "mutant") for sample in samples}
    records = build_records(matrix, groups, mapping, "positive", "study", "source")
    assert {0, 1}.issubset(set(records["group_id"]))
    assert (records["se"] > 0).all()
    assert set(records["n_mutant"]) == {5}
    assert set(records["n_reference"]) == {5}
    assert records.set_index("group_id").loc[0, "beta"] > 0
    assert records.set_index("group_id").loc[1, "beta"] < 0
    assert np.isfinite(records["beta"]).all()
