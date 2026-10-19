from __future__ import annotations

import pandas as pd
import pytest

from build_contrast_atlas_control_worlds import matched_known_keys, shuffled_unresolved_loadings, validate_covariates
from test_contrast_atlas_core import fixtures


def measurement_covariates(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in effects.reset_index(drop=True).iterrows():
        rows.append({
            "study_id": row.study_id, "contrast_id": row.contrast_id, "entity_id": row.entity_id,
            "log_intensity": float(index % 4), "detection_rate": 0.8 + 0.01 * (index % 3),
            "qc_cv": 0.1 + 0.01 * (index % 2), "missingness": 0.0,
            "ms2_peak_count": 10.0 + index % 4, "measurement_bin": "shared",
        })
    return pd.DataFrame(rows)


def test_control_world_matching_and_unresolved_derangement() -> None:
    effects, loadings, _ = fixtures()
    covariates = measurement_covariates(effects)
    covariates = validate_covariates(
        covariates,
        ["log_intensity", "detection_rate", "qc_cv", "missingness", "ms2_peak_count"],
        ["measurement_bin"],
    )
    keys, ledger = matched_known_keys(
        effects, covariates,
        ["log_intensity", "detection_rate", "qc_cv", "missingness", "ms2_peak_count"],
    )
    assert keys
    assert len(ledger) >= 3
    shuffled, shuffle_ledger = shuffled_unresolved_loadings(
        effects, loadings, covariates, ["measurement_bin"], seed=7,
    )
    assert len(shuffled) == len(loadings)
    assert (shuffle_ledger["source_entity_id"] != shuffle_ledger["target_entity_id"]).all()
    assert not shuffled.duplicated(["study_id", "contrast_id", "entity_id", "pattern_id"]).any()


def test_shuffle_rejects_singleton_unresolved_stratum() -> None:
    effects, loadings, _ = fixtures()
    covariates = measurement_covariates(effects)
    unresolved_ids = set(effects.loc[effects["identity_status"] == "unresolved_entity", "entity_id"])
    covariates.loc[covariates["entity_id"].isin(unresolved_ids), "measurement_bin"] = covariates.loc[
        covariates["entity_id"].isin(unresolved_ids), "entity_id"
    ]
    with pytest.raises(RuntimeError, match="fewer than two"):
        shuffled_unresolved_loadings(effects, loadings, covariates, ["measurement_bin"], seed=7)
