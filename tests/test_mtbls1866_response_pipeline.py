from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tasks.estimate_mtbls1866_covid_response import (
    acquisition_order_audit,
    group_moments,
    strict_bool,
)


def test_group_moments_preserve_missingness() -> None:
    values = np.asarray([[1.0, 3.0, np.nan], [2.0, np.nan, np.nan]])
    count, mean, variance = group_moments(values)
    assert count.tolist() == [2, 1]
    assert mean.tolist() == [2.0, 2.0]
    assert variance[0] == 2.0
    assert np.isnan(variance[1])


def test_strict_bool_does_not_treat_false_string_as_true() -> None:
    observed = strict_bool(pd.Series(["True", "False", "1", "0"]), "flag")
    assert observed.tolist() == [True, False, True, False]


def test_acquisition_order_audit_detects_complete_group_separation(
    tmp_path: Path,
) -> None:
    measurement = tmp_path / "measurement"
    measurement.mkdir()
    names = [str(index) for index in range(109)]
    times = pd.date_range("2020-01-01", periods=109, freq="min", tz="UTC")
    pd.DataFrame(
        {"assay_name": names, "acquisition_start_time": times.astype(str)}
    ).to_csv(
        measurement / "phenotype_blind_acquisition_times.tsv",
        sep="\t",
        index=False,
    )
    design = pd.DataFrame(
        {
            "Sample Name": names,
            "group": ["HEALTHY"] * 26 + ["COVID-19"] * 83,
        }
    )
    audit = acquisition_order_audit(measurement, design)
    assert audit["complete"] is True
    assert audit["order_auc_covid_after_healthy"] == 1.0
    assert audit["perfect_temporal_separation"] is True
    assert audit["group_transitions"] == 1


def test_acquisition_order_audit_abstains_when_timestamps_missing(
    tmp_path: Path,
) -> None:
    measurement = tmp_path / "measurement"
    measurement.mkdir()
    names = [str(index) for index in range(109)]
    pd.DataFrame(
        {"assay_name": names, "acquisition_start_time": [""] * 109}
    ).to_csv(
        measurement / "phenotype_blind_acquisition_times.tsv",
        sep="\t",
        index=False,
    )
    design = pd.DataFrame(
        {
            "Sample Name": names,
            "group": ["COVID-19"] * 83 + ["HEALTHY"] * 26,
        }
    )
    audit = acquisition_order_audit(measurement, design)
    assert audit["complete"] is False
    assert audit["parsed_start_times"] == 0
    assert audit["order_auc_covid_after_healthy"] is None

